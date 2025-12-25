import io, json, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils_io import read_tabular, list_numeric_cols
from cgc_engine import compute_cgc, compute_conditional_dcgc
from sandbox_generator import make_sandbox

APP_NAME = "Aliona Field Coupling Studio"
APP_VERSION = "v3.0 (Rebuild)"

st.set_page_config(page_title=APP_NAME, layout="wide")

def plot_curve(x, y, title, yname):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=yname))
    fig.update_layout(title=title, xaxis_title="Lag (samples)", yaxis_title=yname,
                      height=420, margin=dict(l=20,r=20,t=50,b=30))
    return fig

def zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, obj in files.items():
            if isinstance(obj, pd.DataFrame):
                z.writestr(name, obj.to_csv(index=False))
            elif isinstance(obj, (dict, list)):
                z.writestr(name, json.dumps(obj, indent=2))
            else:
                z.writestr(name, str(obj))
    buf.seek(0)
    return buf.getvalue()

st.title(APP_NAME)
st.caption(f"{APP_VERSION} • CGC/dCGC + conditional controls • alignment by index (no timestamp required)")

tabs = st.tabs(["Coupling Test", "Sandbox", "About"])

with tabs[0]:
    st.subheader("Upload A + B and run a coupling scan")
    c1, c2 = st.columns(2)
    with c1:
        upA = st.file_uploader("Dataset A", type=["csv","tsv","txt","dat"], key="A")
    with c2:
        upB = st.file_uploader("Dataset B", type=["csv","tsv","txt","dat"], key="B")

    if not upA or not upB:
        st.info("Use Sandbox to generate known-good datasets if your real files are messy.")
    else:
        try:
            dfA = read_tabular(upA)
            dfB = read_tabular(upB)
        except Exception as e:
            st.error(f"Parse error: {e}")
            st.stop()

        nA = list_numeric_cols(dfA)
        nB = list_numeric_cols(dfB)
        if not nA or not nB:
            st.error("No numeric columns detected. (Try a different file or export as CSV.)")
            st.stop()

        s1, s2, s3, s4 = st.columns([2,2,2,2])
        with s1:
            a_col = st.selectbox("A: signal column", options=nA, index=0)
        with s2:
            b_col = st.selectbox("B: signal column", options=nB, index=0)
        with s3:
            max_lag = st.number_input("Max lag (samples)", min_value=5, max_value=5000, value=300, step=5)
        with s4:
            step = st.number_input("Lag step", min_value=1, max_value=250, value=5, step=1)

        lags = list(range(-int(max_lag), int(max_lag)+1, int(step)))

        st.divider()
        o1, o2, o3 = st.columns([2,2,3])
        with o1:
            weight_window = st.number_input("Stability weight window", min_value=20, max_value=5000, value=200, step=10)
        with o2:
            run_cond = st.checkbox("Run conditional controls (cCGC)", value=True)
        with o3:
            suggested = [c for c in dfA.columns if any(k in c.lower() for k in ["humid","temp","power","vib"])]
            controls = st.multiselect("Control columns", options=sorted(set(dfA.columns).union(dfB.columns)),
                                      default=suggested[:4], disabled=(not run_cond))

        if st.button("Run", type="primary"):
            with st.spinner("Computing…"):
                res = compute_cgc(dfA, dfB, a_col=a_col, b_col=b_col, lags=lags, weight_window=int(weight_window))

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best |CGC| lag", f"{res.best_lag_cgc}")
            k2.metric("CGC@best", f"{res.best_cgc:.3f}" if np.isfinite(res.best_cgc) else "nan")
            k3.metric("Best |dCGC| lag", f"{res.best_lag_dcgc}")
            k4.metric("dCGC@best", f"{res.best_dcgc:.3f}" if np.isfinite(res.best_dcgc) else "nan")

            out = pd.DataFrame({"lag": res.lag_grid, "CGC": res.cgc, "dCGC": res.dcgc})
            st.plotly_chart(plot_curve(out["lag"], out["CGC"], "CGC vs lag", "CGC"), use_container_width=True)
            st.plotly_chart(plot_curve(out["lag"], out["dCGC"], "dCGC vs lag", "dCGC"), use_container_width=True)

            out_ccgc = None
            used_controls = []
            if run_cond and len(controls) > 0:
                with st.spinner("Computing conditional controls…"):
                    out_ccgc = compute_conditional_dcgc(dfA, dfB, a_col=a_col, b_col=b_col, control_cols=controls, lags=lags)
                    used_controls = out_ccgc.attrs.get("controls_used", [])
                st.subheader("Conditional / Controlled Coupling (cCGC)")
                st.caption("If cCGC stays high while raw CGC drops, coupling is less likely to be a shared-driver artifact.")
                st.dataframe(out_ccgc.sort_values(by="cCGC", key=lambda s: s.abs(), ascending=False).head(25),
                             use_container_width=True)

            summary = {
                "app": APP_NAME,
                "version": APP_VERSION,
                "A_file": upA.name, "B_file": upB.name,
                "A_col": a_col, "B_col": b_col,
                "max_lag": int(max_lag), "step": int(step),
                "best_lag_cgc": int(res.best_lag_cgc),
                "best_cgc": float(res.best_cgc) if np.isfinite(res.best_cgc) else None,
                "best_lag_dcgc": int(res.best_lag_dcgc),
                "best_dcgc": float(res.best_dcgc) if np.isfinite(res.best_dcgc) else None,
                "controls_used": used_controls,
            }
            files = {"results/cgc_table.csv": out, "results/summary.json": summary}
            if out_ccgc is not None:
                files["results/conditional_ccgc.csv"] = out_ccgc
            st.download_button("Download results ZIP", data=zip_bytes(files),
                               file_name="aliona_coupling_results.zip", mime="application/zip")

with tabs[1]:
    st.subheader("Sandbox generator (known-truth)")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Samples", min_value=500, max_value=200000, value=5000, step=500)
        true_lag = st.number_input("True lag (samples)", min_value=-2000, max_value=2000, value=30, step=1)
    with c2:
        noise = st.slider("Noise", 0.0, 1.0, 0.15, 0.01)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=1, step=1)
    with c3:
        conf = st.checkbox("Confounder (humidity drives both)", True)
        window = st.checkbox("Coupling window (humidity HIGH)", True)

    if st.button("Generate sandbox"):
        A, B, truth = make_sandbox(int(n), int(true_lag), float(noise), bool(conf), bool(window), int(seed))
        st.session_state["A"] = A
        st.session_state["B"] = B
        st.session_state["truth"] = truth

    if "A" in st.session_state:
        st.json(st.session_state["truth"])
        col1, col2 = st.columns(2)
        with col1:
            st.write("A preview (use A_signal)")
            st.dataframe(st.session_state["A"].head(25), use_container_width=True)
            st.download_button("Download sandbox_A.csv", st.session_state["A"].to_csv(index=False).encode("utf-8"),
                               "sandbox_A.csv", "text/csv")
        with col2:
            st.write("B preview (use B_signal)")
            st.dataframe(st.session_state["B"].head(25), use_container_width=True)
            st.download_button("Download sandbox_B.csv", st.session_state["B"].to_csv(index=False).encode("utf-8"),
                               "sandbox_B.csv", "text/csv")
        st.info("Next: go to Coupling Test and upload sandbox_A.csv + sandbox_B.csv. Pick A_signal and B_signal.")

with tabs[2]:
    st.subheader("What this is for (industrial/startup tone)")
    st.markdown("""
This app is meant to be **used by companies**, not researchers.

It answers three operational questions:
1) **Did a change in Signal A precede a change in Signal B?** (CGC)
2) **Which direction is more consistent (A→B vs B→A)?** (dCGC)
3) **Is that relationship still present after controlling for known environmental/process channels?** (cCGC)

Typical use cases:
- manufacturing diagnostics (process drift → defects),
- sensor drift monitoring,
- “controller readiness” checks (is there a stable lagged relationship you can exploit),
- environment coupling audits (humidity/temp/power/vibration as shared drivers).
""")
