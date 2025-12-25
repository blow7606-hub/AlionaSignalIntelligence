import io
import pandas as pd

def read_tabular(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    # Try separators
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    # Whitespace fallback
    return pd.read_csv(io.BytesIO(raw), delim_whitespace=True, engine="python")

def list_numeric_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(20, int(0.1*len(df))):
            cols.append(c)
    return cols
