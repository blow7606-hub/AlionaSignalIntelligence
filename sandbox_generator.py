import numpy as np
import pandas as pd

def _tri_state(n: int, step: int, seed: int):
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n/step))
    s = rng.choice([0,1,2], size=blocks, p=[0.33,0.34,0.33])
    return np.repeat(s, step)[:n]

def _val(s, low, mid, high):
    return np.where(s==0, low, np.where(s==1, mid, high)).astype(float)

def _lag(x, k):
    if k==0: return x.copy()
    y = np.zeros_like(x, dtype=float)
    if k>0: y[k:] = x[:-k]
    else: y[:k] = x[-k:]
    return y

def make_sandbox(n=5000, true_lag=30, noise=0.15, confounder=True, coupling_window=True, seed=1):
    rng = np.random.default_rng(seed)
    hum_s = _tri_state(n, 250, seed+10)
    tmp_s = _tri_state(n, 200, seed+11)
    pwr_s = _tri_state(n, 150, seed+12)
    vib_s = _tri_state(n, 120, seed+13)

    hum = _val(hum_s, 0.2, 0.5, 0.85)
    tmp = _val(tmp_s, 0.25, 0.55, 0.9)
    pwr = _val(pwr_s, 0.15, 0.5, 0.95)
    vib = _val(vib_s, 0.1, 0.4, 0.9)

    t1 = np.linspace(0, 30*np.pi, n)
    t2 = np.linspace(0, 20*np.pi, n)
    A = 0.6*np.sin(t1) + 0.2*rng.normal(size=n)
    B = 0.4*np.cos(t2) + 0.2*rng.normal(size=n)

    if confounder:
        A += 1.2*(hum-hum.mean())
        B += 1.0*(hum-hum.mean())

    B += 1.5*_lag(pwr-pwr.mean(), true_lag)

    if coupling_window:
        gate = (hum_s==2).astype(float)  # HIGH humidity -> window open
        B += gate*1.8*_lag(pwr-pwr.mean(), true_lag)

    B += 0.4*_lag(tmp-tmp.mean(), 10)
    B += 0.2*_lag(vib-vib.mean(), 5)

    A += noise*rng.normal(size=n)
    B += noise*rng.normal(size=n)

    dfA = pd.DataFrame({
        "idx": np.arange(n),
        "A_signal": A,
        "humidity_state": hum_s,
        "temp_state": tmp_s,
        "power_state": pwr_s,
        "vibration_state": vib_s,
        "humidity": hum,
        "temp": tmp,
        "power": pwr,
        "vibration": vib,
    })
    dfB = pd.DataFrame({
        "idx": np.arange(n),
        "B_signal": B,
        "humidity_state": hum_s,
        "temp_state": tmp_s,
        "power_state": pwr_s,
        "vibration_state": vib_s,
        "humidity": hum,
        "temp": tmp,
        "power": pwr,
        "vibration": vib,
    })
    truth = {
        "true_lag_samples": int(true_lag),
        "primary_driver": "power",
        "confounder": bool(confounder),
        "coupling_window": "humidity_state==2 (HIGH)" if coupling_window else None
    }
    return dfA, dfB, truth
