import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

def _to_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)
    return (x - m) / s

def _diff(x: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(x, dtype=float), prepend=x[0])

def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x).rolling(w, min_periods=max(3, w//5)).std().to_numpy()
    med = np.nanmedian(s)
    if not np.isfinite(med):
        return np.ones_like(x)
    s[np.isnan(s)] = med
    return s

def _lag_align(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag == 0:
        return a.copy(), b.copy()
    if lag > 0:
        a2 = a[:-lag]
        b2 = b[lag:]
    else:
        k = -lag
        a2 = a[k:]
        b2 = b[:-k]
    n = min(len(a2), len(b2))
    return a2[:n], b2[:n]

def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[m]; y = y[m]; w = w[m]
    if len(x) < 10:
        return np.nan
    w = np.clip(w, 1e-6, None)
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    x = x - mx
    y = y - my
    cov = np.average(x*y, weights=w)
    vx = np.average(x*x, weights=w)
    vy = np.average(y*y, weights=w)
    return float(cov / np.sqrt(max(vx*vy, 1e-12)))

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 10 or np.nanstd(x)==0 or np.nanstd(y)==0:
        return np.nan
    return float(np.corrcoef(x, y)[0,1])

def partial_corr(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> float:
    n = min(len(a), len(b), Z.shape[0])
    a = a[:n]; b = b[:n]; Z = Z[:n,:]
    mask = np.isfinite(a) & np.isfinite(b) & np.all(np.isfinite(Z), axis=1)
    a = a[mask]; b = b[mask]; Z = Z[mask,:]
    if len(a) < 20:
        return np.nan
    X = np.column_stack([np.ones(len(a)), Z])
    beta_a, *_ = np.linalg.lstsq(X, a, rcond=None)
    beta_b, *_ = np.linalg.lstsq(X, b, rcond=None)
    ra = a - X @ beta_a
    rb = b - X @ beta_b
    return _corr(ra, rb)

@dataclass
class CGCResult:
    lag_grid: np.ndarray
    cgc: np.ndarray
    dcgc: np.ndarray
    best_lag_cgc: int
    best_cgc: float
    best_lag_dcgc: int
    best_dcgc: float

def compute_cgc(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    a_col: str,
    b_col: str,
    lags: List[int],
    weight_window: int = 200,
    a_weight_col: Optional[str] = None,
    b_weight_col: Optional[str] = None,
) -> CGCResult:
    A = _to_numeric(dfA, a_col)
    B = _to_numeric(dfB, b_col)
    dA = _z(_diff(A))
    dB = _z(_diff(B))

    n = min(len(dA), len(dB))
    dA = dA[:n]; dB = dB[:n]

    if a_weight_col and a_weight_col in dfA.columns:
        wa = 1.0 / (1.0 + np.abs(_z(_to_numeric(dfA, a_weight_col)[:n])))
    else:
        wa = 1.0 / (1.0 + _z(_rolling_std(dA, weight_window))**2)

    if b_weight_col and b_weight_col in dfB.columns:
        wb = 1.0 / (1.0 + np.abs(_z(_to_numeric(dfB, b_weight_col)[:n])))
    else:
        wb = 1.0 / (1.0 + _z(_rolling_std(dB, weight_window))**2)

    w = np.sqrt(np.clip(wa*wb, 1e-6, 1.0))

    lag_grid = np.array(list(lags), dtype=int)
    cgc = np.full_like(lag_grid, np.nan, dtype=float)
    dcgc = np.full_like(lag_grid, np.nan, dtype=float)

    for i, lag in enumerate(lag_grid):
        x, y = _lag_align(dA, dB, int(lag))
        ww, _ = _lag_align(w, w, int(lag))
        c = _weighted_corr(x, y, ww)
        cgc[i] = c

        x2, y2 = _lag_align(dB, dA, int(lag))
        ww2, _ = _lag_align(w, w, int(lag))
        c2 = _weighted_corr(x2, y2, ww2)
        dcgc[i] = c - c2 if np.isfinite(c) and np.isfinite(c2) else np.nan

    if np.all(~np.isfinite(cgc)):
        best_lag_cgc, best_cgc = 0, np.nan
    else:
        k = int(np.nanargmax(np.abs(cgc)))
        best_lag_cgc, best_cgc = int(lag_grid[k]), float(cgc[k])

    if np.all(~np.isfinite(dcgc)):
        best_lag_dcgc, best_dcgc = 0, np.nan
    else:
        k = int(np.nanargmax(np.abs(dcgc)))
        best_lag_dcgc, best_dcgc = int(lag_grid[k]), float(dcgc[k])

    return CGCResult(lag_grid=lag_grid, cgc=cgc, dcgc=dcgc,
                     best_lag_cgc=best_lag_cgc, best_cgc=best_cgc,
                     best_lag_dcgc=best_lag_dcgc, best_dcgc=best_dcgc)

def compute_conditional_dcgc(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    a_col: str,
    b_col: str,
    control_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    n = min(len(dfA), len(dfB))
    A = pd.to_numeric(dfA[a_col], errors="coerce").astype(float).values[:n]
    B = pd.to_numeric(dfB[b_col], errors="coerce").astype(float).values[:n]
    dA = pd.Series(A).diff().fillna(0.0).to_numpy()
    dB = pd.Series(B).diff().fillna(0.0).to_numpy()

    Zs = []
    used = []
    for c in control_cols:
        if c in dfA.columns:
            Zs.append(pd.to_numeric(dfA[c], errors="coerce").astype(float).values[:n])
            used.append(c)
        elif c in dfB.columns:
            Zs.append(pd.to_numeric(dfB[c], errors="coerce").astype(float).values[:n])
            used.append(c)
    Z = np.column_stack(Zs) if Zs else np.zeros((n, 0))

    rows = []
    for lag in lags:
        xa, yb = _lag_align(dA, dB, int(lag))
        if Z.shape[1]:
            zlag, _ = _lag_align(Z, Z, int(lag))
            pc_ab = partial_corr(xa, yb, zlag)
        else:
            pc_ab = _corr(xa, yb)

        xb, ya = _lag_align(dB, dA, int(lag))
        if Z.shape[1]:
            zlag2, _ = _lag_align(Z, Z, int(lag))
            pc_ba = partial_corr(xb, ya, zlag2)
        else:
            pc_ba = _corr(xb, ya)

        rows.append({"lag": int(lag), "pcorr_AB": pc_ab, "pcorr_BA": pc_ba, "cCGC": (pc_ab - pc_ba)})
    out = pd.DataFrame(rows)
    out.attrs["controls_used"] = used
    return out
