"""Load UR3 CobotOps, segment the log, detect stop / grip-loss episodes, compute MTBF, MTTR, availability."""
from __future__ import annotations
import numpy as np
import pandas as pd

GAP_S = 30.0  # a sample gap longer than this splits the log into separate recording segments
JOINTS = range(6)


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Temperature_T0": "Temperature_J0"})
    df["t"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip().str.strip('"'), utc=True, format="ISO8601")
    df = df.sort_values("t").reset_index(drop=True)
    df["dt"] = df["t"].diff().dt.total_seconds()
    df["segment"] = (df["dt"].isna() | (df["dt"] > GAP_S)).cumsum() - 1
    # dt used for time accounting: within-segment only
    df.loc[df["dt"].isna() | (df["dt"] > GAP_S), "dt"] = np.nan
    df["stop"] = df["Robot_ProtectiveStop"]  # NaN kept as unknown
    df["grip"] = df["grip_lost"].astype(bool)
    return df


def rising_edges(flag: pd.Series, segment: pd.Series) -> np.ndarray:
    """Indices where flag goes 0->1 (or starts at 1 at a segment start). NaN counts as 0 for edges."""
    f = flag.fillna(0).astype(int).to_numpy()
    seg = segment.to_numpy()
    prev = np.r_[0, f[:-1]]
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    prev[new_seg] = 0
    return np.flatnonzero((f == 1) & (prev == 0))


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode of 'down' (protective stop OR grip loss), with start, end, duration and type."""
    down = ((df["stop"] == 1) | df["grip"]).astype(int)
    rows = []
    seg = df["segment"].to_numpy()
    d = down.to_numpy()
    t = df["t"].to_numpy()
    i, n = 0, len(df)
    while i < n:
        if d[i] == 1:
            j = i
            while j + 1 < n and d[j + 1] == 1 and seg[j + 1] == seg[i]:
                j += 1
            # duration: from first down sample to first sample after (or last sample if segment ends)
            end_idx = j + 1 if (j + 1 < n and seg[j + 1] == seg[i]) else j
            dur = (t[end_idx] - t[i]) / np.timedelta64(1, "s")
            if dur == 0:
                dur = float(np.nanmedian(df["dt"]))
            kind_stop = bool((df["stop"].iloc[i:j + 1] == 1).any())
            kind_grip = bool(df["grip"].iloc[i:j + 1].any())
            rows.append(dict(start=i, end=j, segment=int(seg[i]), cycle=int(df["cycle"].iloc[i]),
                             duration_s=float(dur), stop=kind_stop, grip=kind_grip))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def observed_seconds(df: pd.DataFrame) -> float:
    return float(df["dt"].sum())


def reliability(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    T = observed_seconds(df)
    down = float(ep["duration_s"].sum())
    up = T - down
    n = len(ep)
    mtbf = up / n
    mttr = down / n
    return dict(observed_s=T, down_s=down, up_s=up, n_episodes=n,
                n_stop_onsets=int(len(rising_edges(df["stop"], df["segment"]))),
                n_grip_onsets=int(len(rising_edges(df["grip"].astype(int), df["segment"]))),
                mtbf_s=mtbf, mttr_s=mttr, availability=mtbf / (mtbf + mttr))


# ---------- reliability statistics ----------
from scipy import stats as _st


def interarrivals(df: pd.DataFrame, onsets: np.ndarray) -> np.ndarray:
    """Onset-to-onset times (s) within a segment; pairs that span a recording gap are dropped."""
    t = df["t"].to_numpy()
    seg = df["segment"].to_numpy()
    o = np.sort(onsets)
    same = seg[o[1:]] == seg[o[:-1]]
    return ((t[o[1:]] - t[o[:-1]]) / np.timedelta64(1, "s"))[same].astype(float)


def weibull_fit(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """2-parameter Weibull MLE (location 0), bootstrap 95% CI on the shape, and a KS test
    against an exponential with the same mean (shape 1 = memoryless, random trigger;
    shape > 1 = wear-out). KS p-value is approximate because the mean is estimated."""
    k, _, lam = _st.weibull_min.fit(x, floc=0)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        xb = rng.choice(x, len(x), replace=True)
        ks.append(_st.weibull_min.fit(xb, floc=0)[0])
    ks_exp = _st.kstest(x, "expon", args=(0, x.mean()))
    return dict(n=int(len(x)), shape=float(k), scale_s=float(lam), shape_lo=float(np.percentile(ks, 2.5)),
                shape_hi=float(np.percentile(ks, 97.5)), mean_s=float(x.mean()), cv=float(x.std(ddof=1) / x.mean()),
                ks_exp_D=float(ks_exp.statistic), ks_exp_p=float(ks_exp.pvalue))


def mtbf_ci(up_s: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Time-terminated chi-square CI on MTBF: [2T/chi2(1-a/2, 2n+2), 2T/chi2(a/2, 2n)]."""
    return (2 * up_s / _st.chi2.ppf(1 - alpha / 2, 2 * n + 2), 2 * up_s / _st.chi2.ppf(alpha / 2, 2 * n))


def row_down_seconds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row: observed seconds (forward interval within segment) and down seconds."""
    fwd = df["t"].shift(-1).sub(df["t"]).dt.total_seconds().to_numpy()
    same = (df["segment"].shift(-1) == df["segment"]).to_numpy()
    fwd = np.where(same, fwd, np.nan)
    obs = np.nan_to_num(fwd)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    return obs, np.where(down, np.where(same, fwd, np.nanmedian(df["dt"])), 0.0)


def availability_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cycle-block bootstrap of A = 1 - D/T (cycles resampled with replacement)."""
    obs, dn = row_down_seconds(df)
    g = df["cycle"].to_numpy()
    cyc, inv = np.unique(g, return_inverse=True)
    T = np.bincount(inv, obs)
    D = np.bincount(inv, dn)
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n_boot):
        i = rng.integers(0, len(cyc), len(cyc))
        a.append(1 - D[i].sum() / T[i].sum())
    return dict(point=float(1 - D.sum() / T.sum()), lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))
