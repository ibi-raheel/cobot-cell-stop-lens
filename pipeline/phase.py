"""Cycle-phase hazard: where in the robot cycle do stops and grip losses start?

Phase = sample index within a (segment, cycle) run. For each phase bin the hazard is
onsets / up-seconds at risk, with an exact Poisson CI. Also: per-joint kinematics per bin,
a downtime Pareto by cycle, and a Poisson rate-ratio power calculation for a fix trial.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, binomtest

BINS = [(0, 1), (2, 5), (6, 10), (11, 14), (15, 30), (31, 10_000)]
HOT = (2, 10)


def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    run = ((df["cycle"] != df["cycle"].shift()) | (df["segment"] != df["segment"].shift())).cumsum()
    df["cycle_run"] = run
    df["phase"] = df.groupby(run).cumcount()
    return df


def poisson_ci(k: int, exposure: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) CI for a Poisson rate k / exposure."""
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def rate_ratio_test(k1: int, e1: float, k0: int, e0: float, alpha: float = 0.05) -> dict:
    """Rate ratio (k1/e1)/(k0/e0) with exact conditional (binomial) CI and p-value."""
    n = k1 + k0
    p0 = e1 / (e1 + e0)
    bt = binomtest(k1, n, p0)
    ci = bt.proportion_ci(1 - alpha, method="exact")
    to_rr = lambda p: (p / (1 - p)) * (e0 / e1) if p < 1 else np.inf
    return dict(rr=(k1 / e1) / (k0 / e0), lo=to_rr(ci.low), hi=to_rr(ci.high), p=bt.pvalue)


def at_risk(df: pd.DataFrame, down: np.ndarray) -> np.ndarray:
    """Rows where the cell is up and the stop label is known (exposure, ~1 s each)."""
    return (~down) & df["stop"].notna().to_numpy()


def hazard_table(df: pd.DataFrame, onset_idx: dict[str, np.ndarray], down: np.ndarray, bins=BINS) -> list[dict]:
    """Per phase bin: up-seconds, onsets by kind, rate per 100 up-s with exact CI, mean |speed| and |acc| per joint."""
    ph = df["phase"].to_numpy()
    risk = at_risk(df, down)
    dt = df["dt"].fillna(df["dt"].median()).to_numpy()
    rows = []
    for lo, hi in bins:
        inb = (ph >= lo) & (ph <= hi)
        expo = float(dt[inb & risk].sum())
        row = dict(bin=f"{lo}-{hi}" if hi < 10_000 else f"{lo}+", lo=lo, hi=hi, up_s=expo,
                   share_up=expo / float(dt[risk].sum()))
        for kind, idx in onset_idx.items():
            k = int(inb[idx].sum())
            rlo, rhi = poisson_ci(k, expo)
            row[kind] = dict(n=k, rate100=100 * k / expo if expo else np.nan, lo100=100 * rlo, hi100=100 * rhi)
        row["speed"] = [float(df.loc[inb, f"Speed_J{j}"].abs().mean()) for j in range(6)]
        if "Acc_J0" in df:
            row["acc"] = [float(df.loc[inb, f"Acc_J{j}"].abs().mean()) for j in range(6)]
        rows.append(row)
    return rows


def hot_window(df: pd.DataFrame, idx: np.ndarray, down: np.ndarray, win=HOT) -> dict:
    """Share of onsets in the window, its rate vs the rest of the cycle, and the rate ratio CI."""
    ph = df["phase"].to_numpy()
    risk = at_risk(df, down)
    dt = df["dt"].fillna(df["dt"].median()).to_numpy()
    inw = (ph >= win[0]) & (ph <= win[1])
    k1, k0 = int(inw[idx].sum()), int((~inw[idx]).sum())
    e1, e0 = float(dt[inw & risk].sum()), float(dt[~inw & risk].sum())
    rr = rate_ratio_test(k1, e1, k0, e0)
    return dict(window=list(win), n_in=k1, n_out=k0, share=k1 / max(k1 + k0, 1), up_share=e1 / (e1 + e0),
                rate100_in=100 * k1 / e1, rate100_out=100 * k0 / e0, rr=rr["rr"], rr_lo=rr["lo"],
                rr_hi=rr["hi"], p=rr["p"])


def pareto(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """Downtime by cycle id: what share of downtime the worst 10% of cycles hold."""
    n_cyc = int(df["cycle"].nunique())
    by = ep.groupby("cycle")["duration_s"].sum().sort_values(ascending=False)
    top = int(round(0.10 * n_cyc))
    cum = by.cumsum() / by.sum()
    n50 = int((cum < 0.5).sum() + 1)
    return dict(n_cycles=n_cyc, cycles_with_episode=int(len(by)), top10pct_cycles=top,
                top10pct_share=float(by.iloc[:top].sum() / by.sum()), cycles_for_half=n50,
                curve=[round(float(c), 4) for c in cum.to_numpy()])


def events_needed(rr: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    """Baseline-arm events needed to detect rate ratio `rr` with equal exposure per arm,
    normal approximation on log(rr): var = 1/k0 + 1/k1, k1 = rr*k0."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z ** 2 * (1 + 1 / rr) / np.log(rr) ** 2


def run_hours_needed(rate_per_h: float, rr: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    """Run-hours per arm (before and after the change) to detect `rr` at the given baseline stop rate."""
    return events_needed(rr, alpha, power) / rate_per_h


def phase_profile(df: pd.DataFrame, down: np.ndarray, onset_idx: dict[str, np.ndarray], max_phase: int = 30) -> list[dict]:
    """Per sample index (0..max_phase): mean signed speed per joint and tool current on up rows,
    plus onset counts by kind. Shows which joint is moving just before the hot samples."""
    ph = df["phase"].to_numpy()
    up = ~down
    out = []
    for p in range(max_phase + 1):
        m = (ph == p) & up
        row = dict(phase=p, n_up=int(m.sum()),
                   speed=[round(float(df.loc[m, f"Speed_J{j}"].mean()), 3) for j in range(6)],
                   tool=round(float(df.loc[m, "Tool_current"].mean()), 3))
        for kind, idx in onset_idx.items():
            row[kind] = int((ph[idx] == p).sum())
        out.append(row)
    return out
