"""Joint load residuals and an early-warning classifier for stop / grip-loss onsets."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

HORIZON_S = 10  # predict an onset within the next 10 samples (~10 s)
WIN = 8         # rolling window (samples) for features


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for j in range(6):
        sp = df[f"Speed_J{j}"].interpolate(limit_direction="both")
        acc = np.zeros(len(df))
        for _, idx in df.groupby("segment").indices.items():
            s = sp.to_numpy()[idx]
            if len(s) >= 7:
                acc[idx] = savgol_filter(s, 7, 2, deriv=1, delta=1.0)
        df[f"Acc_J{j}"] = acc
    return df


def fit_load_models(df: pd.DataFrame, clean: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """current_j ~ b0 + b1*speed + b2*acc + b3*sign(speed) + b4*temp + b5..: other joints' speed (coupling).
    Fitted on clean rows only. Returns residual columns and R^2 per joint."""
    df = df.copy()
    stats = {}
    for j in range(6):
        X = np.column_stack([
            df[f"Speed_J{j}"], df[f"Acc_J{j}"], np.sign(df[f"Speed_J{j}"]), df[f"Temperature_J{j}"],
            *[df[f"Speed_J{k}"] for k in range(6) if k != j],
            *[df[f"Acc_J{k}"] for k in range(6) if k != j],
        ])
        y = df[f"Current_J{j}"].to_numpy()
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        m = ok & clean
        mdl = Ridge(alpha=1.0).fit(X[m], y[m])
        pred = np.full(len(df), np.nan)
        pred[ok] = mdl.predict(X[ok])
        res = y - pred
        sd = np.nanstd(res[m])
        df[f"Res_J{j}"] = res / sd
        r2 = 1 - np.nanvar(res[m]) / np.nanvar(y[m])
        stats[f"J{j}"] = dict(r2_clean=float(r2), resid_sd_A=float(sd))
    return df, stats


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    cols = []
    g = df.groupby("segment")
    for j in range(6):
        a = df[f"Res_J{j}"].abs()
        df[f"f_res{j}_mean"] = a.groupby(df["segment"]).transform(lambda s: s.rolling(WIN, 1).mean())
        df[f"f_res{j}_max"] = a.groupby(df["segment"]).transform(lambda s: s.rolling(WIN, 1).max())
        df[f"f_temp{j}_slope"] = df[f"Temperature_J{j}"].groupby(df["segment"]).diff(WIN)
        cols += [f"f_res{j}_mean", f"f_res{j}_max", f"f_temp{j}_slope"]
    df["f_tool"] = df["Tool_current"]
    df["f_tool_dmin"] = df["Tool_current"].groupby(df["segment"]).transform(lambda s: s.rolling(WIN, 1).min())
    df["f_tool_diff"] = df["Tool_current"].groupby(df["segment"]).diff()
    cols += ["f_tool", "f_tool_dmin", "f_tool_diff"]
    return df, cols


def labels(df: pd.DataFrame, onsets: np.ndarray, down: np.ndarray, horizon: int = HORIZON_S) -> tuple[np.ndarray, np.ndarray]:
    """y=1 if an onset occurs within the next `horizon` samples in the same segment.
    Rows that are already down, or whose stop label is unknown, are excluded (mask False)."""
    n = len(df)
    y = np.zeros(n, int)
    seg = df["segment"].to_numpy()
    for o in onsets:
        lo = max(0, o - horizon)
        for i in range(lo, o):
            if seg[i] == seg[o]:
                y[i] = 1
    mask = (~down) & df["stop"].notna().to_numpy()
    return y, mask


def episode_catch(score, thr, onsets, mask, seg, horizon=HORIZON_S):
    """Fraction of onsets with >=1 alert in the preceding horizon, alerts per observed hour on up rows,
    and lead time (s) of the earliest alert for each caught onset."""
    alert = (score >= thr) & mask
    caught, leads = 0, []
    for o in onsets:
        idx = [i for i in range(max(0, o - horizon), o) if seg[i] == seg[o] and alert[i]]
        if idx:
            caught += 1
            leads.append(o - idx[0])
    return caught / len(onsets), leads


def cross_validate(df, cols, y, mask, n_splits=5, seed=0):
    X = df[cols].to_numpy()
    groups = df["cycle"].to_numpy()
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        tr = tr[mask[tr]]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def pr_summary(y, score, mask):
    yy, ss = y[mask], score[mask]
    ok = ~np.isnan(ss)
    yy, ss = yy[ok], ss[ok]
    p, r, t = precision_recall_curve(yy, ss)
    return dict(ap=float(average_precision_score(yy, ss)), precision=p.tolist(), recall=r.tolist())
