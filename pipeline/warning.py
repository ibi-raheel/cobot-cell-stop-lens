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


# ---------- stronger baselines, ablation, time-ordered holdout, bootstrap ----------
PHASE_CAP = 40


def make_gbm(seed=0):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def phase_rate_fit(phase, y, prior_n: float = 5.0):
    """P(onset within horizon | cycle phase), smoothed toward the overall rate with `prior_n` pseudo-rows."""
    p = np.minimum(phase, PHASE_CAP)
    base = y.mean()
    num = np.bincount(p, y, minlength=PHASE_CAP + 1)
    den = np.bincount(p, minlength=PHASE_CAP + 1)
    return (num + prior_n * base) / (den + prior_n)


def phase_rate_score(table, phase):
    return table[np.minimum(phase, PHASE_CAP)]


def splits_groupkfold(df, n_splits=5):
    g = df["cycle"].to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(g, g, g))


def oof_scores(X, y, mask, splits, phase=None, seed=0):
    """Out-of-fold scores. X=None means the phase-rate baseline (needs `phase`)."""
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        tr = tr[mask[tr]]
        if X is None:
            oof[te] = phase_rate_score(phase_rate_fit(phase[tr], y[tr]), phase[te])
        else:
            oof[te] = make_gbm(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def ap(y, s, mask):
    ok = mask & ~np.isnan(s)
    return float(average_precision_score(y[ok], s[ok]))


def bootstrap_ap(y, scores: dict, mask, groups, n_boot=1000, seed=0, diffs=()):
    """Cycle-block bootstrap of PR-AUC for each scorer, and of differences (a - b) for pairs in `diffs`.
    Rows of a resampled cycle come in together, so within-cycle correlation is respected."""
    idx = np.flatnonzero(mask)
    g = groups[idx]
    cyc, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rows_of = [idx[order[starts[i]:starts[i + 1]]] for i in range(len(cyc))]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}
    dout = {f"{a}-{b}": [] for a, b in diffs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(cyc), len(cyc))
        r = np.concatenate([rows_of[i] for i in pick])
        yy = y[r]
        if yy.sum() == 0:
            continue
        vals = {k: float(average_precision_score(yy, s[r])) for k, s in scores.items()}
        for k, v in vals.items():
            out[k].append(v)
        for a, b in diffs:
            dout[f"{a}-{b}"].append(vals[a] - vals[b])
    ci = lambda v: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    res = {k: ci(v) for k, v in out.items()}
    res.update({k: ci(v) | dict(p_le_0=float(np.mean(np.array(v) <= 0))) for k, v in dout.items()})
    return res


def time_ordered(df, X_by_name: dict, y, mask, train_segs=(0, 1), test_seg=2, seed=0):
    """Train on earlier recording segments, test on the later one. Returns test-segment PR-AUC per scorer."""
    seg = df["segment"].to_numpy()
    phase = df["phase"].to_numpy()
    tr = np.isin(seg, train_segs) & mask
    te = (seg == test_seg) & mask
    out = {}
    for name, X in X_by_name.items():
        if X is None:
            s = phase_rate_score(phase_rate_fit(phase[tr], y[tr]), phase[te])
        else:
            s = make_gbm(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out[name] = float(average_precision_score(y[te], s))
    out["_test_base_rate"] = float(y[te].mean())
    out["_train_rows"], out["_test_rows"] = int(tr.sum()), int(te.sum())
    out["_train_positives"], out["_test_positives"] = int(y[tr].sum()), int(y[te].sum())
    return out
