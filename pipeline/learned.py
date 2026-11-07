"""Learned models under leakage-free validation.

1. Physics-informed expected current per joint (inverse-dynamics style): inertia, viscous + Coulomb
   friction, gravity from integrated joint angles, velocity-product coupling, temperature. Fitted per
   fold on clean training rows only.
2. A small 1D-CNN over the last L samples of 25 channels (6 joint currents, 6 physics residuals,
   6 speeds, 6 temperature changes, tool current), with cycle phase as an extra input.
3. Contiguous time-blocked folds of consecutive cycles, with a purge gap, so no onset, rolling window
   or label horizon is shared between train and test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

L = 12            # sequence window (samples, ~1 s each)
PHASE_CAP = 40


# ---------------------------------------------------------------- folds
def blocked_folds(df: pd.DataFrame, n_folds: int = 5, purge: int = 20) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous blocks of consecutive cycle runs, in time order. A test fold is one block; training rows
    within `purge` samples of the block (either side) are dropped, so label windows (10 s) and the
    12-sample input window never straddle train and test."""
    run = df["cycle_run"].to_numpy()
    runs = np.unique(run)
    edges = np.linspace(0, len(runs), n_folds + 1).round().astype(int)
    n = len(df)
    idx = np.arange(n)
    out = []
    for k in range(n_folds):
        te_runs = runs[edges[k]:edges[k + 1]]
        te = np.flatnonzero(np.isin(run, te_runs))
        lo, hi = te.min(), te.max()
        tr = idx[(idx < lo - purge) | (idx > hi + purge)]
        out.append((tr, te))
    return out


# ---------------------------------------------------------------- physics residual model
def integrated_angles(df: pd.DataFrame) -> np.ndarray:
    """Joint angle relative to the cycle start: q_j = sum(omega_j * dt) within each cycle run.
    The absolute offset is unknown; a*sin(q) + b*cos(q) absorbs any constant offset c, because
    A*sin(q + c) = A cos(c) sin(q) + A sin(c) cos(q)."""
    dt = df["dt"].fillna(1.0).clip(upper=2.0).to_numpy()
    q = np.zeros((len(df), 6))
    for j in range(6):
        w = df[f"Speed_J{j}"].interpolate(limit_direction="both").to_numpy()
        q[:, j] = pd.Series(w * dt).groupby(df["cycle_run"].to_numpy()).cumsum().to_numpy()
    return q


def physics_design(df: pd.DataFrame, j: int, q: np.ndarray) -> np.ndarray:
    """Regressors for expected current of joint j (all linear in the unknown dynamic parameters)."""
    w = np.column_stack([df[f"Speed_J{k}"].to_numpy() for k in range(6)])
    a = np.column_stack([df[f"Acc_J{k}"].to_numpy() for k in range(6)])
    T = df[f"Temperature_J{j}"].to_numpy()
    s1, s12, s123 = q[:, 1], q[:, 1] + q[:, 2], q[:, 1] + q[:, 2] + q[:, 3]
    cols = [
        a[:, j],                         # own inertia
        w[:, j], np.sign(w[:, j]),       # viscous, Coulomb friction
        w[:, j] * T, np.sign(w[:, j]) * T,  # friction drifting with temperature
        T,
        np.sin(s1), np.cos(s1), np.sin(s12), np.cos(s12), np.sin(s123), np.cos(s123),  # gravity (shoulder chain)
        np.sin(q[:, j]), np.cos(q[:, j]),
        *[a[:, k] for k in range(6) if k != j],       # inertial coupling
        *[w[:, k] for k in range(6) if k != j],       # velocity coupling / configuration proxy at 1 Hz
        w[:, 1] * w[:, 2], w[:, 1] ** 2, w[:, 2] ** 2, w[:, 0] ** 2,  # centrifugal / Coriolis products
        df["Tool_current"].to_numpy(),               # payload proxy (gripper load)
    ]
    return np.column_stack(cols)


def ridge_design(df: pd.DataFrame, j: int) -> np.ndarray:
    """The earlier linear load model (section 4): own speed, acc, sign, temperature, others' speed and acc."""
    return np.column_stack([df[f"Speed_J{j}"], df[f"Acc_J{j}"], np.sign(df[f"Speed_J{j}"]), df[f"Temperature_J{j}"],
                            *[df[f"Speed_J{k}"] for k in range(6) if k != j],
                            *[df[f"Acc_J{k}"] for k in range(6) if k != j]])


def fit_residuals(df, q, clean, train_rows, design="physics", alpha=1.0):
    """Fit expected current per joint on clean training rows; return standardised residuals for all rows
    and the held-out R^2 on clean non-training rows (if any)."""
    n = len(df)
    Z = np.full((n, 6), np.nan)
    r2 = []
    trm = np.zeros(n, bool)
    trm[train_rows] = True
    for j in range(6):
        X = physics_design(df, j, q) if design == "physics" else ridge_design(df, j)
        y = df[f"Current_J{j}"].to_numpy()
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        m = ok & clean & trm
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        mdl = Ridge(alpha=alpha).fit((X[m] - mu) / sd, y[m])
        pred = np.full(n, np.nan)
        pred[ok] = mdl.predict((X[ok] - mu) / sd)
        res = y - pred
        s = np.nanstd(res[m])
        Z[:, j] = res / s
        h = ok & clean & ~trm
        r2.append(float(1 - np.nanvar(res[h]) / np.nanvar(y[h])) if h.sum() > 10 else np.nan)
    return Z, r2


def residual_r2_blocked(df, q, clean, folds, design):
    """Held-out R^2 per joint, pooled over blocked folds (residuals predicted on the test block)."""
    n = len(df)
    res_all = np.full((n, 6), np.nan)
    for tr, te in folds:
        for j in range(6):
            X = physics_design(df, j, q) if design == "physics" else ridge_design(df, j)
            y = df[f"Current_J{j}"].to_numpy()
            ok = ~np.isnan(X).any(1) & ~np.isnan(y)
            m = np.zeros(n, bool); m[tr] = True; m &= ok & clean
            mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
            mdl = Ridge(alpha=1.0).fit((X[m] - mu) / sd, y[m])
            t = np.zeros(n, bool); t[te] = True; t &= ok
            res_all[t, j] = y[t] - mdl.predict((X[t] - mu) / sd)
    out = []
    for j in range(6):
        y = df[f"Current_J{j}"].to_numpy()
        h = clean & ~np.isnan(res_all[:, j])
        out.append(float(1 - np.var(res_all[h, j]) / np.var(y[h])))
    return out


# ---------------------------------------------------------------- windows
def channels(df: pd.DataFrame, Z: np.ndarray) -> np.ndarray:
    cur = np.column_stack([df[f"Current_J{j}"].to_numpy() for j in range(6)])
    spd = np.column_stack([df[f"Speed_J{j}"].to_numpy() for j in range(6)])
    tmp = np.column_stack([df[f"Temperature_J{j}"].to_numpy() for j in range(6)])
    tool = df["Tool_current"].to_numpy()[:, None]
    X = np.column_stack([cur, np.nan_to_num(Z), spd, tmp, tool])
    return pd.DataFrame(X).interpolate(limit_direction="both").to_numpy()


def windows(X: np.ndarray, seg: np.ndarray, length: int = L) -> np.ndarray:
    """(n, C, length) array: the last `length` samples up to and including row i, within the segment.
    Rows before the segment start are padded with the first sample of the segment. Temperatures
    (channels 18-23) are expressed relative to the window's last sample (change, not level)."""
    n, C = X.shape
    seg_start = np.zeros(n, int)
    for s in np.unique(seg):
        ix = np.flatnonzero(seg == s)
        seg_start[ix] = ix[0]
    lag = np.arange(length - 1, -1, -1)
    rows = np.maximum(np.arange(n)[:, None] - lag[None, :], seg_start[:, None])
    W = X[rows].transpose(0, 2, 1).copy()
    W[:, 18:24, :] -= W[:, 18:24, -1:]
    return W


def phase_inputs(phase: np.ndarray) -> np.ndarray:
    """Cycle phase as model input: one-hot of min(phase, 30) bucketed into 16 bins plus a ramp."""
    p = np.minimum(phase, 30)
    b = np.minimum(p // 2, 15)
    oh = np.eye(16)[b]
    return np.column_stack([oh, p / 30.0])


# ---------------------------------------------------------------- 1D-CNN
def train_cnn(Wtr, Ptr, ytr, Wte, Pte, epochs=15, seeds=(0, 1), lr=3e-3, wd=1e-3, use_signals=True, drop=()):
    import torch
    import torch.nn as nn
    torch.set_num_threads(3)
    C = Wtr.shape[1]
    mu = Wtr.mean((0, 2), keepdims=True)
    sd = Wtr.std((0, 2), keepdims=True) + 1e-6
    Wtr_n, Wte_n = (Wtr - mu) / sd, (Wte - mu) / sd
    if not use_signals:
        Wtr_n, Wte_n = np.zeros_like(Wtr_n), np.zeros_like(Wte_n)
    if len(drop):  # ablation: silence these channels (after normalisation, so they carry no information)
        Wtr_n, Wte_n = Wtr_n.copy(), Wte_n.copy()
        Wtr_n[:, list(drop)] = 0.0
        Wte_n[:, list(drop)] = 0.0

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(C, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(64 + Ptr.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, w, p):
            h = self.conv(w)
            h = torch.cat([h.mean(2), h.amax(2), p], 1)
            return self.head(h).squeeze(1)

    preds = []
    for s in seeds:
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        pos = max(ytr.mean(), 1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
        wt, pt, yt = (torch.tensor(Wtr_n, dtype=torch.float32), torch.tensor(Ptr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.float32))
        for _ in range(epochs):
            net.train()
            perm = rng.permutation(len(yt))
            for b in range(0, len(perm), 256):
                ix = perm[b:b + 256]
                opt.zero_grad()
                loss = lossf(net(wt[ix], pt[ix]), yt[ix])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(net(torch.tensor(Wte_n, dtype=torch.float32),
                                           torch.tensor(Pte, dtype=torch.float32))).numpy())
    return np.mean(preds, 0)


# ---------------------------------------------------------------- lead-time labels
def lead_labels(df: pd.DataFrame, onsets: np.ndarray, down: np.ndarray, horizon: int = 10,
                min_lead: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """y=1 if an onset falls min_lead..horizon samples ahead (same segment). Rows whose nearest onset is
    only 1..min_lead-1 samples ahead are dropped from the mask (neither positive nor negative), so a model
    gets no credit for spotting an event that is already starting. Down rows are dropped as usual."""
    n = len(df)
    seg = df["segment"].to_numpy()
    y = np.zeros(n, int)
    near = np.zeros(n, bool)
    for o in onsets:
        for i in range(max(0, o - horizon), o):
            if seg[i] != seg[o]:
                continue
            if o - i >= min_lead:
                y[i] = 1
            else:
                near[i] = True
    mask = (~down) & df["stop"].notna().to_numpy() & ~(near & (y == 0))
    return y, mask


# ---------------------------------------------------------------- baselines
def phase_rate(phase_tr, y_tr, phase_te, prior_n=5.0):
    p_tr, p_te = np.minimum(phase_tr, PHASE_CAP), np.minimum(phase_te, PHASE_CAP)
    base = y_tr.mean()
    num = np.bincount(p_tr, y_tr, minlength=PHASE_CAP + 1)
    den = np.bincount(p_tr, minlength=PHASE_CAP + 1)
    return ((num + prior_n * base) / (den + prior_n))[p_te]


def since_last(df: pd.DataFrame, down: np.ndarray) -> np.ndarray:
    """Samples since the last down row in the same segment (capped at 300). A non-sensor recency baseline."""
    seg = df["segment"].to_numpy()
    out = np.full(len(df), 300.0)
    last = -10_000
    for i in range(len(df)):
        if i > 0 and seg[i] != seg[i - 1]:
            last = -10_000
        if down[i]:
            last = i
        out[i] = min(i - last, 300)
    return out


def gbm(seed=0):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


# ---------------------------------------------------------------- metrics
def ap(y, s, m):
    ok = m & ~np.isnan(s)
    return float(average_precision_score(y[ok], s[ok]))


def auc(y, s, m):
    ok = m & ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok]))


def block_bootstrap(y, scores: dict, mask, groups, pairs, metric="ap", n_boot=1000, seed=0):
    """Resample blocks (all rows of a block together) with replacement; CI of each score's metric and of
    paired differences a - b. p_le_0 = share of resamples where a - b <= 0."""
    f = average_precision_score if metric == "ap" else roc_auc_score
    idx = np.flatnonzero(mask)
    g = groups[idx]
    ug, inv = np.unique(g, return_inverse=True)
    rows_of = [idx[inv == i] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in scores}
    dv = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(n_boot):
        r = np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))])
        yy = y[r]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        v = {k: float(f(yy, s[r])) for k, s in scores.items()}
        for k in v:
            vals[k].append(v[k])
        for a, b in pairs:
            dv[f"{a}-{b}"].append(v[a] - v[b])
    ci = lambda x: dict(lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)))
    out = {k: ci(v) for k, v in vals.items()}
    for k, v in dv.items():
        out[k] = ci(v) | dict(p_le_0=float(np.mean(np.array(v) <= 0)))
    return out


# ---------------------------------------------------------------- segment fix arithmetic
def segment_fix_gain(n_in: int, e_in: float, n_out: int, e_out: float, mttr_s: float, observed_s: float,
                     n_other_episodes: int, other_down_s: float) -> dict:
    """If the window's stop rate fell to the cycle-average rate (all stops / all up-time), how many stops go,
    and what availability results? Up-time held fixed (stops are short), downtime per stop = mttr_s."""
    avg = (n_in + n_out) / (e_in + e_out)
    n_in_new = avg * e_in
    removed = n_in - n_in_new
    stops_before = n_in + n_out
    stops_after = stops_before - removed
    up = e_in + e_out
    down_before = stops_before * mttr_s + other_down_s
    down_after = stops_after * mttr_s + other_down_s
    return dict(avg_rate_per_100s=100 * avg, stops_in_window=n_in, stops_in_window_after=n_in_new,
                stops_removed=removed, share_removed=removed / stops_before,
                availability_before=up / (up + down_before), availability_after=up / (up + down_after),
                mtbf_before_s=up / (stops_before + n_other_episodes), mtbf_after_s=up / (stops_after + n_other_episodes))
