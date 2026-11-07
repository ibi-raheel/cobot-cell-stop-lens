"""Adversarial checks on the learned-model results -> public/data/leakage_checks.json.

1. Causal features. warning.add_kinematics uses a centred Savitzky-Golay derivative (window 7), so the
   acceleration at row i uses speeds up to i+3, and NaN gaps are filled by two-sided interpolation. Here
   acceleration comes from a one-sided 7-point quadratic fit (past samples only) and gaps are forward-filled;
   the grip-loss and stop targets are re-scored.
2. Grip-aware recency baseline. The phase + recency baseline counts samples since the last down row of any
   kind, which protective stops dominate. Grip losses cluster (33 of 77 onsets follow another grip onset
   within 25 samples), so a fair baseline also knows the time since the last stop and since the last grip loss.
   The CNN is also scored only on rows with no grip loss in the recent past.

Usage: OMP_NUM_THREADS=3 python -I pipeline/check_leakage.py downloads/dataset_02052023.xlsx public/data
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
import reliability as R
import warning as W
import phase as P
import learned as LM


def r(x, k=3):
    return round(float(x), k)


def causal_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Acceleration at row i from a quadratic fit to speeds i-6..i only (no look-ahead)."""
    c = savgol_coeffs(7, 2, deriv=1, pos=6, use="dot")
    df = df.copy()
    for j in range(6):
        sp = df[f"Speed_J{j}"].groupby(df["segment"]).ffill().fillna(0.0)
        acc = np.zeros(len(df))
        for _, idx in df.groupby("segment").indices.items():
            s = sp.to_numpy()[idx]
            pad = np.r_[np.full(6, s[0]), s]
            acc[idx] = np.array([pad[i:i + 7] @ c for i in range(len(s))])
        df[f"Acc_J{j}"] = acc
    return df


def causal_channels(df: pd.DataFrame, Z: np.ndarray) -> np.ndarray:
    cur = np.column_stack([df[f"Current_J{j}"].to_numpy() for j in range(6)])
    spd = np.column_stack([df[f"Speed_J{j}"].to_numpy() for j in range(6)])
    tmp = np.column_stack([df[f"Temperature_J{j}"].to_numpy() for j in range(6)])
    tool = df["Tool_current"].to_numpy()[:, None]
    X = np.column_stack([cur, np.nan_to_num(Z), spd, tmp, tool])
    return pd.DataFrame(X).ffill().bfill().to_numpy()


def prep(src, causal):
    df = R.load(src)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    df = causal_acc(df) if causal else W.add_kinematics(df)
    df = P.add_phase(df)
    d2 = df.copy()
    if causal:
        for j in range(6):
            d2[f"Speed_J{j}"] = d2[f"Speed_J{j}"].ffill().fillna(0.0)
    q = LM.integrated_angles(d2)
    clean = ~down & df["stop"].notna().to_numpy()
    folds = LM.blocked_folds(df, 5, 20)
    seg = df["segment"].to_numpy()
    ch = causal_channels if causal else LM.channels
    W_by_fold = []
    for tr, te in folds:
        Z, _ = LM.fit_residuals(df, q, clean, tr, "physics")
        W_by_fold.append(LM.windows(ch(df, Z), seg))
    return df, down, folds, W_by_fold


def main(src, out):
    res = {}
    blocks = None
    for causal in (True, False):
        df, down, folds, Wf = prep(src, causal)
        blocks = df["cycle_run"].to_numpy() // 5
        phase = df["phase"].to_numpy()
        Pin = LM.phase_inputs(phase)
        rec = LM.since_last(df, down)
        rec_stop = LM.since_last(df, (df["stop"] == 1).to_numpy())
        rec_grip = LM.since_last(df, df["grip"].to_numpy())
        onset = dict(stop=R.rising_edges(df["stop"], df["segment"]),
                     grip=R.rising_edges(df["grip"].astype(int), df["segment"]))
        X1 = np.column_stack([np.minimum(phase, 40), rec])
        X2 = np.column_stack([np.minimum(phase, 40), rec, rec_stop, rec_grip])
        tasks = {"grip_lead3": ("grip", 3), "stop_10": ("stop", 1)} if causal else {"grip_lead3": ("grip", 3)}
        for t, (kind, lead) in tasks.items():
            y, mask = (W.labels(df, onset[kind], down, 10) if lead == 1
                       else LM.lead_labels(df, onset[kind], down, 10, lead))
            sc = {k: np.full(len(df), np.nan) for k in ["phase_recency", "phase_recency_by_kind", "cnn"]}
            for (tr, te), Wn in zip(folds, Wf):
                trm = tr[mask[tr]]
                sc["phase_recency"][te] = LM.gbm().fit(X1[trm], y[trm]).predict_proba(X1[te])[:, 1]
                sc["phase_recency_by_kind"][te] = LM.gbm().fit(X2[trm], y[trm]).predict_proba(X2[te])[:, 1]
                sc["cnn"][te] = LM.train_cnn(Wn[trm], Pin[trm], y[trm].astype(float), Wn[te], Pin[te])
            pairs = [("cnn", "phase_recency"), ("cnn", "phase_recency_by_kind")]
            b = LM.block_bootstrap(y, sc, mask, blocks, pairs, "ap")
            entry = dict(base_rate=r(y[mask].mean()), positives=int(y[mask].sum()),
                         ap={k: dict(v=r(LM.ap(y, s, mask)), lo=r(b[k]["lo"]), hi=r(b[k]["hi"])) for k, s in sc.items()},
                         diff_ap={f"{a}-{c}": dict(v=r(LM.ap(y, sc[a], mask) - LM.ap(y, sc[c], mask)),
                                                   lo=r(b[f"{a}-{c}"]["lo"]), hi=r(b[f"{a}-{c}"]["hi"])) for a, c in pairs})
            if kind == "grip":
                entry["no_recent_grip_loss"] = {}
                for cut in (12, 25):
                    m = mask & (rec_grip > cut)
                    bb = LM.block_bootstrap(y, sc, m, blocks, [("cnn", "phase_recency_by_kind")], "ap")
                    entry["no_recent_grip_loss"][str(cut)] = dict(
                        rows=int(m.sum()), positives=int(y[m].sum()), base_rate=r(y[m].mean()),
                        ap={k: r(LM.ap(y, s, m)) for k, s in sc.items()},
                        cnn_ci=[r(bb["cnn"]["lo"]), r(bb["cnn"]["hi"])],
                        diff_ci=[r(bb["cnn-phase_recency_by_kind"]["lo"]), r(bb["cnn-phase_recency_by_kind"]["hi"])])
                g = onset["grip"]
                seg = df["segment"].to_numpy()
                gap = np.r_[10 ** 6, np.diff(g)]
                same = np.r_[False, seg[g[1:]] == seg[g[:-1]]]
                entry["grip_onsets"] = int(len(g))
                entry["grip_onsets_within_25_of_previous"] = int(((gap <= 25) & same).sum())
            res[("causal_" if causal else "shipped_") + t] = entry
            print(("causal " if causal else "shipped ") + t, json.dumps(entry["ap"]), json.dumps(entry["diff_ap"]),
                  json.dumps(entry.get("no_recent_grip_loss", {})), flush=True)
    with open(os.path.join(out, "leakage_checks.json"), "w") as f:
        json.dump(res, f, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
