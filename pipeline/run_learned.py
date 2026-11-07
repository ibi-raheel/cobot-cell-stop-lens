"""Learned-model study under time-blocked folds -> public/data/learned.json.
Usage: OMP_NUM_THREADS=3 python -I pipeline/run_learned.py downloads/dataset_02052023.xlsx public/data
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
T0 = time.time()
import reliability as R
import warning as W
import phase as P
import learned as LM


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def rolling_feats(Z, seg, win=8):
    import pandas as pd
    a = pd.DataFrame(np.abs(Z))
    g = a.groupby(seg)
    mean = g.transform(lambda s: s.rolling(win, 1).mean()).to_numpy()
    mx = g.transform(lambda s: s.rolling(win, 1).max()).to_numpy()
    return np.column_stack([mean, mx, Z])


def main(src, out):
    df = R.load(src)
    ep = R.episodes(df)
    rel = R.reliability(df, ep)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    df = W.add_kinematics(df)
    df = P.add_phase(df)
    clean = ~down & df["stop"].notna().to_numpy()
    seg = df["segment"].to_numpy()
    phase = df["phase"].to_numpy()
    q = LM.integrated_angles(df)
    folds = LM.blocked_folds(df, n_folds=5, purge=20)
    blocks = df["cycle_run"].to_numpy() // 5  # bootstrap unit: 5 consecutive cycle runs

    # ---- physics vs earlier linear load model: held-out R^2 on blocked folds ----
    r2_phys = LM.residual_r2_blocked(df, q, clean, folds, "physics")
    r2_lin = LM.residual_r2_blocked(df, q, clean, folds, "ridge")

    onset = dict(any=ep["start"].to_numpy(),
                 stop=R.rising_edges(df["stop"], df["segment"]),
                 grip=R.rising_edges(df["grip"].astype(int), df["segment"]))
    rec = LM.since_last(df, down)
    Pin = LM.phase_inputs(phase)

    # per-fold residuals and windows (residual model refitted on each fold's clean training rows)
    fold_data = []
    for tr, te in folds:
        Z, _ = LM.fit_residuals(df, q, clean, tr, "physics")
        Xc = LM.channels(df, Z)
        Wn = LM.windows(Xc, seg)
        F = np.column_stack([rolling_feats(Z, seg), np.minimum(phase, 40), rec])
        fold_data.append((tr, te, Wn, F))
        print('fold prep', round(time.time() - T0), flush=True)

    # (onset kind, horizon, min lead). min lead 3 = the onset must be 3-10 samples ahead; rows 1-2 samples
    # before an onset are left out, so no credit is given for spotting an event that is already starting.
    tasks = {"any_10": ("any", 10, 1), "stop_10": ("stop", 10, 1), "grip_10": ("grip", 10, 1), "any_30": ("any", 30, 1),
             "stop_lead3": ("stop", 10, 3), "grip_lead3": ("grip", 10, 3)}
    extra = {"any_10": ["cnn_phase_only"], "grip_10": ["cnn_no_tool"], "grip_lead3": ["cnn_no_tool"]}
    results, oof_store = {}, {}
    for tname, (kind, H, lead) in tasks.items():
        if lead == 1:
            y, mask = W.labels(df, onset[kind], down, horizon=H)
        else:
            y, mask = LM.lead_labels(df, onset[kind], down, horizon=H, min_lead=lead)
        sc = {k: np.full(len(df), np.nan) for k in ["phase_baseline", "phase_recency", "residual_gbm", "cnn"] + extra.get(tname, [])}
        for tr, te, Wn, F in fold_data:
            trm = tr[mask[tr]]
            sc["phase_baseline"][te] = LM.phase_rate(phase[trm], y[trm], phase[te])
            PR = np.column_stack([np.minimum(phase, 40), rec])
            sc["phase_recency"][te] = LM.gbm().fit(PR[trm], y[trm]).predict_proba(PR[te])[:, 1]
            sc["residual_gbm"][te] = LM.gbm().fit(F[trm], y[trm]).predict_proba(F[te])[:, 1]
            sc["cnn"][te] = LM.train_cnn(Wn[trm], Pin[trm], y[trm].astype(float), Wn[te], Pin[te])
            print(tname, 'fold', round(time.time() - T0), flush=True)
            if "cnn_phase_only" in sc:
                sc["cnn_phase_only"][te] = LM.train_cnn(Wn[trm], Pin[trm], y[trm].astype(float), Wn[te], Pin[te],
                                                        use_signals=False)
            if "cnn_no_tool" in sc:  # ablation: tool (gripper) current channel silenced
                sc["cnn_no_tool"][te] = LM.train_cnn(Wn[trm], Pin[trm], y[trm].astype(float), Wn[te], Pin[te],
                                                     drop=(24,))
        rng = np.random.default_rng(0)
        sc["random"] = rng.random(len(df))
        pairs = [("cnn", "phase_baseline"), ("cnn", "phase_recency"), ("residual_gbm", "phase_baseline"),
                 ("phase_baseline", "random")]
        boot_ap = LM.block_bootstrap(y, sc, mask, blocks, pairs, "ap")
        boot_auc = LM.block_bootstrap(y, sc, mask, blocks, pairs, "auc")
        results[tname] = dict(
            base_rate=r(y[mask].mean()), rows=int(mask.sum()), positives=int(y[mask].sum()), onsets=int(len(onset[kind])),
            ap={k: dict(v=r(LM.ap(y, s, mask)), lo=r(boot_ap[k]["lo"]), hi=r(boot_ap[k]["hi"])) for k, s in sc.items()},
            auc={k: dict(v=r(LM.auc(y, s, mask)), lo=r(boot_auc[k]["lo"]), hi=r(boot_auc[k]["hi"])) for k, s in sc.items()},
            diff_ap={f"{a}-{b}": {kk: r(vv) for kk, vv in boot_ap[f"{a}-{b}"].items()} | dict(v=r(LM.ap(y, sc[a], mask) - LM.ap(y, sc[b], mask))) for a, b in pairs},
            ap_by_session={str(g): {k: r(LM.ap(y, s, mask & (seg == g))) for k, s in sc.items() if k != "random"}
                           | dict(positives=int(y[mask & (seg == g)].sum()))
                           for g in np.unique(seg) if y[mask & (seg == g)].sum() >= 5},
            diff_auc={f"{a}-{b}": {kk: r(vv) for kk, vv in boot_auc[f"{a}-{b}"].items()} | dict(v=r(LM.auc(y, sc[a], mask) - LM.auc(y, sc[b], mask))) for a, b in pairs},
        )
        oof_store[tname] = (y, mask, sc)
        print(tname, json.dumps({k: v["v"] for k, v in results[tname]["ap"].items()}),
              json.dumps({k: v["v"] for k, v in results[tname]["auc"].items()}), flush=True)

    # ---- regime: same cycle position, different cycles. Rows in phase 0-1 (just before the J2 swing):
    # which cycles will throw a protective stop in samples 2-5? Phase cannot separate rows at the same phase.
    y, mask, sc = oof_store["stop_10"]
    stop_idx = onset["stop"]
    run = df["cycle_run"].to_numpy()
    hot_runs = set(run[stop_idx[(phase[stop_idx] >= 2) & (phase[stop_idx] <= 5)]])
    first = (phase == 1) & clean
    yc = np.array([run[i] in hot_runs for i in range(len(df))]).astype(int)
    cyc_scores = {k: s for k, s in sc.items() if k in ("cnn", "residual_gbm", "phase_recency", "random")}
    cyc_scores["recency_only"] = -rec.astype(float)
    pairs_c = [("cnn", "phase_recency"), ("cnn", "recency_only"), ("residual_gbm", "recency_only")]
    bc = LM.block_bootstrap(yc, cyc_scores, first, blocks, pairs_c, "auc")
    bca = LM.block_bootstrap(yc, cyc_scores, first, blocks, pairs_c, "ap")
    cycle_rank = dict(
        definition="one row per cycle at sample 1 (cell up); target = that cycle has a protective-stop onset in samples 2-5",
        cycles=int(first.sum()), positives=int(yc[first].sum()),
        auc={k: dict(v=r(LM.auc(yc, s, first)), lo=r(bc[k]["lo"]), hi=r(bc[k]["hi"])) for k, s in cyc_scores.items()},
        ap={k: dict(v=r(LM.ap(yc, s, first)), lo=r(bca[k]["lo"]), hi=r(bca[k]["hi"])) for k, s in cyc_scores.items()},
        diff_auc={k: {kk: r(vv) for kk, vv in bc[k].items()} for k in [f"{a}-{b}" for a, b in pairs_c]},
        diff_ap={k: {kk: r(vv) for kk, vv in bca[k].items()} for k in [f"{a}-{b}" for a, b in pairs_c]},
    )
    print("cycle_rank", json.dumps(cycle_rank["auc"]), json.dumps(cycle_rank["diff_auc"]), flush=True)

    # ---- the J2-swing fix: stops removed if samples 2-5 ran at the cycle-average stop rate ----
    hot = P.hot_window(df, stop_idx, down, (2, 5))
    dtv = df["dt"].fillna(df["dt"].median()).to_numpy()
    risk = P.at_risk(df, down)
    inw = (phase >= 2) & (phase <= 5)
    e_in, e_out = float(dtv[inw & risk].sum()), float(dtv[~inw & risk].sum())
    stop_ep = ep[ep["stop"]]
    mttr_stop = float(stop_ep["duration_s"].mean())
    other = ep[~ep["stop"]]
    fix = LM.segment_fix_gain(hot["n_in"], e_in, hot["n_out"], e_out, mttr_stop, rel["observed_s"],
                              int(len(other)), float(other["duration_s"].sum()))
    # bootstrap the fix over cycle blocks
    rng = np.random.default_rng(0)
    ub = np.unique(blocks)
    in_k = np.bincount(np.searchsorted(ub, blocks[stop_idx[inw[stop_idx]]]), minlength=len(ub))
    out_k = np.bincount(np.searchsorted(ub, blocks[stop_idx[~inw[stop_idx]]]), minlength=len(ub))
    bi = np.searchsorted(ub, blocks)
    ein_b = np.bincount(bi, dtv * (inw & risk), minlength=len(ub))
    eout_b = np.bincount(bi, dtv * (~inw & risk), minlength=len(ub))
    sh = []
    for _ in range(1000):
        p = rng.integers(0, len(ub), len(ub))
        a, b, c, d = in_k[p].sum(), ein_b[p].sum(), out_k[p].sum(), eout_b[p].sum()
        avg = (a + c) / (b + d)
        sh.append((a - avg * b) / (a + c))
    fix = {k: r(v, 4) for k, v in fix.items()}
    fix.update(share_removed_lo=r(np.percentile(sh, 2.5), 4), share_removed_hi=r(np.percentile(sh, 97.5), 4),
               mttr_stop_s=r(mttr_stop, 3), stops_per_h_before=r(3600 * (hot["n_in"] + hot["n_out"]) / (e_in + e_out), 2),
               stops_per_h_after=r(3600 * (hot["n_in"] + hot["n_out"] - fix["stops_removed"]) / (e_in + e_out), 2))
    print("fix", json.dumps(fix), flush=True)

    out_d = dict(
        validation="5 contiguous time blocks of consecutive cycle runs, 20-sample purge either side of each test block; "
                   "bootstrap = 1000 resamples of 5-cycle blocks",
        window_samples=LM.L, channels=25,
        load_model_r2_heldout=dict(physics=[r(v) for v in r2_phys], linear=[r(v) for v in r2_lin]),
        tasks=results, cycle_rank=cycle_rank, j2_fix=fix,
    )
    with open(os.path.join(out, "learned.json"), "w") as f:
        json.dump(out_d, f, separators=(",", ":"))
    print("r2", r2_phys, r2_lin)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
