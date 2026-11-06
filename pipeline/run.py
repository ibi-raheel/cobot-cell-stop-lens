"""End-to-end pipeline: raw UR3 CobotOps xlsx -> public/data/*.json.
Usage: python -I pipeline/run.py downloads/dataset_02052023.xlsx public/data
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import reliability as R
import warning as W
import throughput as T
import phase as P


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def main(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    df = R.load(src)
    ep = R.episodes(df)
    rel = R.reliability(df, ep)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()

    df = W.add_kinematics(df)
    clean = ~down & df["stop"].notna().to_numpy()
    df, load_stats = W.fit_load_models(df, clean)
    df, cols = W.features(df)
    onsets = ep["start"].to_numpy()
    y, mask = W.labels(df, onsets, down)
    seg = df["segment"].to_numpy()

    df = P.add_phase(df)
    phase = df["phase"].to_numpy()
    groups = df["cycle"].to_numpy()

    # ---- feature groups and scorers (all out-of-fold, 5-fold GroupKFold by cycle) ----
    splits = W.splits_groupkfold(df)
    df["f_phase"] = np.minimum(phase, W.PHASE_CAP)
    res_cols = [c for c in cols if c.startswith("f_res")]
    groups_X = {
        "model": df[cols].to_numpy(),
        "residuals_only": df[res_cols].to_numpy(),
        "tool_only": df[[c for c in cols if c.startswith("f_tool")]].to_numpy(),
        "temperature_only": df[[c for c in cols if c.startswith("f_temp")]].to_numpy(),
        "residuals_plus_phase": df[res_cols + ["f_phase"]].to_numpy(),
        "model_plus_phase": df[cols + ["f_phase"]].to_numpy(),
    }
    scores = {k: W.oof_scores(X, y, mask, splits) for k, X in groups_X.items()}
    scores["phase_baseline"] = W.oof_scores(None, y, mask, splits, phase=phase)
    oof = scores["model"]
    tool = -df["f_tool_dmin"].fillna(df["f_tool_dmin"].median()).to_numpy()
    rng = np.random.default_rng(0)
    rand = rng.random(len(df))
    scores["tool_threshold"] = tool
    scores["random"] = rand

    boot = W.bootstrap_ap(y, scores, mask, groups, n_boot=1000,
                          diffs=[("model", "phase_baseline"), ("residuals_only", "phase_baseline"),
                                 ("model_plus_phase", "phase_baseline"), ("model", "random"),
                                 ("residuals_only", "model")])
    ablation = {k: dict(ap=r(W.ap(y, s, mask)), lo=r(boot[k]["lo"]), hi=r(boot[k]["hi"])) for k, s in scores.items()}
    diffs = {k: {kk: r(vv) for kk, vv in v.items()} for k, v in boot.items() if "-" in k}

    # ---- time-ordered holdout: train on morning segments 0-1, test on afternoon segment 2 ----
    tX = {k: groups_X[k] for k in ["model", "residuals_only", "residuals_plus_phase", "model_plus_phase"]}
    tX["phase_baseline"] = None
    tord = W.time_ordered(df, tX, y, mask)
    te = (df["segment"].to_numpy() == 2) & mask
    from sklearn.metrics import average_precision_score as _aps
    tord["tool_threshold"] = float(_aps(y[te], tool[te]))
    tord["random"] = float(_aps(y[te], rand[te]))
    # bootstrap CI on the segment-2 test set (cycles resampled within the test segment)
    segv = df["segment"].to_numpy()
    trm = np.isin(segv, (0, 1)) & mask
    ts = {}
    for k in ["model", "residuals_plus_phase"]:
        sc = np.full(len(df), np.nan)
        sc[te] = W.make_gbm().fit(groups_X[k][trm], y[trm]).predict_proba(groups_X[k][te])[:, 1]
        ts[k] = sc
    ts["phase_baseline"] = W.phase_rate_score(W.phase_rate_fit(phase[trm], y[trm]), phase)
    ts["random"] = rand
    tboot = W.bootstrap_ap(y, ts, te, groups, n_boot=1000,
                           diffs=[("model", "phase_baseline"), ("residuals_plus_phase", "phase_baseline"),
                                  ("phase_baseline", "random")])
    tord = {k: (r(v) if isinstance(v, float) else v) for k, v in tord.items()}
    tord["ci"] = {k: {kk: r(vv) for kk, vv in v.items()} for k, v in tboot.items()}

    observed_h = rel["observed_s"] / 3600.0
    sweeps = {}
    summaries = {}
    for name in ["model", "phase_baseline", "tool_threshold", "random"]:
        s = scores[name]
        pr = W.pr_summary(y, s, mask)
        rows = []
        for q in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99]:
            thr = float(np.nanquantile(s[mask], q))
            alert = (s >= thr) & mask
            caught, leads = W.episode_catch(s, thr, onsets, mask, seg)
            fa = int((alert & (y == 0)).sum())
            prec = float((alert & (y == 1)).sum() / max(alert.sum(), 1))
            rows.append(dict(q=q, alert_share=r(alert.sum() / mask.sum()), caught=r(caught),
                             false_alarms_per_h=r(fa / observed_h, 1), precision=r(prec),
                             median_lead_s=r(np.median(leads) if leads else None, 1),
                             leads=[int(v) for v in leads]))
        sweeps[name] = rows
        step = max(1, len(pr["precision"]) // 200)
        summaries[name] = dict(ap=r(pr["ap"]), curve=[[r(a), r(b)] for a, b in
                                                     zip(pr["recall"][::step], pr["precision"][::step])])

    # ---- residual tail before onsets: P95 and mean of |z| in the 10 s before an onset vs other clean rows ----
    pre = np.zeros(len(df), bool)
    pre[(y == 1) & mask] = True
    buildup = []
    for j in range(6):
        a = df[f"Res_J{j}"].abs().to_numpy()
        mx = df[f"f_res{j}_max"].to_numpy()
        cl = clean & ~pre
        buildup.append(dict(joint=f"J{j}", r2=r(load_stats[f"J{j}"]["r2_clean"]),
                            sd_A=r(load_stats[f"J{j}"]["resid_sd_A"]),
                            mean_pre=r(np.nanmean(a[pre])), mean_clean=r(np.nanmean(a[cl])),
                            p95_pre=r(np.nanpercentile(a[pre], 95)), p95_clean=r(np.nanpercentile(a[cl], 95)),
                            winmax_med_pre=r(np.nanmedian(mx[pre])), winmax_med_clean=r(np.nanmedian(mx[cl])),
                            share_gt3_pre=r(np.nanmean(a[pre] > 3)), share_gt3_clean=r(np.nanmean(a[cl] > 3))))

    # ---- cycle-phase hazard ----
    onset_idx = dict(stop=R.rising_edges(df["stop"], df["segment"]),
                     grip=R.rising_edges(df["grip"].astype(int), df["segment"]), any=onsets)
    haz = P.hazard_table(df, onset_idx, down)
    hot = {k: P.hot_window(df, v, down) for k, v in onset_idx.items()}
    hot["stop_2_5"] = P.hot_window(df, onset_idx["stop"], down, (2, 5))
    hot["grip_15_30"] = P.hot_window(df, onset_idx["grip"], down, (15, 30))
    hot_by_seg = {}
    for sgi in sorted(df["segment"].unique()):
        m = (df["segment"] == sgi).to_numpy()
        sub = df[m].reset_index(drop=True)
        hot_by_seg[int(sgi)] = P.hot_window(sub, R.rising_edges(sub["stop"], sub["segment"]), down[m])
    profile = P.phase_profile(df, down, onset_idx, 15)
    par = P.pareto(df, ep)
    stop_rate_h = rel["n_stop_onsets"] / observed_h
    power = dict(rr=0.5, alpha=0.05, power=0.8, events_baseline_arm=r(P.events_needed(), 1),
                 hours_per_arm={f"{x}": r(P.run_hours_needed(x), 1) for x in [0.125, 0.25, 0.625, 1.25, stop_rate_h]},
                 rig_stop_rate_per_h=r(stop_rate_h, 1))

    # ---- reliability statistics ----
    ia = R.interarrivals(df, onsets)
    wb = R.weibull_fit(ia)
    lo, hi = R.mtbf_ci(rel["up_s"], rel["n_episodes"])
    aci = R.availability_ci(df)
    wb_seg = {}
    for sgi in sorted(df["segment"].unique()):
        m = (df["segment"] == sgi).to_numpy()
        x = R.interarrivals(df, onsets[m[onsets]])
        if len(x) >= 20:
            wb_seg[int(sgi)] = {k: r(v, 4) for k, v in R.weibull_fit(x, n_boot=500).items()}
    rel_stats = dict(weibull={k: r(v, 4) for k, v in wb.items()}, weibull_by_segment=wb_seg, mtbf_lo=r(lo, 2), mtbf_hi=r(hi, 2),
                     availability_ci={k: r(v, 4) for k, v in aci.items()})

    # timeline (all rows, compact)
    t0 = df["t"].iloc[0]
    tl = dict(
        t=[r((x - t0).total_seconds(), 1) for x in df["t"]],
        segment=df["segment"].astype(int).tolist(),
        cycle=df["cycle"].astype(int).tolist(),
        tool=[r(v, 4) for v in df["Tool_current"]],
        j1=[r(v, 3) for v in df["Current_J1"]],
        j2=[r(v, 3) for v in df["Current_J2"]],
        risk=[r(v, 3) for v in oof],
        stop=[None if np.isnan(v) else int(v) for v in df["stop"]],
        grip=df["grip"].astype(int).tolist(),
    )
    episodes = [dict(start=int(e.start), end=int(e.end), cycle=int(e.cycle), dur=r(e.duration_s, 2),
                     kind=("stop+grip" if e.stop and e.grip else "stop" if e.stop else "grip"))
                for e in ep.itertuples()]

    tp = T.sensitivity(case_rate=12.0, shift_h=8.0, target_a=0.996)
    tp["hot_window_stop_share"] = r(hot["stop"]["share"])
    tp["stops_removed_if_hot_rate_halved"] = r(0.5 * hot["stop"]["share"])

    summary = dict(
        dataset=dict(rows=int(len(df)), cols=24, segments=int(df["segment"].nunique()),
                     cycles=int(df["cycle"].nunique()), missing_cells=1072,
                     unknown_stop_labels=int(df["stop"].isna().sum())),
        reliability={k: r(v, 4) for k, v in rel.items()},
        reliability_stats=rel_stats,
        load_models=buildup,
        phase=dict(bins=haz, hot=hot, hot_by_segment=hot_by_seg, profile=profile, pareto=par, power=power),
        warning=dict(horizon_s=W.HORIZON_S, window=W.WIN, base_rate=r(y[mask].mean()),
                     scored_rows=int(mask.sum()), summaries=summaries, sweeps=sweeps,
                     features=cols, cv="5-fold GroupKFold by cycle",
                     ablation=ablation, diffs=diffs, time_ordered=tord, bootstrap="1000 cycle-block resamples"),
        throughput=tp,
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    with open(os.path.join(out, "timeline.json"), "w") as f:
        json.dump(dict(series=tl, episodes=episodes), f, separators=(",", ":"))
    print(json.dumps(dict(reliability=summary["reliability"], rel_stats=rel_stats, ablation=ablation, diffs=diffs,
                          time_ordered=tord, q95={k: next(x for x in v if x["q"] == 0.95) | {"leads": "..."} for k, v in sweeps.items()},
                          buildup=buildup, hot=hot, hot_by_seg=hot_by_seg, pareto={k: v for k, v in par.items() if k != "curve"},
                          power=power, throughput=tp), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
