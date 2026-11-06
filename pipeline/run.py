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

    oof = W.cross_validate(df, cols, y, mask)
    tool = -df["f_tool_dmin"].fillna(df["f_tool_dmin"].median()).to_numpy()
    rng = np.random.default_rng(0)
    rand = rng.random(len(df))

    observed_h = rel["observed_s"] / 3600.0
    sweeps = {}
    summaries = {}
    for name, s in [("model", oof), ("tool_threshold", tool), ("random", rand)]:
        pr = W.pr_summary(y, s, mask)
        rows = []
        for q in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99]:
            thr = float(np.nanquantile(s[mask], q))
            caught, leads = W.episode_catch(s, thr, onsets, mask, seg)
            alert = (s >= thr) & mask
            fa = int((alert & (y == 0)).sum())
            prec = float((alert & (y == 1)).sum() / max(alert.sum(), 1))
            rows.append(dict(q=q, caught=r(caught), false_alarms_per_h=r(fa / observed_h, 1),
                             precision=r(prec), median_lead_s=r(np.median(leads) if leads else None, 1),
                             leads=[int(v) for v in leads]))
        sweeps[name] = rows
        step = max(1, len(pr["precision"]) // 200)
        summaries[name] = dict(ap=r(pr["ap"]), curve=[[r(a), r(b)] for a, b in
                                                     zip(pr["recall"][::step], pr["precision"][::step])])

    # residual build-up before onsets, per joint: mean |z| in the 10 s before onset vs clean rows
    pre = np.zeros(len(df), bool)
    pre[(y == 1) & mask] = True
    buildup = []
    for j in range(6):
        a = df[f"Res_J{j}"].abs().to_numpy()
        buildup.append(dict(joint=f"J{j}", pre=r(np.nanmean(a[pre])), clean=r(np.nanmean(a[clean & ~pre])),
                            r2=r(load_stats[f"J{j}"]["r2_clean"]), sd_A=r(load_stats[f"J{j}"]["resid_sd_A"])))

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

    default_rate, default_shift = 12.0, 8.0
    model_95 = next(x for x in sweeps["model"] if x["q"] == 0.95)
    tp = T.cases_lost(default_rate, default_shift, rel["availability"], model_95["caught"])

    summary = dict(
        dataset=dict(rows=int(len(df)), cols=24, segments=int(df["segment"].nunique()),
                     cycles=int(df["cycle"].nunique()), missing_cells=1072,
                     unknown_stop_labels=int(df["stop"].isna().sum())),
        reliability={k: r(v, 4) for k, v in rel.items()},
        load_models=buildup,
        warning=dict(horizon_s=W.HORIZON_S, window=W.WIN, base_rate=r(y[mask].mean()),
                     scored_rows=int(mask.sum()), summaries=summaries, sweeps=sweeps,
                     features=cols, cv="5-fold GroupKFold by cycle"),
        throughput=dict(default_rate=default_rate, default_shift_h=default_shift, at_q95=tp),
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    with open(os.path.join(out, "timeline.json"), "w") as f:
        json.dump(dict(series=tl, episodes=episodes), f, separators=(",", ":"))
    print(json.dumps(dict(reliability=summary["reliability"], ap={k: v["ap"] for k, v in summaries.items()},
                          q90={k: next(x for x in v if x["q"] == 0.9) | {"leads": "..."} for k, v in sweeps.items()},
                          buildup=buildup, throughput=tp), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
