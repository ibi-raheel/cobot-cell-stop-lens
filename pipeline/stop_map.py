"""Stop map: one cell log in, a ranked fix list out.

A thin wrapper over the existing pipeline modules (reliability, phase, learned, throughput). It ranks the
moves of the cycle by onset rate for each failure type, names the hot move with its rate ratio, re-checks it
per recording session, estimates what fixing it is worth, prices a stop against an uptime budget, and sizes
the before/after trial.

Usage:
  OMP_NUM_THREADS=3 python -I pipeline/stop_map.py downloads/dataset_02052023.xlsx --window 2-5 --grip-window 15-30
  options: --window A-B|auto  --grip-window A-B|auto  --recovery-s 120  --case-rate 12  --shift-h 8
           --target 0.996  --json out.json
Input: an xlsx in the UR3 CobotOps column format (Timestamp, cycle, Robot_ProtectiveStop, grip_lost, ...).
A controller log mapped to those columns works the same way, with the program waypoint index standing in for
the cycle sample.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import reliability as R
import phase as P
import learned as LM
import throughput as T

ACTION = dict(stop="test speed/accel, payload/TCP or collision sensitivity on this move",
              grip="test gripper force / transfer acceleration on this move")


def _parse(w):
    if w is None or w == "auto":
        return None
    a, b = w.split("-")
    return int(a), int(b)


def _fix(df, idx, down, win, ep, rel, n_boot=1000, seed=0):
    """Stops removed if the window ran at the cycle-average rate (same arithmetic as run_learned.py)."""
    ph = df["phase"].to_numpy()
    dtv = df["dt"].fillna(df["dt"].median()).to_numpy()
    risk = P.at_risk(df, down)
    inw = (ph >= win[0]) & (ph <= win[1])
    e_in, e_out = float(dtv[inw & risk].sum()), float(dtv[~inw & risk].sum())
    n_in, n_out = int(inw[idx].sum()), int((~inw[idx]).sum())
    stop_ep, other = ep[ep["stop"]], ep[~ep["stop"]]
    fx = LM.segment_fix_gain(n_in, e_in, n_out, e_out, float(stop_ep["duration_s"].mean()), rel["observed_s"],
                             int(len(other)), float(other["duration_s"].sum()))
    # interval: resample blocks of 5 consecutive cycle runs
    blocks = df["cycle_run"].to_numpy() // 5
    ub = np.unique(blocks)
    bi = np.searchsorted(ub, blocks)
    in_k = np.bincount(bi[idx[inw[idx]]], minlength=len(ub))
    out_k = np.bincount(bi[idx[~inw[idx]]], minlength=len(ub))
    ein = np.bincount(bi, dtv * (inw & risk), minlength=len(ub))
    eout = np.bincount(bi, dtv * (~inw & risk), minlength=len(ub))
    rng, sh = np.random.default_rng(seed), []
    for _ in range(n_boot):
        p = rng.integers(0, len(ub), len(ub))
        a, b, c, d = in_k[p].sum(), ein[p].sum(), out_k[p].sum(), eout[p].sum()
        sh.append((a - (a + c) / (b + d) * b) / (a + c))
    up = e_in + e_out
    fx.update(share_lo=float(np.percentile(sh, 2.5)), share_hi=float(np.percentile(sh, 97.5)),
              per_up_h_before=3600 * (n_in + n_out) / up, per_up_h_after=3600 * (n_in + n_out - fx["stops_removed"]) / up)
    return fx


def stop_map(path, window=None, grip_window=None, recovery_s=120.0, case_rate=12.0, shift_h=8.0, target=0.996):
    df = R.load(path)
    ep = R.episodes(df)
    rel = R.reliability(df, ep)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    df = P.add_phase(df)
    onsets = dict(stop=R.rising_edges(df["stop"], df["segment"]),
                  grip=R.rising_edges(df["grip"].astype(int), df["segment"]))
    bins = P.hazard_table(df, onsets, down)
    out = dict(observed_h=rel["observed_s"] / 3600, n_stops=int(len(onsets["stop"])), n_grip=int(len(onsets["grip"])),
               bins=[dict(bin=b["bin"], lo=b["lo"], hi=b["hi"], share_up=b["share_up"],
                          stop=b["stop"], grip=b["grip"]) for b in bins], kinds={})
    for kind, given in (("stop", _parse(window)), ("grip", _parse(grip_window))):
        ranked = sorted(bins, key=lambda b: -b[kind]["rate100"])
        win = given or (ranked[0]["lo"], ranked[0]["hi"])
        hot = P.hot_window(df, onsets[kind], down, win)
        by_seg = {}
        for s in sorted(df["segment"].unique()):
            m = (df["segment"] == s).to_numpy()
            sub = df[m].reset_index(drop=True)
            idx = R.rising_edges(sub[kind].astype(float) if kind == "stop" else sub[kind].astype(int), sub["segment"])
            by_seg[int(s)] = P.hot_window(sub, idx, down[m], win)
        k = dict(window=list(win), picked="given" if given else "auto", hot=hot, by_segment=by_seg,
                 is_hot=bool(hot["rr_lo"] > 1), action=ACTION[kind],
                 runner_up=dict(bin=ranked[1]["bin"], rate100=ranked[1][kind]["rate100"],
                                lo100=ranked[1][kind]["lo100"], hi100=ranked[1][kind]["hi100"]),
                 top=dict(bin=ranked[0]["bin"], rate100=ranked[0][kind]["rate100"],
                          lo100=ranked[0][kind]["lo100"], hi100=ranked[0][kind]["hi100"]))
        if kind == "stop":
            k["fix"] = _fix(df, onsets["stop"], down, win, ep, rel)
            k["fix"]["label"] = "upper-end: window picked on this data" if not given else \
                "upper-end unless the window was fixed before looking at this data"
        out["kinds"][kind] = k
    stop_rate_h = out["n_stops"] / out["observed_h"]
    out["cost"] = dict(recovery_s=recovery_s, case_rate=case_rate, shift_h=shift_h, target=target,
                       budget_min=T.budget_min(shift_h, target), cases_per_stop=T.cases_lost(1, recovery_s, case_rate),
                       max_stops=T.max_stops(shift_h, target, recovery_s), assumed=True)
    out["trial"] = dict(events_baseline_arm=P.events_needed(0.5), stops_per_h=stop_rate_h,
                        hours_per_arm=P.run_hours_needed(stop_rate_h),
                        hours_per_arm_1_per_shift=P.run_hours_needed(1 / shift_h),
                        hours_per_arm_10_per_shift=P.run_hours_needed(10 / shift_h))
    return out


def report(m) -> str:
    ci = lambda a, b, k=2: f"[{a:.{k}f}, {b:.{k}f}]"
    L = [f"STOP MAP  {m['observed_h']:.2f} h observed · {m['n_stops']} protective stops · {m['n_grip']} grip losses"]
    for kind, name in (("stop", "Protective stops"), ("grip", "Grip losses")):
        k = m["kinds"][kind]
        h = k["hot"]
        L.append(f"{name} by cycle sample   n   rate/100 up-s [95% CI]")
        for b in sorted(m["bins"], key=lambda b: -b[kind]["rate100"]):
            tag = ""
            if [b["lo"], b["hi"]] == k["window"]:
                tag = (f"   {'HOT' if k['is_hot'] else 'not hot'}  RR {h['rr']:.1f} {ci(h['rr_lo'], h['rr_hi'], 1)}"
                       f" · {h['share']:.0%} of onsets in {h['up_share']:.0%} of up-time")
            L.append(f"  {b['bin']:<6}{b[kind]['n']:>4}   {b[kind]['rate100']:.2f} {ci(b[kind]['lo100'], b[kind]['hi100'])}{tag}")
        segs = []
        for v in k["by_segment"].values():
            ok = v["rr_lo"] > 1 if v["n_in"] + v["n_out"] else False
            segs.append(f"{v['rr']:.1f}{'' if ok else ' (n.s.)'}" if np.isfinite(v["rr"]) else "n/a")
        L.append(f"  per session RR: {' / '.join(segs)}")
        if k["picked"] == "auto":
            L.append(f"  auto window {k['top']['bin']} ({k['top']['rate100']:.2f}) vs runner-up {k['runner_up']['bin']}"
                     f" ({k['runner_up']['rate100']:.2f}); intervals "
                     f"{'overlap, a narrow win' if k['runner_up']['hi100'] >= k['top']['lo100'] else 'do not overlap'}")
    s, g = m["kinds"]["stop"], m["kinds"]["grip"]
    L.append(f"Test first: stops -> {s['action']} (samples {s['window'][0]}-{s['window'][1]})")
    L.append(f"            drops -> {g['action']} (samples {g['window'][0]}-{g['window'][1]})")
    fx = s["fix"]
    L.append(f"Fix value ({fx['label']}): {fx['stops_removed']:.1f} of {m['n_stops']} stops "
             f"({fx['share_removed']:.0%} [{fx['share_lo']:.0%}, {fx['share_hi']:.0%}]) · "
             f"{fx['per_up_h_before']:.0f} -> {fx['per_up_h_after']:.0f} stops/up-h")
    c = m["cost"]
    L.append(f"Cost @ {c['recovery_s']:.0f} s recovery, {c['case_rate']:.0f} cases/min (assumed): "
             f"{c['cases_per_stop']:.0f} cases/stop · budget {c['budget_min']:.2f} min/shift fits {c['max_stops']:.2f} stops")
    t = m["trial"]
    L.append(f"Trial: {t['events_baseline_arm']:.0f} baseline stops for a 50% drop at 80% power · "
             f"{t['hours_per_arm']:.1f} h/arm here · {t['hours_per_arm_10_per_shift']:.0f} h/arm at 10 stops/shift")
    L.append("Optional next step: grip-loss early-warning model (pipeline/run_learned.py, about 7 min on CPU).")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--window", default="auto")
    ap.add_argument("--grip-window", default="auto")
    ap.add_argument("--recovery-s", type=float, default=120.0)
    ap.add_argument("--case-rate", type=float, default=12.0)
    ap.add_argument("--shift-h", type=float, default=8.0)
    ap.add_argument("--target", type=float, default=0.996)
    ap.add_argument("--json")
    a = ap.parse_args()
    m = stop_map(a.log, a.window, a.grip_window, a.recovery_s, a.case_rate, a.shift_h, a.target)
    print(report(m))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(m, f, indent=1, default=float)
