import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { Data } from "./data";
import { JOINTS, ci, f, integratePose, loadAll, pct, sgn } from "./data";
import { Arm, Readouts } from "./Arm";
import { ApChart, CostLadder, CycleStrip, FixGrid, ModelSchematic, RateRatioPlot, RunLog, type ApRow } from "./charts";

const STEPS = [
  { id: "cell", short: "The cell" },
  { id: "cost", short: "What a stop costs" },
  { id: "where", short: "Where stops start" },
  { id: "warn", short: "Grip-loss warning" },
  { id: "fix", short: "Fixing the move" },
  { id: "proof", short: "How we know" },
];

export function App() {
  const [d, setD] = useState<Data | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { loadAll().then(setD).catch((e) => setErr(String(e))); }, []);
  const active = useActiveStep(!!d);

  return (
    <div className="page">
      <header className="bar">
        <span className="bar-id"><i className="bar-led" aria-hidden />CELL STOP LENS</span>
        <span className="bar-meta">UR3 cobot &middot; real telemetry &middot; {d ? `${f(d.s.dataset.rows, 0)} samples` : "loading"}</span>
      </header>

      <div className="progress" aria-hidden><span style={{ transform: `scaleX(${(active + 1) / STEPS.length})` }} /></div>

      <div className="layout">
        <nav className="rail" aria-label="Steps">
          <ol>
            {STEPS.map((s, i) => (
              <li key={s.id} className={i === active ? "is-on" : i < active ? "is-done" : ""}>
                <a href={`#${s.id}`} aria-current={i === active ? "step" : undefined}>
                  <span className="rail-n">{String(i + 1).padStart(2, "0")}</span>
                  <span className="rail-t">{s.short}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <main className="main">
          <section className="intro">
            <p className="kicker">Cobot cell health</p>
            <h1>Where a robot cell loses its time, and which move to fix first.</h1>
            <p className="lede">Two hours of real telemetry from a six-axis collaborative arm, walked through in six steps.</p>
          </section>

          {err && <p className="err" role="alert">Could not load the results: {err}</p>}
          {!d && !err && <p className="loading">Loading results&hellip;</p>}
          {d && <Story d={d} />}

          <footer className="foot">
            <p>
              Data: <a href="https://doi.org/10.24432/C5J891">UR3 CobotOps</a>, Tyrovolas, Aliev, Antonelli and Stylios (2024),
              UCI Machine Learning Repository, licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
              Real telemetry, no synthetic data. Case rate, shift length and the uptime target are assumptions, not measurements.
            </p>
            <p className="foot-more">More work at <a href="https://ibiraheel.com">ibiraheel.com</a></p>
          </footer>
        </main>
      </div>
    </div>
  );
}

function useActiveStep(ready: boolean): number {
  const [a, setA] = useState(0);
  useEffect(() => {
    if (!ready) return;
    const els = STEPS.map((s) => document.getElementById(s.id)).filter(Boolean) as HTMLElement[];
    const onScroll = () => {
      const mid = window.innerHeight * 0.4;
      let k = 0;
      els.forEach((el, i) => { if (el.getBoundingClientRect().top < mid) k = i; });
      setA(k);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, [ready]);
  return a;
}

function Step({ n, id, title, lead, panel, children, details }: {
  n: number; id: string; title: string; lead: ReactNode; panel: string; children: ReactNode; details?: ReactNode;
}) {
  return (
    <section className="step" id={id} aria-labelledby={`${id}-h`}>
      <div className="step-head">
        <span className="step-n">{String(n).padStart(2, "0")}</span>
        <h2 id={`${id}-h`}>{title}</h2>
      </div>
      <p className="step-lead">{lead}</p>
      <div className="panel">
        <div className="panel-bar"><span>{panel}</span><i aria-hidden /></div>
        <div className="panel-body">{children}</div>
      </div>
      {details && (
        <details className="det">
          <summary>Show details</summary>
          <div className="det-body">{details}</div>
        </details>
      )}
    </section>
  );
}

function Story({ d }: { d: Data }) {
  const { s, tl } = d;
  const r = s.reliability;
  const hot = s.phase.hot.stop_2_5;
  const tp = s.throughput;
  const manual = tp.rows.find((x) => x.mttr_s === 120)!;

  return (
    <>
      {/* 01 ---------------------------------------------------------------- */}
      <Step n={1} id="cell" title="A robot cell, running"
        panel={`RUN LOG · ${s.dataset.segments} SESSIONS · ${f(r.observed_s / 3600, 1)} H`}
        lead={<>A UR3 arm picks and places for {f(r.observed_s / 60, 0)} minutes across {s.dataset.cycles} cycles. It goes down {f(r.n_episodes, 0)} times: {f(r.n_stop_onsets, 0)} protective stops and {f(r.n_grip_onsets, 0)} grip losses.</>}
        details={<>
          <table className="tbl">
            <tbody>
              <tr><th>Samples, columns</th><td>{f(s.dataset.rows, 0)} &times; {s.dataset.cols}, about 1 s apart</td></tr>
              <tr><th>Recording sessions</th><td>{s.dataset.segments}, split at recording gaps</td></tr>
              <tr><th>Up time / down time</th><td>{f(r.up_s, 0)} s / {f(r.down_s, 0)} s</td></tr>
              <tr><th>Down episodes</th><td>{f(r.n_episodes, 0)} (a stop and a grip loss can share one episode)</td></tr>
              <tr><th>Missing cells</th><td>{f(s.dataset.missing_cells, 0)}; {s.dataset.unknown_stop_labels} unknown stop labels</td></tr>
            </tbody>
          </table>
        </>}>
        <RunLog tl={tl} />
      </Step>

      {/* 02 ---------------------------------------------------------------- */}
      <Step n={2} id="cost" title="Every stop costs cases"
        panel={`DOWNTIME BUDGET · ${pct(tp.target_availability, 1)} UPTIME · ${tp.shift_h} H SHIFT`}
        lead={<>This rig resets itself in about {f(r.mttr_s, 1)} s. On a line, a {f(manual.mttr_s / 60, 0)}-minute manual reset loses {f(manual.cases_per_stop, 0)} cases, and a {pct(tp.target_availability, 1)} uptime budget fits fewer than one of those per shift.</>}
        details={<>
          <p>Budget = (1 &minus; {pct(tp.target_availability, 1)}) &times; {tp.shift_h} h = {f(tp.budget_min, 2)} min of downtime per shift. Cases lost per stop = recovery time &times; {f(tp.case_rate, 0)} cases/min. Case rate, shift length and target are assumptions for a palletizing line, not measurements from this rig.</p>
          <table className="tbl tbl-num">
            <thead><tr><th>Recovery</th><th>Cases / stop</th><th>Stops in budget</th></tr></thead>
            <tbody>{tp.rows.map((x) => <tr key={x.mttr_s}><td>{x.mttr_s < 60 ? `${x.mttr_s} s` : `${x.mttr_s / 60} min`}</td><td>{f(x.cases_per_stop, x.cases_per_stop < 1 ? 2 : 0)}</td><td>{f(x.max_stops_in_budget, 2)}</td></tr>)}</tbody>
          </table>
        </>}>
        <CostLadder tp={tp} />
      </Step>

      {/* 03 ---------------------------------------------------------------- */}
      <WhereStep d={d} />

      {/* 04 ---------------------------------------------------------------- */}
      <WarnStep d={d} />

      {/* 05 ---------------------------------------------------------------- */}
      <FixStep d={d} />

      {/* 06 ---------------------------------------------------------------- */}
      <Step n={6} id="proof" title="How we know"
        panel="STOP RATE RATIO · SAMPLES 2–5 VS REST OF CYCLE"
        lead={<>The window holds in each recording session that has enough stops. Availability and MTBF are checked by hand, and every model is scored on blocks of time it never saw.</>}
        details={<ProofDetails d={d} />}>
        <RateRatioPlot pooled={hot} bySeg={s.phase.hot_by_segment_2_5} />
        <dl className="checks">
          <div><dt>Availability</dt><dd>{f(r.up_s, 1)} / {f(r.observed_s, 1)} s = <b>{pct(r.availability, 1)}</b> <span>{ci(s.reliability_stats.availability_ci.lo * 100, s.reliability_stats.availability_ci.hi * 100)}%</span></dd></div>
          <div><dt>MTBF</dt><dd>{f(r.up_s, 1)} s / {f(r.n_episodes, 0)} = <b>{f(r.mtbf_s, 1)} s</b> <span>{ci(s.reliability_stats.mtbf_lo, s.reliability_stats.mtbf_hi)} s</span></dd></div>
          <div><dt>MTTR</dt><dd>{f(r.down_s, 1)} s / {f(r.n_episodes, 0)} = <b>{f(r.mttr_s, 2)} s</b></dd></div>
          <div><dt>Wear-out</dt><dd>Weibull shape <b>{f(s.reliability_stats.weibull.shape, 2)}</b> <span>{ci(s.reliability_stats.weibull.shape_lo, s.reliability_stats.weibull.shape_hi, 2)}</span>, no rising hazard inside two hours</dd></div>
        </dl>
      </Step>
      <p className="handoff">Next on a real line: tag stops by program waypoint, change one variable on the hottest move, and compare. Spotting a 50% drop with 80% power needs {f(s.phase.power.events_baseline_arm, 0)} stops in the baseline arm.</p>
    </>
  );
}

/* ---------------------------------------------------------------- step 3 */

function WhereStep({ d }: { d: Data }) {
  const prof = d.s.phase.profile;
  const hot = d.s.phase.hot.stop_2_5;
  const poses = useMemo(() => integratePose(prof), [prof]);
  const [sample, setSample] = useState(3);
  const [joint, setJoint] = useState(2);
  const p = prof[sample];
  const inWin = p.phase >= 2 && p.phase <= 5;
  const fastest = prof.reduce((b, r, i) => (Math.abs(r.speed[2]) > Math.abs(prof[b].speed[2]) ? i : b), 0);
  const g = d.s.phase.hot.grip_15_30;
  return (
    <Step n={3} id="where" title="Stops start right after the J2 swing"
      panel={`CYCLE TIMELINE · ${JOINTS[joint].id} ${JOINTS[joint].name.toUpperCase()}`}
      lead={<>{hot.n_in} of {hot.n_in + hot.n_out} protective stops ({pct(hot.share, 0)}) start in cycle samples 2–5, just after the fastest J2 swing. That window is {pct(hot.up_share, 0)} of up-time, so its stop rate is {f(hot.rr, 0)}&times; the rest of the cycle.</>}
      details={<>
        <p>Rate ratio {f(hot.rr, 1)} {ci(hot.rr_lo, hot.rr_hi)} (exact Poisson). The fastest mean J2 speed is at sample {prof[fastest].phase} ({f(prof[fastest].speed[2], 2)} rad/s), after the tool-current peak at sample 1, likely the gripper acting. This locates the stops in one move; it does not say what in that move causes them.</p>
        <p>Grip losses cluster elsewhere: {g.n_in} of {g.n_in + g.n_out} start in samples 15–30 (rate ratio {f(g.rr, 1)} {ci(g.rr_lo, g.rr_hi)}), the transfer and place part of the cycle. That points to a different fix.</p>
        <table className="tbl tbl-num">
          <thead><tr><th>Cycle samples</th><th>Share of up-time</th><th>Stops</th><th>Stops / 100 s</th><th>Grip losses</th></tr></thead>
          <tbody>{d.s.phase.bins.map((b) => (
            <tr key={b.bin}><td>{b.bin}</td><td>{pct(b.share_up, 0)}</td><td>{b.stop.n}</td><td>{f(b.stop.rate100, 2)} {ci(b.stop.lo100, b.stop.hi100, 2)}</td><td>{b.grip.n}</td></tr>
          ))}</tbody>
        </table>
        <p className="small">Arm pose is a sketch: angles are integrated from the mean joint speed per sample, relative to cycle start.</p>
      </>}>
      <div className="pendant">
        <div className={`pendant-arm${inWin ? " is-hot" : ""}`}>
          <div className="pendant-tag">
            <span>Sample <b>{String(p.phase).padStart(2, "0")}</b></span>
            <span className={inWin ? "tag-hot" : ""}>{inWin ? "In stop window" : "Outside window"}</span>
          </div>
          <Arm pose={poses[sample]} selected={joint} onSelect={setJoint} hot={inWin && joint === 2} />
        </div>
        <Readouts speed={p.speed} pose={poses[sample]} selected={joint} onSelect={setJoint} />
      </div>
      <CycleStrip profile={prof} joint={joint} sample={sample} onSample={setSample} hot={hot} />
    </Step>
  );
}

/* ---------------------------------------------------------------- step 4 */

type Mode = "grip" | "grip_quiet" | "grip_causal" | "grip_notool" | "stop";

function WarnStep({ d }: { d: Data }) {
  const [mode, setMode] = useState<Mode>("grip");
  const gl = d.l.tasks.grip_lead3, st = d.l.tasks.stop_10;
  const sh = d.k.shipped_grip_lead3, ca = d.k.causal_grip_lead3;
  const BK = "Baseline: cycle phase + time since last stop and last grip loss";
  const BR = "Baseline: cycle phase + time since last down episode";
  const CNN = "Learned model (1D-CNN)";

  const view = useMemo((): { rows: ApRow[]; chance: number; verdict: ReactNode; max: number } => {
    if (mode === "stop") {
      const df = st.diff_ap["cnn-phase_recency"];
      return {
        chance: st.base_rate, max: 0.7,
        rows: [
          { key: "cnn", label: CNN, model: true, ...st.ap.cnn },
          { key: "pr", label: BR, ...st.ap.phase_recency },
          { key: "pb", label: "Baseline: cycle phase only", ...st.ap.phase_baseline },
        ],
        verdict: <>No clear win. The model edges the phase-only baseline but not phase + recency (difference {sgn(df.v, 3)}, interval {ci(df.lo, df.hi, 2)} spans zero). For stops, the cycle position from step 3 is most of the story.</>,
      };
    }
    if (mode === "grip_quiet") {
      const q = sh.no_recent_grip_loss["25"];
      return {
        chance: q.base_rate, max: 0.7,
        rows: [
          { key: "cnn", label: CNN, model: true, v: q.ap.cnn, lo: q.cnn_ci[0], hi: q.cnn_ci[1] },
          { key: "bk", label: BK, v: q.ap.phase_recency_by_kind },
          { key: "pr", label: BR, v: q.ap.phase_recency },
        ],
        verdict: <>Only rows with no grip loss in the previous 25 s, so the model cannot lean on runs of grip losses. It still scores {f(q.ap.cnn, 2)} against {f(q.ap.phase_recency_by_kind, 2)}.</>,
      };
    }
    const cnn = mode === "grip_causal" ? ca.ap.cnn : mode === "grip_notool" ? gl.ap.cnn_no_tool : sh.ap.cnn;
    const df = mode === "grip_causal" ? ca.diff_ap["cnn-phase_recency_by_kind"] : sh.diff_ap["cnn-phase_recency_by_kind"];
    const verdict = mode === "grip_causal"
      ? <>Rebuilt with strictly causal features (no filter that looks ahead). The result holds: {f(cnn.v, 3)}.</>
      : mode === "grip_notool"
        ? <>With the gripper-current channel removed the model scores {f(cnn.v, 3)}, so it is not just reading the gripper.</>
        : <>The model wins. Its edge over the stronger baseline is {sgn(df.v, 2)} {ci(df.lo, df.hi, 2)}, an interval that stays above zero.</>;
    return {
      chance: sh.base_rate, max: 0.7,
      rows: [
        { key: "cnn", label: CNN, model: true, ...cnn },
        { key: "bk", label: BK, ...sh.ap.phase_recency_by_kind },
        { key: "pr", label: BR, ...sh.ap.phase_recency },
      ],
      verdict,
    };
  }, [mode, gl, st, sh, ca]);

  const opts: { v: Mode; l: string }[] = [
    { v: "grip", l: "All rows" },
    { v: "grip_quiet", l: "No recent grip loss" },
    { v: "grip_causal", l: "Causal features" },
    { v: "grip_notool", l: "No gripper channel" },
  ];
  const isStop = mode === "stop";

  return (
    <Step n={4} id="warn" title="A learned model warns of grip losses"
      panel={isStop ? "PREDICT · PROTECTIVE STOP WITHIN 10 S" : "PREDICT · GRIP LOSS 3–10 S AHEAD"}
      lead={<>A small neural network reads the last {d.l.window_samples} s of {d.l.channels} sensor channels and warns of a grip loss 3 to 10 s ahead. It beats the best simple rule by a clear margin; for protective stops it does not.</>}
      details={<>
        <p><b>Score.</b> PR-AUC: the average precision over all alert thresholds. A scorer that guesses scores the event rate ({f(sh.base_rate, 3)} for grip losses). Higher is better; intervals are 95% from a block bootstrap.</p>
        <p><b>Validation.</b> {d.l.validation}.</p>
        <p><b>Why the stronger baseline.</b> Grip losses come in runs ({sh.grip_onsets_within_25_of_previous} of {sh.grip_onsets} start within 25 s of the previous one), so a fair rule knows the cycle phase and the time since the last stop and the last grip loss.</p>
        <p><b>Physics residuals.</b> Each joint has an expected-current model (inertia, friction, gravity, coupling) refitted inside each fold; the network sees measured minus expected current. Held-out R&sup2; per joint, physics vs linear: {d.l.load_model_r2_heldout.physics.map((v, i) => `${JOINTS[i].id} ${f(v, 2)}/${f(d.l.load_model_r2_heldout.linear[i], 2)}`).join(", ")}.</p>
        <table className="tbl tbl-num">
          <thead><tr><th>Session</th><th>Model</th><th>Phase + recency</th><th>Cycle phase</th></tr></thead>
          <tbody>{Object.entries(gl.ap_by_session).map(([k2, v]) => (
            <tr key={k2}><td>{+k2 + 1}</td><td>{f(v.cnn, 2)}</td><td>{f(v.phase_recency, 2)}</td><td>{f(v.phase_baseline, 2)}</td></tr>
          ))}</tbody>
        </table>
        <p className="small">One robot and {sh.grip_onsets} grip losses: a lead to test on a live cell, not a validated alarm.</p>
      </>}>
      <ModelSchematic window={d.l.window_samples} channels={d.l.channels} />
      <div className="warn-ctl">
        <div className="seg" role="radiogroup" aria-label="Target">
          <button type="button" role="radio" aria-checked={!isStop} className={!isStop ? "is-on" : ""} onClick={() => setMode("grip")}>Grip loss</button>
          <button type="button" role="radio" aria-checked={isStop} className={isStop ? "is-on" : ""} onClick={() => setMode("stop")}>Protective stop</button>
        </div>
        {!isStop && (
          <label className="sel">
            <span>Check</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              {opts.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
            </select>
          </label>
        )}
      </div>
      <ApChart rows={view.rows} chance={view.chance} max={view.max} chanceLabel="guessing" />
      <p className={`verdict${isStop ? " is-no" : " is-yes"}`} aria-live="polite">{view.verdict}</p>
    </Step>
  );
}

/* ---------------------------------------------------------------- step 5 */

function FixStep({ d }: { d: Data }) {
  const fx = d.l.j2_fix;
  const r = d.s.reliability;
  const [after, setAfter] = useState(false);
  const rows: [string, string, string][] = [
    ["Protective stops", f(r.n_stop_onsets, 0), f(r.n_stop_onsets - fx.stops_removed, 0)],
    ["Stops per up-hour", f(fx.stops_per_h_before, 0), f(fx.stops_per_h_after, 0)],
    ["MTBF", `${f(fx.mtbf_before_s, 1)} s`, `${f(fx.mtbf_after_s, 1)} s`],
    ["Availability", pct(fx.availability_before, 1), pct(fx.availability_after, 1)],
  ];
  return (
    <Step n={5} id="fix" title="What fixing that move is worth"
      panel="ESTIMATE · SAMPLES 2–5 AT CYCLE-AVERAGE STOP RATE"
      lead={<>If samples 2–5 stopped no more often than the rest of the cycle, {f(fx.stops_removed, 0)} of {f(r.n_stop_onsets, 0)} stops would go. That is an upper-end target for a trial, not a forecast.</>}
      details={<>
        <p>Stops in the window after the fix = cycle-average rate ({f(fx.avg_rate_per_100s, 2)} per 100 s) &times; window up-time = {f(fx.stops_in_window_after, 1)}; removed = {f(fx.stops_in_window, 0)} &minus; {f(fx.stops_in_window_after, 1)} = {f(fx.stops_removed, 1)} ({pct(fx.share_removed, 0)} of stops, interval {pct(fx.share_removed_lo, 0)} to {pct(fx.share_removed_hi, 0)}).</p>
        <p>The window was picked as the hottest one on this same data, so the figure leans high, and the interval does not cover that choice.</p>
      </>}>
      <div className="fix">
        <div className="fix-vis">
          <div className="seg" role="radiogroup" aria-label="Show">
            <button type="button" role="radio" aria-checked={!after} className={!after ? "is-on" : ""} onClick={() => setAfter(false)}>As recorded</button>
            <button type="button" role="radio" aria-checked={after} className={after ? "is-on" : ""} onClick={() => setAfter(true)}>J2 swing fixed</button>
          </div>
          <FixGrid total={r.n_stop_onsets} inWin={fx.stops_in_window} removed={fx.stops_removed} after={after} />
          <div className="fix-key"><span><i className="sw sw-stop" />Stop in samples 2–5</span><span><i className="sw sw-out" />Stop elsewhere</span>{after && <span><i className="sw sw-gone" />Removed by the fix</span>}</div>
        </div>
        <table className="fix-tbl">
          <thead><tr><th /><th>Recorded</th><th>Fixed</th></tr></thead>
          <tbody>{rows.map(([a, b, c]) => (
            <tr key={a}><th>{a}</th><td className={!after ? "is-on" : ""}>{b}</td><td className={after ? "is-on" : ""}>{c}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </Step>
  );
}

/* ---------------------------------------------------------------- step 6 details */

function ProofDetails({ d }: { d: Data }) {
  const s = d.s;
  const w = s.reliability_stats.weibull, ws = s.reliability_stats.weibull_stop_only;
  const so = d.l.tasks.stop_10;
  return (
    <>
      <p><b>Per session.</b> The 2–5 window was picked from the pooled data, so it is re-checked per session. Session 3 holds most of the stops, so the pooled figure leans on it; session 1 has only {s.phase.hot_by_segment_2_5["0"].n_in + s.phase.hot_by_segment_2_5["0"].n_out} stops and is inconclusive.</p>
      <p><b>Reliability.</b> Gaps between down episodes fit a Weibull with shape {f(w.shape, 2)} {ci(w.shape_lo, w.shape_hi, 2)}; stops alone {f(ws.shape, 2)} {ci(ws.shape_lo, ws.shape_hi, 2)}. The gaps are lumpier than exponential (KS p = {f(w.ks_exp_p, 2)}). Two hours cannot rule out wear over weeks; the data does show that stops track cycle position.</p>
      <p><b>Models.</b> Folds are contiguous blocks of time with a 20-sample purge either side. An earlier gradient-boosted stop model looked better only when folds split single cycles; with cycles kept together its edge was gone. The stop result here: model {f(so.ap.cnn.v, 3)}, phase + recency {f(so.ap.phase_recency.v, 3)}.</p>
      <p><b>Tests.</b> 34 pytest tests cover the hand calculations, known statistical values, leakage and time order. Equations and assumptions are in METHOD.md in the repository.</p>
    </>
  );
}
