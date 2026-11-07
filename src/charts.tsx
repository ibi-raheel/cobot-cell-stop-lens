import { useMemo, useRef, useState } from "react";
import type { Episode, Hot, ProfileRow, Summary, Timeline } from "./data";
import { JOINTS, f } from "./data";
import { useWidth } from "./hooks";

/* ------------------------------------------------------------------ cycle timeline strip */

export function CycleStrip({ profile, joint, sample, onSample, hot }: {
  profile: ProfileRow[]; joint: number; sample: number; onSample: (s: number) => void; hot: Hot;
}) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const svgRef = useRef<SVGSVGElement>(null);
  const n = profile.length;
  const ml = 34, mr = 10;
  const bw = (w - ml - mr) / n;
  const x = (i: number) => ml + (i + 0.5) * bw;
  const rows = { a: [44, 92], b: [184, 72], c: [292, 36] } as const; // [top, height]
  const H = 356;
  const maxStop = Math.max(...profile.map((p) => p.stop));
  const yA = (v: number) => rows.a[0] + rows.a[1] - (v / maxStop) * rows.a[1];
  const sMax = 1.3;
  const yB = (v: number) => rows.b[0] + rows.b[1] / 2 - (v / sMax) * (rows.b[1] / 2);
  const tMax = 0.3;
  const yC = (v: number) => rows.c[0] + rows.c[1] - (v / tMax) * rows.c[1];
  const colW = Math.min(18, bw * 0.56);
  const [lo, hi] = [2, 5];
  const bandX0 = x(lo) - bw / 2, bandX1 = x(hi) + bw / 2;

  const pick = (clientX: number) => {
    const r = svgRef.current!.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * w;
    onSample(Math.max(0, Math.min(n - 1, Math.round((px - ml) / bw - 0.5))));
  };
  const drag = useRef(false);
  const line = (ys: (v: number) => number, vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
  const p = profile[sample];

  return (
    <div className="strip" ref={ref}>
      <svg ref={svgRef} width={w} height={H} viewBox={`0 0 ${w} ${H}`} className="strip-svg"
        onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture?.(e.pointerId); pick(e.clientX); }}
        onPointerMove={(e) => { if (drag.current) pick(e.clientX); }}
        onPointerUp={() => { drag.current = false; }}
        onPointerCancel={() => { drag.current = false; }}
        role="img" aria-label={`Cycle timeline. Stop onsets, ${JOINTS[joint].id} speed and tool current for cycle samples 0 to ${n - 1}.`}>
        {/* hot window band */}
        <rect x={bandX0} y={rows.a[0] - 8} width={bandX1 - bandX0} height={rows.c[0] + rows.c[1] - rows.a[0] + 8} className="band" />
        

        {/* row A: stop onsets */}
        <text x={ml} y={rows.a[0] - 24} className="row-lbl">Protective-stop onsets per cycle sample</text>
        <line x1={ml} x2={w - mr} y1={rows.a[0] + rows.a[1]} y2={rows.a[0] + rows.a[1]} className="axis" />
        {[0, 10, 20, 30].filter((t) => t <= maxStop).map((t) => (
          <g key={t}><line x1={ml} x2={w - mr} y1={yA(t)} y2={yA(t)} className="grid" /><text x={ml - 6} y={yA(t) + 4} className="tick" textAnchor="end">{t}</text></g>
        ))}
        {profile.map((r, i) => {
          if (!r.stop) return null;
          const top = yA(r.stop), h = rows.a[0] + rows.a[1] - top;
          const rad = Math.min(3, h / 2);
          return (
            <g key={i}>
              <path d={`M${x(i) - colW / 2},${top + h} V${top + rad} q0,-${rad} ${rad},-${rad} H${x(i) + colW / 2 - rad} q${rad},0 ${rad},${rad} V${top + h} Z`} className="col-stop" />
              {r.stop >= 10 && <text x={x(i)} y={top - 5} className="val" textAnchor="middle">{r.stop}</text>}
            </g>
          );
        })}

        {/* row B: selected joint speed */}
        <text x={ml} y={rows.b[0] - 16} className="row-lbl">{JOINTS[joint].id} {JOINTS[joint].name.toLowerCase()}, mean speed (rad/s)</text>
        {[-1, 0, 1].map((t) => (
          <g key={t}><line x1={ml} x2={w - mr} y1={yB(t)} y2={yB(t)} className={t === 0 ? "axis" : "grid"} /><text x={ml - 6} y={yB(t) + 4} className="tick" textAnchor="end">{t > 0 ? "+1" : t < 0 ? "−1" : "0"}</text></g>
        ))}
        <path d={line(yB, profile.map((r) => r.speed[joint]))} className="trace" />
        {profile.map((r, i) => <circle key={i} cx={x(i)} cy={yB(r.speed[joint])} r={i === sample ? 5 : 3} className={i === sample ? "dot-on" : "dot"} />)}

        {/* row C: tool current */}
        <text x={ml} y={rows.c[0] - 10} className="row-lbl">Tool current (A)</text>
        <line x1={ml} x2={w - mr} y1={rows.c[0] + rows.c[1]} y2={rows.c[0] + rows.c[1]} className="axis" />
        <text x={ml - 6} y={yC(0.2) + 4} className="tick" textAnchor="end">0.2</text>
        <line x1={ml} x2={w - mr} y1={yC(0.2)} y2={yC(0.2)} className="grid" />
        <path d={line(yC, profile.map((r) => r.tool))} className="trace trace-2" />

        {/* x axis */}
        {profile.map((r, i) => (bw > 26 || i % 2 === 0 || i === sample) && (
          <text key={i} x={x(i)} y={H - 6} className={`tick${i === sample ? " tick-on" : ""}`} textAnchor="middle">{r.phase}</text>
        ))}

        {/* playhead */}
        <g className="playhead" style={{ transform: `translateX(${x(sample)}px)` }}>
          <line x1={0} x2={0} y1={rows.a[0] - 8} y2={H - 20} />
        </g>
      </svg>
      <div className="jog">
        <label htmlFor="jog" className="jog-lbl">Cycle sample <b>{String(p.phase).padStart(2, "0")}</b></label>
        <input id="jog" type="range" min={0} max={n - 1} step={1} value={sample} onChange={(e) => onSample(+e.target.value)}
          aria-valuetext={`Sample ${p.phase}: ${p.stop} stop onsets, ${JOINTS[joint].id} speed ${f(p.speed[joint], 2)} rad/s`} />
        <span className="jog-read"><b>{p.stop}</b> stop onset{p.stop === 1 ? "" : "s"} &middot; <b>{f(p.tool, 2)}</b> A tool</span>
      </div>
      <p className="note"><i className="sw sw-band" aria-hidden />Shaded: samples {lo}–{hi}, where {hot.n_in} of the {hot.n_in + hot.n_out} stop onsets fall. Samples are about 1 s apart; the first 16 of the cycle are shown. Drag the chart or the slider to scrub.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ run log (whole recording) */

export function RunLog({ tl }: { tl: Timeline }) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const [hover, setHover] = useState<Episode | null>(null);
  const segs = useMemo(() => {
    const s = tl.series.segment;
    const out: { id: number; i0: number; i1: number; dur: number }[] = [];
    let i0 = 0;
    for (let i = 1; i <= s.length; i++) {
      if (i === s.length || s[i] !== s[i - 1]) {
        out.push({ id: s[i0], i0, i1: i - 1, dur: tl.series.t[i - 1] - tl.series.t[i0] });
        i0 = i;
      }
    }
    return out;
  }, [tl]);
  const gap = 8;
  const total = tl.series.segment.length;
  const usable = w - gap * (segs.length - 1);
  const segX = (k: number) => segs.slice(0, k).reduce((a, s) => a + ((s.i1 - s.i0 + 1) / total) * usable + gap, 0);
  const xOf = (i: number) => {
    const k = segs.findIndex((s) => i >= s.i0 && i <= s.i1);
    const s = segs[k];
    return segX(k) + ((i - s.i0) / (s.i1 - s.i0 + 1)) * ((s.i1 - s.i0 + 1) / total) * usable;
  };
  const H = 92;
  const lanes = { stop: 12, grip: 42 };
  const near = (px: number) => {
    let best: Episode | null = null, bd = 1e9;
    for (const e of tl.episodes) { const d = Math.abs(xOf(e.start) - px); if (d < bd) { bd = d; best = e; } }
    setHover(bd < 14 ? best : null);
  };
  const segOf = (e: Episode) => segs.find((s) => e.start >= s.i0 && e.start <= s.i1)!;
  return (
    <div className="runlog" ref={ref}>
      <div className="runlog-key">
        <span><i className="sw sw-stop" />Protective stop</span>
        <span><i className="sw sw-grip" />Grip loss</span>
      </div>
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} onPointerMove={(e) => near(e.clientX - e.currentTarget.getBoundingClientRect().left)} onPointerDown={(e) => near(e.clientX - e.currentTarget.getBoundingClientRect().left)} onPointerLeave={() => setHover(null)}
        role="img" aria-label={`Run log: ${tl.episodes.length} down episodes over ${segs.length} recording sessions.`}>
        {segs.map((s, k) => (
          <rect key={s.id} x={segX(k)} y={4} width={((s.i1 - s.i0 + 1) / total) * usable} height={H - 26} className="sess" />
        ))}
        {tl.episodes.map((e, i) => {
          const isStop = e.kind !== "grip";
          const xx = xOf(e.start);
          return <rect key={i} x={xx - 0.75} y={isStop ? lanes.stop : lanes.grip} width={Math.max(1.5, xOf(e.end) - xx)} height={24}
            className={`${isStop ? "ep-stop" : "ep-grip"}${hover === e ? " ep-on" : ""}`} />;
        })}
        {segs.map((s, k) => (
          <text key={s.id} x={segX(k)} y={H - 4} className="tick">S{s.id + 1}{((s.i1 - s.i0 + 1) / total) * usable > 104 ? <> &middot; {f(s.dur / 60, 0)} min</> : null}</text>
        ))}
      </svg>
      <p className="runlog-read" aria-live="polite">
        {hover
          ? <>Session {segOf(hover).id + 1}, cycle {hover.cycle}: {hover.kind === "grip" ? "grip loss" : hover.kind === "stop" ? "protective stop" : "stop and grip loss"}, down {f(hover.dur, 1)} s</>
          : <>Each mark is one down episode. Point at or tap a mark to read it.</>}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ stops that fit in an uptime budget */

export function CostLadder({ tp }: { tp: Summary["throughput"] }) {
  const label = (s: number) => (s < 60 ? `${f(s, s % 1 ? 1 : 0)} s` : `${f(s / 60, 0)} min`);
  const note: Record<number, string> = { 2.9: "this rig's auto-resume", 120: "manual reset" };
  return (
    <div className="ladder" role="table" aria-label="Stops that fit in the shift downtime budget, by recovery time">
      <div className="ld-row ld-head" role="row">
        <span role="columnheader">Recovery per stop</span>
        <span role="columnheader">Stops that fit in {f(tp.budget_min, 2)} min of downtime per shift (one square = one stop)</span>
        <span role="columnheader" className="num">Cases lost per stop</span>
      </div>
      {tp.rows.map((r) => {
        const full = Math.floor(r.max_stops_in_budget);
        const part = r.max_stops_in_budget - full;
        const hl = r.mttr_s === 120;
        return (
          <div key={r.mttr_s} className={`ld-row${hl ? " is-hl" : ""}`} role="row">
            <span role="cell" className="ld-lbl">{label(r.mttr_s)}{note[r.mttr_s] && <small>{note[r.mttr_s]}</small>}</span>
            <span role="cell" className="ld-sq" aria-label={`${f(r.max_stops_in_budget, 2)} stops`}>
              {Array.from({ length: full }, (_, i) => <i key={i} />)}
              {part > 0.005 && <i className="part"><b style={{ width: `${part * 100}%` }} /></i>}
              <em>{f(r.max_stops_in_budget, r.max_stops_in_budget < 10 ? 2 : 1)}</em>
            </span>
            <span role="cell" className="num ld-cases">{f(r.cases_per_stop, r.cases_per_stop < 1 ? 1 : 0)}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ PR-AUC comparison */

export interface ApRow { key: string; label: string; v: number; lo?: number; hi?: number; model?: boolean }

export function ApChart({ rows, chance, max = 0.7, chanceLabel = "chance" }: { rows: ApRow[]; chance: number; max?: number; chanceLabel?: string }) {
  const X = (v: number) => `${(v / max) * 100}%`;
  const ticks = [0, 0.2, 0.4, 0.6].filter((t) => t <= max);
  return (
    <div className="ap" role="table" aria-label="PR-AUC by scorer">
      {rows.map((r) => (
        <div key={r.key} className={`ap-row${r.model ? " is-model" : ""}`} role="row">
          <span className="ap-lbl" role="rowheader">{r.label}</span>
          <span className="ap-track" aria-hidden>
            {ticks.map((t) => <span key={t} className="ap-grid" style={{ left: X(t) }} />)}
            <span className="ap-bar" style={{ transform: `scaleX(${r.v / max})` }} />
            {r.lo !== undefined && r.hi !== undefined && (
              <span className="ap-ci" style={{ left: X(r.lo), width: `${((r.hi - r.lo) / max) * 100}%` }} />
            )}
            <span className="ap-chance" style={{ left: X(chance) }} />
          </span>
          <span className="ap-val" role="cell">
            {f(r.v, 3)}{r.lo !== undefined && <small> [{f(r.lo, 2)}, {f(r.hi!, 2)}]</small>}
          </span>
        </div>
      ))}
      <div className="ap-row ap-axis" aria-hidden>
        <span />
        <span className="ap-track">
          {ticks.map((t) => <span key={t} className="ap-tick" style={{ left: X(t) }}>{f(t, 1)}</span>)}
        </span>
        <span className="ap-chance-lbl"><i />{chanceLabel} {f(chance, 3)}</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ model input schematic */

export function ModelSchematic({ window: win, channels }: { window: number; channels: number }) {
  const groups = [
    { n: 6, label: "Joint currents" },
    { n: 6, label: "Physics residuals" },
    { n: 6, label: "Joint speeds" },
    { n: 6, label: "Temperature change" },
    { n: channels - 24, label: "Tool current" },
  ];
  return (
    <figure className="schem" aria-label={`Model input: last ${win} seconds of ${channels} channels plus cycle phase, into a small 1D convolutional network.`}>
      <div className="schem-in">
        <div className="schem-groups">
          {groups.map((g) => (
            <div key={g.label} className="schem-g">
              <span className="schem-gl">{g.label}<small>&times;{g.n}</small></span>
              <span className="schem-cells" style={{ gridTemplateColumns: `repeat(${win}, 1fr)` }}>
                {Array.from({ length: g.n * win }, (_, i) => <i key={i} />)}
              </span>
            </div>
          ))}
        </div>
        <div className="schem-time"><span>&minus;{win} s</span><span>now</span></div>
        <div className="schem-phase">+ cycle phase</div>
      </div>
      <svg className="schem-arrow" viewBox="0 0 40 16" aria-hidden><path d="M2 8 H34 M28 3 L35 8 L28 13" /></svg>
      <div className="schem-net">
        <b>1D-CNN</b>
        <span>2 conv layers, ~8k weights</span>
        <span>trained on time-blocked folds</span>
      </div>
      <svg className="schem-arrow" viewBox="0 0 40 16" aria-hidden><path d="M2 8 H34 M28 3 L35 8 L28 13" /></svg>
      <div className="schem-out">
        <span className="lamp" aria-hidden />
        <span>Grip-loss warning<small>3–10 s ahead</small></span>
      </div>
    </figure>
  );
}

/* ------------------------------------------------------------------ 108 stops, before and after the fix */

export function FixGrid({ total, inWin, removed, after }: { total: number; inWin: number; removed: number; after: boolean }) {
  const out = total - inWin;
  const rem = Math.round(removed);
  return (
    <div className="fixgrid" role="img" aria-label={after
      ? `${total} stops; after the fix about ${rem} of the ${inWin} stops in samples 2 to 5 are gone.`
      : `${total} stops: ${inWin} in samples 2 to 5, ${out} elsewhere in the cycle.`}>
      {Array.from({ length: total }, (_, i) => {
        const isIn = i >= out;
        const gone = after && isIn && i >= total - rem;
        return <i key={i} className={`${isIn ? "fx-in" : "fx-out"}${gone ? " fx-gone" : ""}`} style={{ transitionDelay: gone || !after ? `${(i - (total - rem)) * 6}ms` : "0ms" }} />;
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ rate ratio forest plot (log scale) */

export function RateRatioPlot({ pooled, bySeg }: { pooled: Hot; bySeg: Record<string, Hot> }) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const rows = [
    ...Object.entries(bySeg).map(([k, h]) => ({ key: k, label: `Session ${+k + 1}`, sub: `${h.n_in + h.n_out} stops`, h })),
    { key: "all", label: "All sessions", sub: `${pooled.n_in + pooled.n_out} stops`, h: pooled },
  ];
  const narrow = w < 520;
  const ml = narrow ? 92 : 150, mr = 16, rowH = 38, top = 10;
  const H = top + rows.length * rowH + 30;
  const lo = 0.5, hi = 50;
  const X = (v: number) => ml + (Math.log(v / lo) / Math.log(hi / lo)) * (w - ml - mr);
  const ticks = narrow ? [1, 2, 5, 10, 20, 50] : [0.5, 1, 2, 5, 10, 20, 50];
  return (
    <div className="rr" ref={ref}>
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} role="img"
        aria-label={rows.map((r) => `${r.label}: rate ratio ${f(r.h.rr, 1)}, 95% interval ${f(r.h.rr_lo, 1)} to ${f(r.h.rr_hi, 1)}`).join("; ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={top - 4} y2={H - 26} className={t === 1 ? "axis" : "grid"} />
            <text x={X(t)} y={H - 10} className="tick" textAnchor="middle">{t}&times;</text>
          </g>
        ))}
        <text x={X(1) + 5} y={top + 4} className="tick tick-l">no difference</text>
        {rows.map((r, i) => {
          const y = top + i * rowH + rowH / 2 + 4;
          const pooledRow = r.key === "all";
          const weak = r.h.rr_lo < 1;
          return (
            <g key={r.key} className={pooledRow ? "rr-pooled" : ""}>
              {pooledRow && <line x1={0} x2={w} y1={y - rowH / 2 - 1} y2={y - rowH / 2 - 1} className="rule" />}
              <text x={0} y={y - 2} className="rr-lbl">{r.label}</text>
              <text x={0} y={y + 12} className="tick tick-l">{r.sub}</text>
              <line x1={X(Math.max(lo, r.h.rr_lo))} x2={X(Math.min(hi, r.h.rr_hi))} y1={y} y2={y} className={`rr-ci${weak ? " is-weak" : ""}`} />
              <rect x={X(r.h.rr) - 5} y={y - 5} width={10} height={10} className={`rr-pt${weak ? " is-weak" : ""}`} />
              <text x={X(r.h.rr)} y={y - 10} className="val" textAnchor="middle">{f(r.h.rr, 1)}&times;</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
