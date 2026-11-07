import { JOINTS, f } from "./data";
import { useReducedMotion, useTween } from "./hooks";

const deg = (r: number) => (r * 180) / Math.PI;
const dir = (a: number) => [Math.sin((a * Math.PI) / 180), -Math.cos((a * Math.PI) / 180)];
const L1 = 92, L2 = 82, L3 = 28;
const S: [number, number] = [176, 214];

/**
 * Side-view sketch of a six-axis arm. Joint angles are relative to cycle start (integrated from
 * the mean joint speed per sample); the fixed offsets only set a readable home pose.
 */
export function Arm({ pose, selected, onSelect, hot }: {
  pose: number[]; selected: number; onSelect?: (j: number) => void; hot?: boolean;
}) {
  const reduced = useReducedMotion();
  const q = useTween(pose, 520, reduced);
  const a1 = 26 + deg(q[1]);
  const a2 = a1 + 100 + deg(q[2]);
  const a3 = a2 + 62 + deg(q[3]);
  const [d1, d2, d3] = [dir(a1), dir(a2), dir(a3)];
  const E = [S[0] + L1 * d1[0], S[1] + L1 * d1[1]];
  const W = [E[0] + L2 * d2[0], E[1] + L2 * d2[1]];
  const T = [W[0] + L3 * d3[0], W[1] + L3 * d3[1]];
  const G = [T[0] + 14 * d3[0], T[1] + 14 * d3[1]];
  const n = [-d3[1], d3[0]]; // normal to the tool axis
  const spread = 7 + 2.5 * Math.sin(q[5] * 3);
  const base = [S[0], 250];
  const pts: { j: number; x: number; y: number; r: number }[] = [
    { j: 0, x: base[0], y: base[1], r: 11 },
    { j: 1, x: S[0], y: S[1], r: 13 },
    { j: 2, x: E[0], y: E[1], r: 12 },
    { j: 3, x: W[0], y: W[1], r: 10 },
    { j: 4, x: (W[0] + T[0]) / 2, y: (W[1] + T[1]) / 2, r: 8 },
    { j: 5, x: T[0], y: T[1], r: 8 },
  ];
  const link = (a: number[], b: number[], w: number) => (
    <g>
      <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="var(--ink)" strokeWidth={w + 3} strokeLinecap="round" />
      <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="var(--arm)" strokeWidth={w} strokeLinecap="round" />
    </g>
  );
  const b0 = deg(q[0]) * 2; // base dial, drawn x2 so small base turns stay visible
  return (
    <svg className="arm" viewBox="0 0 360 290" role="img"
      aria-label={`Arm sketch. Selected joint ${JOINTS[selected].id} ${JOINTS[selected].name}.`}>
      <defs>
        <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--rule-strong)" strokeWidth="1" />
        </pattern>
      </defs>
      {/* work envelope */}
      <path d={`M ${S[0] - 196} ${S[1]} A 196 196 0 0 1 ${S[0] + 196} ${S[1]}`} fill="none" stroke="var(--rule)" strokeDasharray="2 5" />
      {/* fixture and floor */}
      <rect x="10" y="266" width="340" height="12" fill="url(#hatch)" />
      <line x1="10" y1="266" x2="350" y2="266" stroke="var(--ink-2)" />
      <rect x="248" y="232" width="86" height="34" fill="var(--panel)" stroke="var(--ink-2)" />
      <text x="291" y="253" textAnchor="middle" className="arm-tag">fixture</text>
      {/* pedestal */}
      <path d={`M ${S[0] - 24} 266 L ${S[0] - 18} 236 L ${S[0] + 18} 236 L ${S[0] + 24} 266 Z`} fill="var(--arm)" stroke="var(--ink)" strokeWidth="1.5" />
      <line x1={S[0]} y1={236} x2={S[0]} y2={S[1]} stroke="var(--ink)" strokeWidth="17" />
      <line x1={S[0]} y1={236} x2={S[0]} y2={S[1]} stroke="var(--arm)" strokeWidth="14" />
      {/* base dial (J0) */}
      <g transform={`translate(${base[0]} ${base[1]})`}>
        <circle r="9" fill="var(--surface)" stroke="var(--ink)" strokeWidth="1.5" />
        <line x1="0" y1="0" x2={8 * Math.sin((b0 * Math.PI) / 180)} y2={-8 * Math.cos((b0 * Math.PI) / 180)} stroke="var(--ink)" strokeWidth="2" strokeLinecap="round" />
      </g>
      {link(S, E, 18)}
      {link(E, W, 15)}
      {link(W, T, 11)}
      {/* gripper */}
      <g stroke="var(--ink)" strokeWidth="3" strokeLinecap="round">
        <line x1={T[0] + n[0] * spread} y1={T[1] + n[1] * spread} x2={G[0] + n[0] * spread} y2={G[1] + n[1] * spread} />
        <line x1={T[0] - n[0] * spread} y1={T[1] - n[1] * spread} x2={G[0] - n[0] * spread} y2={G[1] - n[1] * spread} />
        <line x1={T[0] + n[0] * spread} y1={T[1] + n[1] * spread} x2={T[0] - n[0] * spread} y2={T[1] - n[1] * spread} />
      </g>
      {pts.map((p) => {
        const sel = p.j === selected;
        return (
          <g key={p.j} className={`jt${onSelect ? " jt-click" : ""}`} onClick={onSelect ? () => onSelect(p.j) : undefined}
            role={onSelect ? "button" : undefined} aria-label={onSelect ? `Select ${JOINTS[p.j].id} ${JOINTS[p.j].name}` : undefined}>
            <circle cx={p.x} cy={p.y} r={p.r + 9} fill="transparent" />
            {sel && <circle cx={p.x} cy={p.y} r={p.r + 5} fill="none" stroke={hot ? "var(--stop)" : "var(--select)"} strokeWidth="4" />}
            {p.j > 0 && <circle cx={p.x} cy={p.y} r={p.r} fill="var(--joint)" stroke="var(--ink)" strokeWidth="1.5" />}
            {p.j > 0 && <circle cx={p.x} cy={p.y} r={p.r * 0.38} fill="var(--arm)" />}
            {(p.j < 3 || sel) && <text x={p.x + (p.j === 0 ? -22 : p.r + 6)} y={p.y + (p.j === 0 ? 4 : -p.r - 2)} className={`arm-lbl${sel ? " is-sel" : ""}`}
              textAnchor={p.j === 0 ? "end" : "start"}>{JOINTS[p.j].id}</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** Teach-pendant joint readouts: mean speed at this sample and angle relative to cycle start. */
export function Readouts({ speed, pose, selected, onSelect }: {
  speed: number[]; pose: number[]; selected: number; onSelect?: (j: number) => void;
}) {
  const max = 1.3;
  return (
    <div className="readouts" role={onSelect ? "radiogroup" : undefined} aria-label="Joint readouts">
      <div className="ro-head"><span>Joint</span><span>Speed rad/s</span><span>&Delta;&theta;</span></div>
      {JOINTS.map((jn, j) => {
        const v = speed[j];
        const w = Math.min(1, Math.abs(v) / max) * 50;
        const sel = j === selected;
        const Tag = onSelect ? "button" : "div";
        return (
          <Tag key={jn.id} className={`ro${sel ? " is-sel" : ""}`} {...(onSelect ? { type: "button" as const, role: "radio", "aria-checked": sel, onClick: () => onSelect(j) } : {})}>
            <span className="ro-id"><i className="ro-sw" aria-hidden />{jn.id}<small>{jn.name}</small></span>
            <span className="ro-val">
              <b>{v >= 0 ? "+" : "−"}{f(Math.abs(v), 3)}</b>
              <span className="ro-bar" aria-hidden><span style={{ left: v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} /></span>
            </span>
            <span className="ro-ang">{pose[j] >= 0 ? "+" : "−"}{f(Math.abs(deg(pose[j])), 0)}&deg;</span>
          </Tag>
        );
      })}
      {onSelect && <p className="ro-hint">&Delta;&theta;: angle change since cycle start. Pick a joint to plot its speed below.</p>}
    </div>
  );
}
