type Row = { q: number; caught: number; precision: number; false_alarms_per_h: number; median_lead_s: number | null };
interface Summary {
  reliability: Record<string, number>;
  warning: { horizon_s: number; base_rate: number; summaries: Record<string, { ap: number }>; sweeps: Record<string, Row[]> };
  throughput: { default_rate: number; default_shift_h: number; at_q95: Record<string, number> };
}
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const f = (x: number, d = 1) => x.toFixed(d);

async function main() {
  const el = document.getElementById("app")!;
  const s: Summary = await (await fetch("/data/summary.json")).json();
  const r = s.reliability;
  const w = s.warning;
  const at = (k: string) => w.sweeps[k].find((x) => x.q === 0.95)!;
  const t = s.throughput.at_q95;
  el.innerHTML = `
  <h2>1. Reliability, measured</h2>
  <table><tbody>
  <tr><td>Stop or grip-loss episodes</td><td>${r.n_episodes} (${r.n_stop_onsets} stop onsets, ${r.n_grip_onsets} grip-loss onsets)</td></tr>
  <tr><td>MTBF</td><td>${f(r.mtbf_s)} s</td></tr>
  <tr><td>MTTR</td><td>${f(r.mttr_s, 2)} s</td></tr>
  <tr><td>Availability A = MTBF / (MTBF + MTTR)</td><td>${pct(r.availability)}</td></tr>
  </tbody></table>
  <h2>2. Early warning, ${w.horizon_s} s horizon, held-out cycles</h2>
  <table><thead><tr><th>Scorer</th><th>PR-AUC</th><th>Precision at top 5%</th><th>Onsets caught</th><th>False alarms / h</th></tr></thead><tbody>
  ${[["model", "Gradient-boosted model"], ["tool_threshold", "Tool-current threshold"], ["random", "Chance"]]
    .map(([k, n]) => `<tr><td>${n}</td><td>${f(w.summaries[k].ap, 3)}</td><td>${pct(at(k).precision)}</td><td>${pct(at(k).caught)}</td><td>${f(at(k).false_alarms_per_h)}</td></tr>`)
    .join("")}
  </tbody></table>
  <p>Base rate of positive rows: ${pct(w.base_rate)}.</p>
  <h2>3. Cases lost at an end-of-line palletizer</h2>
  <p>At ${s.throughput.default_rate} cases/min (input default) and an ${s.throughput.default_shift_h} h shift: ${f(t.down_min)} min down, <b>${Math.round(t.cases_lost)} cases lost per shift</b>.
  If every onset the model catches at the top-5% threshold were avoided (upper bound): ${Math.round(t.cases_lost_after)} cases.</p>`;
}
main().catch((e) => { document.getElementById("app")!.textContent = `Failed to load results: ${e}`; });
