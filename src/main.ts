type Row = { q: number; caught: number; precision: number; false_alarms_per_h: number; median_lead_s: number | null };
interface Summary {
  reliability: Record<string, number>;
  warning: { horizon_s: number; base_rate: number; summaries: Record<string, { ap: number }>; sweeps: Record<string, Row[]> };
  throughput: { case_rate: number; shift_h: number; target_availability: number; budget_min: number;
    rows: { mttr_s: number; cases_per_stop: number; max_stops_in_budget: number }[] };
  phase: { hot: Record<string, { share: number; rate100_in: number; rate100_out: number; rr: number; rr_lo: number; rr_hi: number; up_share: number }> };
}
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const f = (x: number, d = 1) => x.toFixed(d);

async function main() {
  const el = document.getElementById("app")!;
  const s: Summary = await (await fetch("/data/summary.json")).json();
  const r = s.reliability;
  const w = s.warning;
  const at = (k: string) => w.sweeps[k].find((x) => x.q === 0.95)!;
  const tp = s.throughput;
  const h = s.phase.hot.stop_2_5;
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
  ${[["model", "Gradient-boosted model"], ["phase_baseline", "Cycle-phase baseline"], ["tool_threshold", "Tool-current threshold"], ["random", "Chance"]]
    .map(([k, n]) => `<tr><td>${n}</td><td>${f(w.summaries[k].ap, 3)}</td><td>${pct(at(k).precision)}</td><td>${pct(at(k).caught)}</td><td>${f(at(k).false_alarms_per_h)}</td></tr>`)
    .join("")}
  </tbody></table>
  <p>Base rate of positive rows: ${pct(w.base_rate)}.</p>
  <h2>3. Where in the cycle stops start</h2>
  <p>${pct(h.share)} of protective-stop onsets fall in cycle samples 2 to 5 (${pct(h.up_share)} of up-time):
  ${f(h.rate100_in)} vs ${f(h.rate100_out)} onsets per 100 up-seconds, rate ratio ${f(h.rr)} (95% CI ${f(h.rr_lo)} to ${f(h.rr_hi)}).</p>
  <h2>4. Cost per stop against a ${pct(tp.target_availability)} uptime budget</h2>
  <p>${tp.shift_h} h shift, ${tp.case_rate} cases/min (inputs). Budget: ${f(tp.budget_min, 2)} min of downtime per shift.</p>
  <table><thead><tr><th>Site recovery time per stop</th><th>Cases lost per stop</th><th>Most stops per shift inside budget</th></tr></thead><tbody>
  ${tp.rows.map((x) => `<tr><td>${x.mttr_s} s</td><td>${f(x.cases_per_stop)}</td><td>${f(x.max_stops_in_budget, 2)}</td></tr>`).join("")}
  </tbody></table>`;
}
main().catch((e) => { document.getElementById("app")!.textContent = `Failed to load results: ${e}`; });
