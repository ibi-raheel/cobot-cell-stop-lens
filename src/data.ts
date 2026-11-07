// Types for the static JSON written by the pipeline. The front end only reads these files.

export interface CI { v: number; lo: number; hi: number }
export interface Rate { n: number; rate100: number; lo100: number; hi100: number }
export interface Hot {
  window: [number, number]; n_in: number; n_out: number; share: number; up_share: number;
  rate100_in: number; rate100_out: number; rr: number; rr_lo: number; rr_hi: number; p: number;
}
export interface ProfileRow { phase: number; n_up: number; speed: number[]; tool: number; stop: number; grip: number; any: number }
export interface Weibull { n: number; shape: number; shape_lo: number; shape_hi: number; scale_s: number; ks_exp_p: number }

export interface Summary {
  dataset: { rows: number; cols: number; segments: number; cycles: number; missing_cells: number; unknown_stop_labels: number };
  reliability: {
    observed_s: number; down_s: number; up_s: number; n_episodes: number; n_stop_onsets: number; n_grip_onsets: number;
    mtbf_s: number; mttr_s: number; availability: number;
  };
  reliability_stats: { weibull: Weibull; weibull_stop_only: Weibull; mtbf_lo: number; mtbf_hi: number; availability_ci: { point: number; lo: number; hi: number } };
  phase: {
    bins: { bin: string; up_s: number; share_up: number; stop: Rate; grip: Rate }[];
    hot: Record<"stop_2_5" | "grip_15_30" | "stop", Hot>;
    hot_by_segment_2_5: Record<string, Hot>;
    profile: ProfileRow[];
    pareto: { n_cycles: number; cycles_with_episode: number; top10pct_cycles: number; top10pct_share: number; cycles_for_half: number };
    power: { rr: number; power: number; events_baseline_arm: number; hours_per_arm: Record<string, number>; rig_stop_rate_per_h: number };
  };
  throughput: {
    case_rate: number; shift_h: number; target_availability: number; budget_min: number;
    rows: { mttr_s: number; cases_per_stop: number; max_stops_in_budget: number }[];
  };
}

type ApSet = Record<string, CI>;
export interface Task {
  base_rate: number; rows: number; positives: number; onsets: number;
  ap: ApSet; auc: ApSet; diff_ap: Record<string, CI & { p_le_0: number }>;
  ap_by_session: Record<string, Record<string, number>>;
}
export interface Learned {
  validation: string; window_samples: number; channels: number;
  load_model_r2_heldout: { physics: number[]; linear: number[] };
  tasks: Record<"grip_lead3" | "grip_10" | "stop_10" | "stop_lead3" | "any_10" | "any_30", Task>;
  cycle_rank: { cycles: number; positives: number; auc: ApSet; diff_auc: Record<string, { lo: number; hi: number }> };
  j2_fix: {
    avg_rate_per_100s: number; stops_in_window: number; stops_in_window_after: number; stops_removed: number;
    share_removed: number; share_removed_lo: number; share_removed_hi: number;
    availability_before: number; availability_after: number; mtbf_before_s: number; mtbf_after_s: number;
    mttr_stop_s: number; stops_per_h_before: number; stops_per_h_after: number;
  };
}
interface NoRecent { rows: number; positives: number; base_rate: number; ap: Record<string, number>; cnn_ci: [number, number]; diff_ci: [number, number] }
export interface GripCheck {
  base_rate: number; positives: number; ap: ApSet; diff_ap: Record<string, CI>;
  no_recent_grip_loss: Record<"12" | "25", NoRecent>;
  grip_onsets: number; grip_onsets_within_25_of_previous: number;
}
export interface Leakage {
  causal_grip_lead3: GripCheck; shipped_grip_lead3: GripCheck;
  causal_stop_10: { ap: ApSet; diff_ap: Record<string, CI> };
}
export interface Episode { start: number; end: number; cycle: number; dur: number; kind: "stop" | "grip" | "stop+grip" }
export interface Timeline { series: { t: number[]; segment: number[] }; episodes: Episode[] }

export interface Data { s: Summary; l: Learned; k: Leakage; tl: Timeline }

const url = (f: string) => `${import.meta.env.BASE_URL}data/${f}`;
const get = async <T,>(f: string): Promise<T> => {
  const r = await fetch(url(f));
  if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
};

export async function loadAll(): Promise<Data> {
  const [s, l, k, tl] = await Promise.all([
    get<Summary>("summary.json"), get<Learned>("learned.json"), get<Leakage>("leakage_checks.json"), get<Timeline>("timeline.json"),
  ]);
  return { s, l, k, tl };
}

// formatting
export const f = (x: number, d = 1) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const pct = (x: number, d = 1) => `${f(x * 100, d)}%`;
export const sgn = (x: number, d = 2) => `${x >= 0 ? "+" : "−"}${f(Math.abs(x), d)}`;
export const ci = (lo: number, hi: number, d = 1) => `[${f(lo, d)}, ${f(hi, d)}]`;

export const JOINTS = [
  { id: "J0", name: "Base" },
  { id: "J1", name: "Shoulder" },
  { id: "J2", name: "Elbow" },
  { id: "J3", name: "Wrist 1" },
  { id: "J4", name: "Wrist 2" },
  { id: "J5", name: "Wrist 3" },
];

/** Angle of each joint relative to cycle start, integrated from the mean signed speed per sample (dt about 1 s). */
export function integratePose(profile: ProfileRow[]): number[][] {
  const out: number[][] = [];
  let acc = [0, 0, 0, 0, 0, 0];
  for (const r of profile) {
    acc = acc.map((a, j) => a + r.speed[j]);
    out.push(acc);
  }
  return out;
}
