# Cobot Cell Stop Lens

**The question.** When a robot cell keeps throwing protective stops and dropping parts, which move in the cycle is causing it, what does each stop cost against a contracted uptime target, and what single change should be tested first?

**The answer, measured on real UR3 cobot telemetry.** 71 % of protective stops (77 of 108) started in one 4-second window right after the fastest J2 swing, 18x the rate of the rest of the cycle. Fixing that move would remove up to 59 % of stops. Dropped parts came from a different move (transfer and place), and a small AI model on physics-residual joint currents warned of them 3-10 s ahead at 0.55 PR-AUC, vs 0.34 for the best simple rule. For stops the model does not beat knowing the cycle position, so stops get a fix, not an alarm. At 99.6 % uptime, one 2-minute manual reset uses the whole shift's downtime budget (case rate, shift length and the target are assumptions). One robot, 2.08 h: caveats in result 1 below.

**The solution.** A one-page playbook and a small tool, `pipeline/stop_map.py`: one cell log in, a ranked fix list, its cost against the uptime budget and the trial size out. See **[SOLUTION.md](SOLUTION.md)**.

```bash
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/stop_map.py downloads/dataset_02052023.xlsx --window 2-5 --grip-window 15-30
```

## The problem
At the end of a packaging line, the palletizing cell sets the pace for everything upstream. Each protective stop or dropped case costs output, and finding the move that causes them is usually manual root-cause work.

## Key results (all measured on the real data; 95 % intervals in brackets)
1. **Where stops start.** 71 % of protective-stop onsets (77 of 108) start in cycle samples 2-5 (about 1 s per sample), one to three samples after the cycle's fastest J2 swing, which follows the tool-current peak (likely the gripper acting). That window holds 12 % of up-time: stop rate ratio 18.1 [11.8, 28.4] vs the rest of the cycle. The window was picked from the pooled data, so it is re-checked per recording session: segment 1 (14 stops) RR 11.4 [3.4, 43.1], segment 2 (87 stops) RR 24.2 [14.7, 41.2]; segment 0 has only 7 stops and is inconclusive (RR 4.5 [0.7, 26.3]). Segment 2 holds most of the stops, so the pooled figure leans on that session. This locates the stops in one move of the cycle; it does not by itself say what in that move causes them. Grip losses cluster elsewhere, in the transfer and place portion (samples 15-30, RR 2.9 [1.8, 4.8]), which points to a different fix.
2. **No sign of wear-out inside the log.** Gaps between down episodes (stop or grip loss) fit a Weibull with shape 1.02 [0.92, 1.19]; protective stops alone give 0.97 [0.86, 1.15]. There is no rising hazard within these two hours, but the pooled gaps are not strictly memoryless either (KS vs exponential p = 0.04; they are lumpier than exponential). Two hours cannot rule out wear over weeks; what the data does show is that stops track cycle position. MTBF between down episodes 37.8 s [32.7, 43.9], availability 92.9 % [91.6, 94.1] on this test rig, checked against a hand calculation.
3. **Where learning wins: grip-loss early warning.** A small 1D-CNN reads the last 12 s of 25 channels (joint currents, physics-model current residuals, joint speeds, temperature changes, tool current) plus cycle phase. The residuals come from an inverse-dynamics-style expected-current model per joint (inertia, friction, gravity from integrated joint angles, coupling), refitted inside each fold; it beats the earlier linear load model on held-out blocks (mean R^2 0.31 vs 0.28). On 5 contiguous time blocks with a 20-sample purge, predicting a grip loss 3 to 10 s ahead, the CNN reaches PR-AUC **0.549 [0.466, 0.628]** (chance 0.081). Grip losses come in runs (33 of 77 start within 25 s of the previous one), so the fair baseline knows the cycle phase and the time since the last stop *and* since the last grip loss: it scores **0.343 [0.267, 0.417]**, and the CNN's edge over it is **+0.21 [+0.13, +0.29]**. (A baseline that only knows the time since the last down episode of any kind scores 0.129; that gap, +0.42, overstates the gain.) On rows with no grip loss in the previous 25 s the CNN still scores 0.32 [0.19, 0.45] against 0.09 for that baseline (base rate 0.05). The result holds with the gripper-current channel removed (0.574) and with strictly causal features (0.567; the shipped acceleration filter looks 3 samples ahead). Per session against the weaker baseline: 0.61 / 0.66 / 0.50 vs 0.19 / 0.24 / 0.11. One robot, 77 grip losses: a lead to test, not a validated alarm.
4. **Protective stops: the cycle position is most of the story.** For protective stops 10 s ahead the CNN beats the cycle-phase baseline (0.272 vs 0.208, difference [+0.03, +0.11]) but not the phase + recency baseline (0.246, difference [-0.03, +0.09]). So the stop result is the cycle window in result 1, and the fix in result 5, not an alarm. An earlier gradient-boosted model looked better only when folds split single cycles (0.438 vs 0.327); with consecutive cycles kept together the edge was gone (0.343 vs 0.332).
5. **What fixing the J2 swing would buy (upper-end estimate).** If samples 2-5 ran at the cycle-average stop rate, 64 of 108 protective stops would go (59 % [50 %, 69 %]): 56 to 23 stops per up-hour, MTBF 37.5 to 57.5 s, rig availability 92.9 % to 95.1 %. The window was picked as the hottest one on this same data, so this in-sample figure leans high, and the interval does not cover that choice; it is a target for a trial, not a forecast.
6. **Cost per stop.** Against a 99.6 % uptime budget (1.92 min per 8 h shift), a 2-minute manual reset costs 24 cases at 12 cases/min, and the budget fits fewer than one such stop per shift. A sensitivity table covers 2.9 s to 5 min recovery times. Case rate, shift and target are assumptions, not measurements.
7. **Week-1 test plan.** Tag stops by program waypoint, rank by exact Poisson rate, change one variable on the hottest move; detecting a 50 % drop at 80 % power needs 49 stops in the baseline arm (hours per arm tabulated by stop rate).

The web page walks through these results in seven steps: the run log, cost per stop, the cycle timeline with a joint-by-joint arm view, the grip-loss model against its baselines, the J2-swing fix estimate, the checks, and the stop-map playbook.

## How it works
- `pipeline/reliability.py`: cleaning, segmentation at recording gaps, episode detection, MTBF/MTTR/availability, chi-square MTBF interval, cycle-bootstrap availability interval, Weibull inter-arrival fit with KS test.
- `pipeline/phase.py`: cycle-phase hazard with exact Poisson intervals, rate-ratio tests, per-sample joint motion profile, downtime Pareto, rate-ratio power calculation.
- `pipeline/warning.py`: Savitzky-Golay acceleration, per-joint ridge load models, rolling features, GroupKFold gradient boosting (per-cycle and blocked folds), cycle-phase baseline, feature-group ablation, time-ordered holdout, cycle-block bootstrap.
- `pipeline/learned.py`, `pipeline/run_learned.py`: physics residual model on integrated joint angles, 1D-CNN (PyTorch, CPU), contiguous time-blocked folds with purge, lead-time labels, phase + recency baseline, channel ablation, block bootstrap, J2-swing fix arithmetic; writes `public/data/learned.json`.
- `pipeline/check_leakage.py`: re-scores the grip-loss and stop targets with strictly causal features and a grip-aware recency baseline; writes `public/data/leakage_checks.json`.
- `pipeline/throughput.py`: cost per stop and stops allowed by an availability budget.
- `pipeline/stop_map.py`: the stop-map tool, a thin CLI over the modules above (see SOLUTION.md).
- `pipeline/run.py`: writes static JSON to `public/data/`; the front end (Vite + TypeScript) reads it. No API keys.
- `tests/`: 36 pytest tests: hand calculations, known statistical values, leakage and time-order checks. Equations, assumptions and limits: [METHOD.md](METHOD.md).

## Run locally
```bash
npm install
npm run dev
```
Pipeline and tests: see METHOD.md, section 8.

## Data
UR3 CobotOps, Tyrovolas, Aliev, Antonelli, Stylios (2024), UCI Machine Learning Repository, doi:10.24432/C5J891, licensed CC BY 4.0. Real telemetry; no synthetic data. Case rate, shift length, site recovery time and the 99.6 % target in the cost step are inputs, not measurements.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
