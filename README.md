# Cobot Cell Stop Lens

Finds where in the robot cycle protective stops and grip losses start, using real UR3 cobot telemetry, tests whether they can be predicted a few seconds ahead against honest baselines, and prices each stop against a contracted uptime budget.

## The problem
At the end of a packaging line, the palletizing cell sets the pace for everything upstream. Each protective stop or dropped case costs output, and finding the move that causes them is usually manual root-cause work.

## Key results (all measured on the real data; 95 % intervals in brackets)
1. **Where stops start.** 71 % of protective-stop onsets (77 of 108) start in cycle samples 2-5 (about 1 s per sample), one to three samples after the cycle's fastest J2 swing, which follows the tool-current peak (likely the gripper acting). That window holds 12 % of up-time: stop rate ratio 18.1 [11.8, 28.4] vs the rest of the cycle. The window was picked from the pooled data, so it is re-checked per recording session: segment 1 (14 stops) RR 11.4 [3.4, 43.1], segment 2 (87 stops) RR 24.2 [14.7, 41.2]; segment 0 has only 7 stops and is inconclusive (RR 4.5 [0.7, 26.3]). Segment 2 holds most of the stops, so the pooled figure leans on that session. This locates the stops in one move of the cycle; it does not by itself say what in that move causes them. Grip losses cluster elsewhere, in the transfer and place portion (samples 15-30, RR 2.9 [1.8, 4.8]), which points to a different fix.
2. **No sign of wear-out inside the log.** Gaps between down episodes (stop or grip loss) fit a Weibull with shape 1.02 [0.92, 1.19]; protective stops alone give 0.97 [0.86, 1.15]. There is no rising hazard within these two hours, but the pooled gaps are not strictly memoryless either (KS vs exponential p = 0.04; they are lumpier than exponential). Two hours cannot rule out wear over weeks; what the data does show is that stops track cycle position. MTBF between down episodes 37.8 s [32.7, 43.9], availability 92.9 % [91.6, 94.1] on this test rig, checked against a hand calculation.
3. **Early warning: not yet better than knowing the cycle position.** With folds split by single cycles, a gradient-boosted model on per-joint load residuals (physics-informed current models: inertia, Coulomb friction, temperature, coupling) reaches PR-AUC 0.438 [0.391, 0.486] against 0.327 for a cycle-phase baseline and 0.236 for chance. That split is optimistic: 56 % of the 10 s pre-onset windows span two cycles, so neighbouring cycles in different folds share onsets and rolling features. With folds made of 5 consecutive cycles the edge shrinks to 0.343 vs 0.332 (difference [-0.036, +0.058]), and on a time-ordered split (train on the two earlier sessions, test on the later one) it is 0.323 vs 0.345. Tool current and temperature alone are near chance. The cycle-window finding is the robust result; the residual early warning is a lead that needs more data.
4. **Cost per stop.** Against a 99.6 % uptime budget (1.92 min per 8 h shift), a 2-minute manual reset costs 24 cases at 12 cases/min, and the budget fits fewer than one such stop per shift. A sensitivity table covers 2.9 s to 5 min recovery times. Case rate, shift and target are assumptions, not measurements.
5. **Week-1 test plan.** Tag stops by program waypoint, rank by exact Poisson rate, change one variable on the hottest move; detecting a 50 % drop at 80 % power needs 49 stops in the baseline arm (hours per arm tabulated by stop rate).

The current page is a plain results view; the visual walkthrough is in progress.

## How it works
- `pipeline/reliability.py`: cleaning, segmentation at recording gaps, episode detection, MTBF/MTTR/availability, chi-square MTBF interval, cycle-bootstrap availability interval, Weibull inter-arrival fit with KS test.
- `pipeline/phase.py`: cycle-phase hazard with exact Poisson intervals, rate-ratio tests, per-sample joint motion profile, downtime Pareto, rate-ratio power calculation.
- `pipeline/warning.py`: Savitzky-Golay acceleration, per-joint ridge load models, rolling features, GroupKFold gradient boosting (per-cycle and blocked folds), cycle-phase baseline, feature-group ablation, time-ordered holdout, cycle-block bootstrap.
- `pipeline/throughput.py`: cost per stop and stops allowed by an availability budget.
- `pipeline/run.py`: writes static JSON to `public/data/`; the front end (Vite + TypeScript) reads it. No API keys.
- `tests/`: 20 pytest tests: hand calculations, known statistical values, leakage and time-order checks. Equations, assumptions and limits: [METHOD.md](METHOD.md).

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
