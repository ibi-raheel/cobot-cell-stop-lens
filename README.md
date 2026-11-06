# Cobot Cell Stop Lens

Measures protective stops and grip losses in real UR3 cobot telemetry, tests whether they can be predicted a few seconds ahead, and translates the downtime into cases lost at an end-of-line palletizer.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
At the end of a packaging line, the palletizing cell sets the pace for everything upstream. Each protective stop or dropped case costs output, and finding which signal moves before a stop is usually manual root-cause work.

## What it does
1. **Reliability, measured.** Detects 184 stop or grip-loss episodes in 7,409 one-second samples and computes MTBF 37.8 s, MTTR 2.9 s, availability 92.9 %, checked against a hand calculation.
2. **Joint load model.** A physics-informed regression of each joint's current on speed, acceleration, friction, temperature and coupling, fitted on clean cycles; the residual is the "unexpected load" signal.
3. **Early warning.** A gradient-boosted classifier predicts an onset in the next 10 s, cross-validated by cycle. PR-AUC 0.438 against 0.248 for a tool-current threshold and 0.236 for chance.
4. **Throughput translation.** Turns measured availability into cases lost per shift (409 at 12 cases/min, 8 h, as illustrative inputs).

The current page is a plain results view; the visual walkthrough is in progress.

## How it works
- `pipeline/reliability.py`: cleaning, segmentation at recording gaps, episode detection, MTBF/MTTR/availability.
- `pipeline/warning.py`: Savitzky-Golay acceleration, per-joint ridge load models, rolling features, GroupKFold gradient boosting, baselines.
- `pipeline/throughput.py`: downtime to cases lost.
- `pipeline/run.py`: writes static JSON to `public/data/`; the front end (Vite + TypeScript) reads it. No API keys.
- `tests/`: hand-calculation and leakage tests (pytest). Equations, assumptions and limits: [METHOD.md](METHOD.md).

## Run locally
```bash
npm install
npm run dev
```
Pipeline and tests: see METHOD.md, section 7.

## Data
UR3 CobotOps, Tyrovolas, Aliev, Antonelli, Stylios (2024), UCI Machine Learning Repository, doi:10.24432/C5J891, licensed CC BY 4.0. Real telemetry; no synthetic data. Case rate and shift length in the throughput step are illustrative inputs.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
