# The solution: a stop map for any cell

**Question.** When a robot cell keeps throwing protective stops and dropping parts, which move in the cycle is causing it, what does each stop cost against a contracted uptime target, and what single change should be tested first?

**Answer on the reference data** (UR3 CobotOps, one robot, 2.08 h; see [README](README.md) and [METHOD.md](METHOD.md)):
- 71 % of protective stops (77 of 108) start in one 4-second window, cycle samples 2-5, right after the fastest J2 swing. That window is 12 % of up-time, so its stop rate is 18x the rest of the cycle (rate ratio 18.1 [11.8, 28.4]).
- Fixing that move would remove up to 59 % of stops [50 %, 69 %]: 56 to 23 stops per up-hour. "Up to", because the window was picked on the same data.
- Dropped parts come from a different move, transfer and place (samples 15-30, 49 of 77 grip losses, rate ratio 2.9 [1.8, 4.8]). A small 1D-CNN on physics-residual joint currents warns of them 3-10 s ahead at PR-AUC 0.55 [0.47, 0.63], against 0.34 for the best simple rule.
- For protective stops the model does not beat knowing the cycle position (0.272 vs 0.246, interval of the difference spans zero). Stops get a fix, not an alarm.
- At a 99.6 % uptime target, one 2-minute manual reset uses the whole shift's downtime budget (1.92 min per 8 h shift, which fits 0.96 such stops). Case rate, shift length, recovery time and the target are assumptions, not measurements.

Caveats that travel with the 18x: one robot, 2.08 h; one recording session holds 81 % of the stops, so the pooled ratio leans on it (the session with 14 stops gives 11.4 [3.4, 43.1]; the session with 7 is inconclusive).

## The playbook: find the move, fix the move

1. **Tag.** Export one cell's controller stop/fault log with the program line or waypoint (or the sample index in the cycle). Pull the recovery time for each stop from the maintenance system.
2. **Split.** Protective stops and dropped parts are separate failures. On the reference data they came from different moves and need different fixes.
3. **Rank.** Compare moves by onsets per up-hour with exact Poisson intervals.
4. **Check.** Confirm the hot move holds in each shift or session, so you are not reacting to one bad afternoon.
5. **Price.** Stops x site recovery time x case rate, against the contracted uptime budget.
6. **Change one variable on the hot move.** For stops: speed/acceleration, payload/TCP setting, or collision sensitivity. For drops: gripper force or transfer acceleration.
7. **Size the before/after.** Count stops before and after the change on the same move.
8. **Treat the grip-loss warning model as a lead** to confirm on the cell. It is not an alarm yet.
9. **Write the result into the build standard** (program defaults, end-of-arm tooling spec) for the next cells.

## Decision rules (thresholds from the measured results)

| Decision | Rule | On the reference data |
| --- | --- | --- |
| Call a move "hot" | Lower bound of its rate ratio vs the rest of the cycle is above 1 | Stops, samples 2-5: RR 18.1, lower bound 11.8. Grip losses, samples 15-30: RR 2.9, lower bound 1.8 |
| Trust it across time | Lower bound above 1 in each session that has enough events; flag sessions that are not | 11.4 and 24.2 in the two larger sessions; the 7-stop session is inconclusive |
| Stops vs drops | Rank separately; fix them separately | Different windows (2-5 vs 15-30) |
| Narrow winner | If the top window's interval overlaps the runner-up's, say so and test both | Grip losses: 15-30 (1.88 per 100 up-s) vs 2-5 (1.80) overlap |
| Value of a fix | Stops removed if the hot window ran at the cycle-average rate; report as "up to" when the window was picked on the same data | Up to 59 % [50 %, 69 %] |
| Is one stop affordable? | Stops allowed per shift = budget minutes / recovery minutes | 2-min reset at 99.6 %: 0.96 per shift |
| Trial size | 49 stops in the baseline arm detect a 50 % drop at 80 % power (alpha 0.05) | 0.9 h per arm on this rig; 39 h per arm at 10 stops per shift |
| Pool or not | If one cell will not reach 49 stops in a reasonable time, pool the same move across cells of the same configuration | |
| Alarm or fix | Ship a warning model only if it beats the phase + recency baseline with an interval above zero | Grip losses: +0.21 [+0.13, +0.29], a lead. Stops: interval spans zero, no alarm |

## Week 1 on a real line

- **Day 1.** Pull the robot controller's stop/fault log and the maintenance tickets for one cell, plus any dashboard export with program-line context. Map them to the tool's columns: stop flag, drop/grip flag, program waypoint, timestamps.
- **Day 2.** Run the stop map. Deliver the moves ranked by stops per up-hour with exact intervals, protective stops split from dropped parts, and a check that the ranking holds shift by shift.
- **Day 3.** Price the top moves with the site's real recovery time and case rate in place of the assumptions here (12 cases/min, 2-minute reset), against the contracted uptime.
- **Day 4.** Sit with deployment engineering and field service to pick one variable on the hottest move (speed/acceleration, payload/TCP, collision sensitivity; or gripper force for drops). Write the test plan with its sample size: 49 baseline stops to detect a 50 % drop, pooled across cells of the same configuration if needed.
- **Day 5.** Hand over this playbook filled in for that cell: the ranked list, the cost, the test plan, and a proposed change to the program or tooling standard for cells now being built.

Honest scope: week 1 ends with the diagnosis and a started trial. Confirming the fix takes as long as it takes to collect those 49 stops.

## Running the tool

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/stop_map.py downloads/dataset_02052023.xlsx --window 2-5 --grip-window 15-30
# or let it pick the windows:
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/stop_map.py cell_log.xlsx --recovery-s 120 --case-rate 12 --shift-h 8 --target 0.996 --json out.json
```

Input: an xlsx in the UR3 CobotOps column format (`Timestamp`, `cycle`, `Robot_ProtectiveStop`, `grip_lost`, joint channels). A controller log mapped to those columns works the same way, with the program waypoint index standing in for the cycle sample. `--window auto` (default) picks the top bin by rate.

Output on the reference data:

```
STOP MAP  2.08 h observed · 108 protective stops · 77 grip losses
Protective stops by cycle sample   n   rate/100 up-s [95% CI]
  2-5     77   9.23 [7.28, 11.53]   HOT  RR 18.1 [11.8, 28.4] · 71% of onsets in 12% of up-time
  6-10    12   1.02 [0.53, 1.79]
  15-30   15   0.58 [0.32, 0.95]
  ...
  per session RR: 4.5 (n.s.) / 11.4 / 24.2
Grip losses by cycle sample   n   rate/100 up-s [95% CI]
  15-30   49   1.88 [1.39, 2.48]   HOT  RR 2.9 [1.8, 4.8] · 64% of onsets in 38% of up-time
  2-5     15   1.80 [1.01, 2.96]
  ...
Test first: stops -> test speed/accel, payload/TCP or collision sensitivity on this move (samples 2-5)
            drops -> test gripper force / transfer acceleration on this move (samples 15-30)
Fix value (upper-end ...): 63.9 of 108 stops (59% [50%, 69%]) · 56 -> 23 stops/up-h
Cost @ 120 s recovery, 12 cases/min (assumed): 24 cases/stop · budget 1.92 min/shift fits 0.96 stops
Trial: 49 baseline stops for a 50% drop at 80% power · 0.9 h/arm here · 39 h/arm at 10 stops/shift
```

The tool is a thin wrapper over `reliability.py`, `phase.py`, `learned.py` (fix arithmetic) and `throughput.py`; it adds no new statistics. `tests/test_stop_map.py` checks that it reproduces the shipped results (77 / 31, RR 18.07, 59.2 % removed, 24 cases, 0.96 stops, 49 stops). The grip-loss warning model is not in the tool: it takes about 7 minutes to train and stays in `pipeline/run_learned.py`.
