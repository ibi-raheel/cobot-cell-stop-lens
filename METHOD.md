# Method

## Research question

When a robot cell keeps throwing protective stops and dropping parts, which move in the cycle is causing it, what does each stop cost against a contracted uptime target, and what single change should be tested first? Section 3 answers where onsets cluster in the cycle and sizes the trial (3.1), 5.5 tests whether a learned model adds warning beyond cycle position, 5.6 estimates what fixing the hot move is worth, and 6 prices a stop against the uptime budget; [SOLUTION.md](SOLUTION.md) turns the answer into a playbook and the `pipeline/stop_map.py` tool.

All numbers below are produced by `pipeline/run.py` from the raw dataset and written to `public/data/summary.json`; the learned-model study (sections 5.5 and 5.6) is produced by `pipeline/run_learned.py` into `public/data/learned.json`. Nothing is tuned to a target result.

## 1. Data

**UR3 CobotOps** (Tyrovolas, Aliev, Antonelli, Stylios, 2024), UCI Machine Learning Repository, doi:[10.24432/C5J891](https://doi.org/10.24432/C5J891), licence **CC BY 4.0**. File `dataset_02052023.xlsx`.

- 7,409 rows x 24 columns, sampled at about 1 Hz (median interval 1.005 s).
- Per row: current, temperature and speed for joints J0 to J5, tool current, cycle id, `Robot_ProtectiveStop` (0/1), `grip_lost` (bool).
- 240 distinct cycle ids (numbered 1 to 264), 1,072 missing cells. 54 rows have no stop label: they are treated as **unknown** and excluded from scoring, never filled.
- The full dataset is used; no subsetting.

Cleaning: column names stripped, quoted ISO timestamps parsed, rows sorted by time. A gap longer than 30 s starts a new recording segment. The log has 3 segments (08:17 to 08:37, 08:37 to 09:06, 14:19 to 15:36 UTC). Time across a gap is not counted as observed time.

## 2. Episodes and reliability

A row is **down** if `Robot_ProtectiveStop == 1` or `grip_lost` is true. An **episode** is a maximal run of consecutive down rows inside one segment. Its duration runs from its first down sample to the first sample after it (if the episode ends a segment, one median interval is used).

With observed time $T$ (sum of within-segment intervals), total down time $D$ and $n$ episodes:

$$\text{MTBF} = \frac{T - D}{n}, \qquad \text{MTTR} = \frac{D}{n}, \qquad A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} = 1 - \frac{D}{T}$$

**Intervals.** MTBF uses the time-terminated chi-square interval $\left[\frac{2(T-D)}{\chi^2_{0.975,\,2n+2}},\ \frac{2(T-D)}{\chi^2_{0.025,\,2n}}\right]$. Availability uses a cycle-block bootstrap (1,000 resamples of the 240 cycles with replacement, $A = 1 - \sum D_c / \sum T_c$).

| Quantity | Value | 95 % interval |
|---|---|---|
| Observed time T | 7,479 s (2.08 h) | |
| Episodes n | 184 (108 stop onsets, 77 grip-loss onsets, 1 episode is both) | |
| Down time D | 531 s | |
| MTBF | 37.8 s | 32.7 to 43.9 s |
| MTTR | 2.89 s | |
| Availability A | 92.9 % | 91.6 to 94.1 % |

The onset counts (108, 77) are rising edges of each label and match the counts in the dataset description.

### 2.1 Wear-out or random trigger? (inter-arrival times)

Onset-to-onset times between down episodes (protective stop or grip loss) within a segment (181 gaps; pairs spanning a recording gap are dropped) are fitted with a two-parameter Weibull by maximum likelihood, $F(t) = 1 - e^{-(t/\lambda)^k}$. Shape $k > 1$ means the hazard rises with time since the last stop (wear-out, fixed by a maintenance interval); $k \approx 1$ means a constant hazard (memoryless, a trigger that fires at random times, fixed by removing the trigger).

| Data | n | Shape k (95 % bootstrap CI) | Scale λ | CV of gaps | KS vs exponential |
|---|---|---|---|---|---|
| All segments | 181 | **1.02 (0.92 to 1.19)** | 39.5 s | 1.23 | D = 0.103, p = 0.04 |
| Protective stops only, all segments | 105 | 0.97 (0.86 to 1.15) | 65.9 s | 1.21 | D = 0.115, p = 0.12 |
| Segment 0 | 20 | 0.82 (0.71 to 1.32) | 48.7 s | 1.49 | p = 0.05 |
| Segment 1 | 33 | 0.95 (0.81 to 1.33) | 42.3 s | 1.32 | p = 0.48 |
| Segment 2 | 128 | 1.13 (0.99 to 1.36) | 37.3 s | 1.04 | p = 0.04 |

Reading: every shape interval includes 1, so there is no evidence of wear-out between stops. The KS test rejects a pure exponential at the 5 % level, but the deviation is not a rising hazard: the gaps are lumpier than exponential (CV > 1 when segments are pooled), which is what a time-varying trigger rate gives (the later segment stops more often than the earlier ones). Because of 1 Hz sampling the gaps are near-integer, and the KS p-value is approximate because the mean is estimated (Lilliefors); estimating the mean makes the standard p-value conservative, so the true p is if anything smaller. Conclusion: within this two-hour log there is no sign of wear-out, and the pooled gaps are not strictly memoryless. Two hours cannot speak to wear over weeks or months, so this does not rule out a maintenance effect at that scale; it says that on this timescale stops are better explained by where the robot is in its cycle than by time since the last stop. Section 3 locates that.

**Validation (hand calculation).** `tests/test_pipeline.py` checks:
- a 12-sample toy log worked by hand: 11 s observed, 4 s down, 3 episodes, so MTBF = 7/3 s, MTTR = 4/3 s, A = 7/11;
- on the real data, an independent calculation (sum the forward interval of every down sample, $A = 1 - D/T$) agrees with the episode-table result to within 0.2 percentage points, and $A$ equals MTBF/(MTBF+MTTR) exactly;
- the chi-square MTBF interval against the closed form, and the Weibull fit recovering shape 1.0 and 2.0 from simulated exponential and Weibull(2) gaps;
- inter-arrival extraction drops pairs that span a recording gap.

Note: this is a test-cell recording where stops and grip losses are frequent by design. 92.9 % is the measured availability of this rig, not of a production cell.

## 3. Where in the cycle stops start (cycle-phase hazard)

**Phase** is the sample index within a run of one cycle id inside a segment (median cycle 25 samples, about 25 s). For a phase bin $b$, the hazard is onsets per up-second at risk:

$$h_b = \frac{k_b}{E_b}, \qquad E_b = \sum_{i \in b,\ \text{up, label known}} \Delta t_i$$

with an exact (Garwood) Poisson interval $\left[\tfrac{1}{2}\chi^2_{0.025,\,2k},\ \tfrac{1}{2}\chi^2_{0.975,\,2k+2}\right] / E_b$. A window is compared with the rest of the cycle by the rate ratio $\mathrm{RR} = (k_1/E_1)/(k_0/E_0)$; conditional on $k_1 + k_0$, $k_1$ is binomial with $p = E_1/(E_1+E_0)$, which gives an exact CI and p-value.

| Cycle samples | Up-seconds | Stop onsets | Stop rate / 100 up-s (95 % CI) | Grip-loss onsets | Grip rate / 100 up-s (95 % CI) |
|---|---|---|---|---|---|
| 0-1 | 500 | 0 | 0.0 (0 to 0.74) | 2 | 0.40 (0.05 to 1.45) |
| **2-5** | 835 | **77** | **9.23 (7.28 to 11.53)** | 15 | 1.80 (1.01 to 2.96) |
| 6-10 | 1,172 | 12 | 1.02 (0.53 to 1.79) | 1 | 0.09 (0.00 to 0.48) |
| 11-14 | 959 | 1 | 0.10 (0.00 to 0.58) | 4 | 0.42 (0.11 to 1.07) |
| **15-30** | 2,607 | 15 | 0.58 (0.32 to 0.95) | **49** | **1.88 (1.39 to 2.48)** |
| 31+ | 832 | 3 | 0.36 (0.07 to 1.05) | 6 | 0.72 (0.26 to 1.57) |

**Protective stops.** 77 of 108 stop onsets (**71 %**) start in cycle samples 2-5, which hold 12 % of up-time: 9.2 vs 0.51 onsets per 100 up-s for the rest of the cycle, **RR = 18.1 (95 % CI 11.8 to 28.4)**. Widening to samples 2-10 captures 89 of 108 (82 %) in 29 % of up-time, RR = 11.4 (6.9 to 19.9).

**Which motion.** The per-sample profile (mean signed joint speed on up rows, in `summary.json` → `phase.profile`) shows what is happening:

| Sample | J2 speed (rad/s) | J3 | J5 | Tool current (A) | Stop onsets |
|---|---|---|---|---|---|
| 1 | -0.02 | +0.07 | -0.01 | **0.28** (peak) | 0 |
| 2 | **-1.23** (fastest move of the cycle) | +0.41 | -0.31 | 0.11 | 1 |
| 3 | -0.32 | -0.17 | +0.18 | 0.11 | **33** |
| 4 | 0.00 | -0.38 | +0.29 | 0.09 | **19** |
| 5 | 0.00 | -0.02 | +0.01 | 0.18 | **24** |
| 10 | +0.91 | -0.16 | +0.17 | 0.09 | 3 |
| 12 | -1.01 | +0.23 | -0.24 | 0.09 | 0 |

Stops begin one to three samples after the J2 swing at sample 2, which follows the tool-current peak at sample 1 (most likely the gripper acting; the dataset has no gripper-state signal). A J2 swing of similar size later in the cycle (sample 12, -1.01 rad/s) is followed by almost no stops, so speed alone does not explain it. What differs is what came just before: the sample-2 move is the first move after the gripper acts. Candidate causes, in order of what I would test: the J2 speed/acceleration on that move with the payload just picked up, the payload/TCP setting the controller uses for collision detection on that move, and collision sensitivity. The data cannot separate these (no joint positions, no payload signal, 1 Hz); that is what a trial is for.

**Grip losses are a different failure.** Their rate is not raised in samples 2-10 (RR 0.64, CI 0.34 to 1.12) but is in samples 15-30, the transfer and place portion: 49 of 77 (64 %) in 38 % of up-time, RR 2.9 (1.8 to 4.8). Two failure modes, two motion segments, two different fixes (motion/collision settings vs gripper force and transfer acceleration).

**Is the hot window real or picked after looking?** The window was chosen from the pooled data, so its CI above is optimistic (it does not account for the choice). It is re-checked per recording segment:

| Segment | Stops | Samples 2-5: in / out, RR (95 % CI) | Samples 2-10: RR (95 % CI) |
|---|---|---|---|
| 0 (08:17-08:37 UTC) | 7 | 3 / 4, 4.5 (0.7 to 26.3), p = 0.07 | 1.4 (0.2 to 8.5) |
| 1 (08:37-09:06 UTC) | 14 | 9 / 5, 11.4 (3.4 to 43.1) | 8.0 (2.1 to 44.7) |
| 2 (14:19-15:36 UTC) | 87 | 65 / 22, 24.2 (14.7 to 41.2) | 17.0 (9.2 to 34.4) |

Segment 1 is a partial check: it was part of the data the window was picked from, and it has only 14 stops. Segment 2 holds 81 % of all stops, so the pooled result leans on that session. Segment 0 is too small to call either way. A clean confirmation needs a recording the window was not chosen on.

**Downtime Pareto.** 107 of 240 cycles have any episode. The worst 24 cycles (10 %) hold 49 % of all downtime; 25 cycles hold half.

### 3.1 Week 1 on a production palletizing cell

1. Pull the robot controller's stop log and the CMMS tickets for one cell and tag each protective stop with the program line or waypoint that was executing, the same way the phase index is used here.
2. Rank waypoints by stops per hour of exposure with the exact Poisson interval above, and split protective stops from dropped cases, since here they live in different motion segments.
3. For the hottest move, change one variable only: speed/acceleration on that move, the payload/TCP setting, or collision sensitivity.
4. Size the before/after comparison with a Poisson rate-ratio power calculation. To detect a 50 % drop at 80 % power and two-sided $\alpha = 0.05$, with equal exposure per arm and $\mathrm{Var}(\ln \widehat{\mathrm{RR}}) \approx 1/k_0 + 1/k_1$:

$$k_0 = \frac{(z_{0.975} + z_{0.80})^2 (1 + 1/\mathrm{RR})}{(\ln \mathrm{RR})^2} = \frac{7.85 \times 3}{0.480} = 49 \text{ stops in the baseline arm}$$

| Baseline stop rate | Run-hours per arm |
|---|---|
| This rig (52 stops/h) | 0.9 h |
| 10 stops per 8 h shift | 39 h |
| 5 per shift | 78 h |
| 2 per shift | 196 h |
| 1 per shift | 392 h |

At production stop rates a single cell needs weeks per arm, so the trial should pool the same move across several cells of the same configuration.

**Validation.** Unit tests check the hazard on a 12-row toy log worked by hand (bin 2-3: 2 onsets over 2 up-seconds), the Poisson interval against the textbook values (0 events: upper 3.689; 10 events: 4.795 to 18.390), the rate ratio of equal rates, the 49-event hand calculation, the Pareto on a toy table, and on the real data that the bins account for all 108 stop onsets and 100 % of up-time.

## 4. Joint load model (expected current)

Motor current is proportional to joint torque, and joint torque is the sum of inertial, friction and gravity terms. Without joint positions in the dataset, the gravity term cannot be computed directly, so each joint $j$ gets a linear model fitted on **clean rows only** (not down, label known):

$$\hat I_j = \beta_0 + \beta_1 \omega_j + \beta_2 \dot\omega_j + \beta_3\,\mathrm{sign}(\omega_j) + \beta_4 T_j + \sum_{k\ne j}(\gamma_k \omega_k + \delta_k \dot\omega_k)$$

- $\omega_j$: joint speed; $\dot\omega_j$: acceleration from a Savitzky-Golay derivative (window 7, order 2) within each segment;
- $\mathrm{sign}(\omega_j)$: Coulomb friction; $T_j$: joint temperature (viscosity, winding resistance);
- other joints' speed and acceleration: dynamic coupling.

Ridge regression ($\alpha = 1$). The residual $z_j = (I_j - \hat I_j)/\sigma_j$ is standardised by the clean-row residual SD. The table compares rows in the 10 s before an onset ("pre") with other clean rows, using the upper tail (P95 of $|z|$), not only the mean:

| Joint | R^2 (clean) | Residual SD (A) | mean abs z pre / clean | **P95 abs z pre / clean** | share abs z > 3 pre / clean |
|---|---|---|---|---|---|
| J0 | 0.18 | 0.77 | 0.48 / 0.43 | **1.89 / 1.48** | 3.9 % / 3.0 % |
| J1 | 0.23 | 0.70 | 0.67 / 0.70 | 2.12 / 2.42 | 2.3 % / 2.4 % |
| J2 | 0.28 | 0.51 | 0.64 / 0.67 | 1.90 / 2.10 | 1.7 % / 2.6 % |
| J3 | 0.27 | 0.44 | 0.58 / 0.56 | 2.18 / 2.41 | 3.0 % / 3.2 % |
| J4 | 0.10 | 0.60 | 0.51 / 0.48 | **2.22 / 2.07** | 3.0 % / 2.2 % |
| J5 | 0.61 | 0.08 | 0.80 / 0.75 | **2.00 / 1.87** | 1.8 % / 1.1 % |

Honest reading: the residual tail rises before onsets on the base and wrist joints (J0, J4, J5) and falls on the shoulder, elbow and wrist-1 joints (J1, J2, J3). So the residual is not a uniform "load builds up" signal. It is a pattern across joints, and the classifier (section 5) uses that pattern. At 1 Hz, without positions or payload, the linear model explains only 10 to 61 % of current variance.

## 5. Early warning

**Target.** Row $i$ is positive if an episode onset occurs in the next $H = 10$ samples (about 10 s) of the same segment. Rows that are already down, or have an unknown stop label, are excluded. 6,837 rows are scored; 23.2 % are positive (episodes are dense: one every 38 s).

**Features** (rolling window of 8 samples, per segment): mean and max of $|z_j|$ for each joint, 8-sample temperature change per joint, tool current, its rolling minimum and first difference (21 features).

**Model.** scikit-learn `HistGradientBoostingClassifier` (200 iterations, learning rate 0.05, 15 leaves, L2 = 1).

**Baselines.** (a) chance (uniform random score, seed 0); (b) single tool-current threshold (score = minus the 8-sample rolling minimum of tool current); (c) **cycle-phase baseline**: the positive rate by cycle sample index learned on the training folds, smoothed toward the overall rate with 5 pseudo-rows, $\hat p(\phi) = (\sum y_\phi + 5\bar y)/(n_\phi + 5)$. This is the honest bar: it knows where in the cycle stops happen but nothing about the signals.

**Validation designs.**
1. **5-fold GroupKFold by cycle**: no cycle appears in both train and test (asserted in a test). All scores are out-of-fold. This split is optimistic: neighbouring cycles land in different folds, and 56 % of the 10 s pre-onset windows span two cycles, so the same onset labels rows on both sides of a fold boundary and 8-sample rolling features overlap. Section 5.4 repeats it with blocked folds.
2. **Time-ordered holdout**: train on the two earlier segments (0-1, 08:17-09:06 UTC, 2,654 rows, 487 positives), test on the later segment (2, 14:19-15:36 UTC, 4,183 rows, 1,102 positives). This is how a model trained on past runs would be used. A test asserts the test segment starts after the training segments end.

**Uncertainty.** 95 % intervals from 1,000 cycle-block bootstrap resamples (all rows of a resampled cycle enter together), for each PR-AUC and for paired differences.

### 5.1 Feature-group ablation (GroupKFold by cycle)

| Scorer | PR-AUC (95 % CI) |
|---|---|
| Chance | 0.236 (0.209 to 0.266) |
| Tool current only (GBM) | 0.234 (0.205 to 0.267) |
| Tool-current threshold | 0.248 (0.216 to 0.289) |
| Temperature only (GBM) | 0.255 (0.226 to 0.286) |
| **Cycle-phase baseline** | **0.327 (0.286 to 0.376)** |
| Joint-load residuals only (GBM) | 0.423 (0.372 to 0.477) |
| Full model, 21 features | **0.438 (0.391 to 0.486)** |
| Full model + phase | 0.472 (0.422 to 0.524) |
| Residuals + phase | 0.479 (0.423 to 0.536) |

Paired differences: full model minus phase baseline +0.111 (CI +0.066 to +0.154); residuals only minus phase baseline +0.096 (CI +0.049 to +0.143); residuals only minus full model (CI -0.038 to +0.009, not different). Tool current and temperature carry little or no skill on their own; under this split the model's skill over the phase baseline comes from the joint-load residuals. Section 5.4 shows most of that skill does not survive blocked folds, so read this table as relative, not as transferable skill.

### 5.2 Equal alert budgets (GroupKFold by cycle; optimistic, see 5.4)

Thresholds are set at a score quantile so every scorer raises alerts on the same share of rows. An onset is **caught** if at least one alert falls in the 10 samples before it. False alarms per hour = alerted rows not followed by an onset within 10 s, divided by observed hours.

| Scorer | Top 5 %: precision | onsets caught | false alarms / h | Top 2 %: precision | onsets caught | false alarms / h |
|---|---|---|---|---|---|---|
| Gradient-boosted model | **61.4 %** | 41.3 % | **63.5** | **68.6 %** | **23.4 %** | **20.7** |
| Cycle-phase baseline | 35.6 % | 45.7 % | 112.1 | 35.5 % | 17.4 % | 42.8 |
| Tool-current threshold | 31.8 % | 12.5 % | 113.6 | 21.7 % | 3.8 % | 52.0 |
| Chance | 25.7 % | 37.5 % | 122.3 | 24.8 % | 17.4 % | 49.6 |

Median lead time of the first alert for caught onsets: 8 s (model), 6 s (phase baseline). "Onsets caught" is dominated by event density (one onset every 38 s), so chance touches many onsets; at the top-10 % budget chance catches 64.7 % against the model's 59.2 %. Precision and false alarms per hour are the fairer comparison, but at the top-5 % budget the model touches 41 % of onsets against 38 % for chance alerts: its higher precision comes from alerts bunching inside pre-onset windows, not from reaching more onsets.

### 5.3 Time-ordered holdout (train on segments 0-1, test on segment 2)

| Scorer | PR-AUC on segment 2 (95 % CI) |
|---|---|
| Chance | 0.272 (0.239 to 0.307) |
| Tool-current threshold | 0.285 |
| Residuals only | 0.313 |
| Full model | 0.323 (0.272 to 0.384) |
| Full model + phase | 0.339 |
| Residuals + phase | 0.341 (0.288 to 0.406) |
| **Cycle-phase baseline** | **0.345 (0.295 to 0.401)** |

Base rate in the test segment: 26.3 %. Phase baseline minus chance: CI +0.033 to +0.121. Full model minus phase baseline: CI -0.075 to +0.035.

### 5.4 Blocked folds (runs of consecutive cycles)

Same model and phase baseline, 5-fold GroupKFold where each group is a block of consecutive cycle ids, so neighbouring cycles share a fold. Bootstrap as above.

| Block size | Full model | Cycle-phase baseline | Model minus phase (95 % CI) |
|---|---|---|---|
| 1 cycle (section 5.1) | 0.438 | 0.327 | +0.066 to +0.154 |
| 5 cycles | 0.343 (0.301 to 0.392) | 0.332 (0.289 to 0.384) | -0.036 to +0.058 |
| 10 cycles | 0.316 (0.277 to 0.357) | 0.307 (0.271 to 0.350) | -0.028 to +0.045 |
| 20 cycles | 0.293 (0.257 to 0.336) | 0.303 (0.267 to 0.347) | -0.047 to +0.026 |

The phase baseline barely moves, because it uses only the sample index. The model drops by about 0.1 as soon as neighbouring cycles share a fold. Most of its per-cycle edge came from the fold overlap, not from transferable signal. The precision and false-alarm figures in section 5.2 use the per-cycle folds and are inflated the same way.

**Honest reading (feature model).** Under per-cycle folds the model looks clearly better than the phase baseline (+0.11 PR-AUC). With blocked folds (section 5.4) and with the time-ordered split (section 5.3) that edge is gone: the model is no better than knowing where in the cycle the robot is, and only the phase structure carries over from the earlier sessions to the later one. For a fleet, that ranks the work: the cycle-window finding (section 3) is the result to act on, and it still needs confirmation on a recording it was not chosen on; for protective stops, a signal-based early warning needs more training hours, joint positions and a higher sample rate. Section 5.5 tests a stronger sequence model on stricter folds and finds the task where learning does win: grip loss.

### 5.5 Sequence model and physics residuals on time-blocked folds (the main test)

Sections 5.1 to 5.4 use a gradient-boosted model on hand-made rolling features. This section replaces them as the main early-warning test with a stronger model, a stricter split and a stronger baseline. Code: `pipeline/learned.py`, `pipeline/run_learned.py`; output `public/data/learned.json`.

**Physics residual model (inverse-dynamics style).** The dataset has no joint positions, so the angle of each joint is rebuilt by integrating speed within each cycle run, $q_j(t) = \sum_{\tau \le t} \omega_j(\tau)\,\Delta t$. The unknown offset at cycle start does not matter for gravity terms, because $A\sin(q + c) = A\cos c\,\sin q + A\sin c\,\cos q$, which is linear in $\sin q$ and $\cos q$ (unit test). Expected current per joint:

$$\hat I_j = \beta^\top\big[\dot\omega_j,\ \omega_j,\ \mathrm{sign}\,\omega_j,\ \omega_j T_j,\ \mathrm{sign}(\omega_j) T_j,\ T_j,\ \sin/\cos(q_1),\ \sin/\cos(q_1{+}q_2),\ \sin/\cos(q_1{+}q_2{+}q_3),\ \sin/\cos(q_j),\ \dot\omega_{k\ne j},\ \omega_{k\ne j},\ \omega_1\omega_2,\ \omega_1^2,\ \omega_2^2,\ \omega_0^2,\ I_\text{tool}\big]$$

inertia, viscous and Coulomb friction (both drifting with temperature), gravity of the shoulder-elbow-wrist chain, inertial and velocity coupling, centrifugal/Coriolis products, and tool current as a payload proxy. All terms are linear in the unknown dynamic parameters, so it is fitted by ridge regression ($\alpha = 1$) **inside each fold, on clean training rows only**. Held-out $R^2$ on the time blocks, physics vs the section 4 linear model:

| | J0 | J1 | J2 | J3 | J4 | J5 | mean |
|---|---|---|---|---|---|---|---|
| Physics model | 0.17 | **0.28** | **0.35** | **0.34** | **0.12** | **0.62** | **0.31** |
| Linear model (section 4) | 0.18 | 0.23 | 0.28 | 0.27 | 0.09 | 0.61 | 0.28 |

The gravity terms help most on the shoulder and elbow (J1, J2), as they should. A test asserts the physics model beats the linear one on held-out blocks.

**Sequence model.** A small 1D-CNN (two 3-tap convolutions, 32 channels, mean- and max-pooling over time, dropout 0.2, about 8k weights) reads the last 12 samples of 25 channels: 6 joint currents, 6 standardised physics residuals, 6 joint speeds, 6 temperature changes (relative to the newest sample) and tool current. Cycle phase enters the head as 16 one-hot buckets plus a ramp. AdamW (lr 3e-3, weight decay 1e-3), 15 epochs, class-weighted BCE, mean of 2 seeds. CPU PyTorch, 3 threads.

**Validation.** 5 contiguous time blocks of consecutive cycle runs, in time order; each block is the test fold once, and training rows within 20 samples of the block on either side are purged, so no 10 s label window or 12-sample input window straddles train and test (unit test: blocks are contiguous, disjoint in cycles, purged, and cover every row once). Every score is out-of-fold. Intervals: 1,000 bootstrap resamples of 5-cycle blocks, for each score and for paired differences.

**Baselines** (all fitted on the same training folds): chance; **cycle phase** (positive rate by sample index, smoothed as in section 5); **phase + recency** (gradient boosting on cycle phase and samples since the last down row: knows where the robot is and how long ago it last stopped, nothing about the signals); a **phase-only CNN** (same network, signal channels zeroed); and the section 5 gradient-boosted residual model refitted with the physics residuals.

**Lead time.** A 1 Hz flag can lag the physical event, so each target is also run with a 3 s minimum lead: a row is positive only if the onset is 3 to 10 samples ahead, and rows 1 to 2 samples before an onset are left out of scoring (`lead_labels`, hand-checked in a test). The model gets no credit for spotting an event that is already under way. Caveat: the acceleration input (section 4) is a centred 7-point Savitzky-Golay derivative, so the acceleration at row $i$ uses speeds up to $i+3$, and gaps are filled by two-sided interpolation. A 3 s lead therefore does not on its own rule out look-ahead; the causal re-run below does.

**Results (PR-AUC, 95 % CI).**

| Target | Base rate | Chance | Cycle phase | Phase + recency | Residual GBM | **1D-CNN** | CNN minus phase + recency |
|---|---|---|---|---|---|---|---|
| **Grip loss, 3-10 s ahead** | 8.1 % | 0.081 | 0.091 | 0.129 (0.092-0.169) | 0.380 | **0.549 (0.466-0.628)** | **+0.420 (+0.342 to +0.494)** |
| Grip loss, 1-10 s ahead | 9.8 % | 0.099 | 0.104 | 0.144 (0.104-0.190) | 0.380 | **0.563 (0.479-0.637)** | +0.419 (+0.341 to +0.481) |
| Any down episode, 10 s | 23.2 % | 0.236 | 0.277 | 0.306 (0.252-0.363) | 0.351 | **0.410 (0.351-0.472)** | +0.103 (+0.050 to +0.157) |
| Protective stop, 10 s | 13.7 % | 0.141 | 0.208 | 0.246 (0.183-0.321) | 0.228 | 0.272 (0.219-0.337) | +0.027 (-0.027 to +0.085) |
| Protective stop, 3-10 s ahead | 11.3 % | 0.118 | 0.195 | 0.232 (0.172-0.304) | 0.204 | 0.253 (0.197-0.330) | +0.021 (-0.036 to +0.080) |
| Any down episode, 30 s | 52.7 % | 0.530 | 0.455 | 0.539 | 0.561 | 0.617 (0.531-0.691) | +0.077 (+0.001 to +0.143) |

ROC-AUC for grip loss 3-10 s ahead: CNN 0.831, phase + recency 0.695, chance 0.49. The "phase + recency" column counts time since the last down episode of any kind; for grip loss a grip-aware version (below) is the fairer bar and scores 0.343, so the last column overstates the grip-loss gain by about half.

**Checks on the grip-loss result.**
- *Not the gripper current.* With the tool-current channel silenced the CNN scores 0.574 (0.493-0.650) at 3-10 s; the warning comes from the joint channels, not from the gripper signal itself.
- *Not phase.* The phase-only CNN on the any-episode target scores 0.304, level with phase + recency (0.306); the signal channels add the +0.10.
- *Every session.* Out-of-fold PR-AUC at 3-10 s, CNN vs phase + recency: session 0 (96 positive rows) 0.61 vs 0.19, session 1 (147) 0.66 vs 0.24, session 2 (298) 0.50 vs 0.11.
- *Not look-ahead in the features.* `pipeline/check_leakage.py` recomputes acceleration from a one-sided quadratic fit to the last 7 samples and forward-fills gaps (unit test: the shipped filter reacts to a speed step 3 samples early, the causal one does not). Grip loss 3-10 s ahead: CNN 0.567 (0.476-0.651) against 0.549 with the shipped features. Protective stop, 10 s: 0.280 against 0.272. No change in either conclusion.
- *Against a grip-aware baseline.* Grip losses come in runs: 33 of 77 onsets start within 25 samples of the previous grip onset. The phase + recency baseline counts samples since the last down row of any kind, which protective stops dominate, so it misses that. A gradient-boosted baseline on cycle phase, samples since the last down row, since the last stop and since the last grip loss scores **0.343 (0.267-0.417)** at 3-10 s; the CNN's edge over it is **+0.206 (+0.130 to +0.288)**, about half the +0.42 against the weaker baseline. Scored only on rows with no grip loss in the previous 25 samples (5,386 rows, 271 positive, base rate 0.050), the CNN scores 0.318 (0.193-0.450) against 0.089, difference +0.088 to +0.373; with no grip loss in the previous 12 samples, 0.401 against 0.126. So part of the warning is "a grip loss happened a few seconds ago", and part survives without it.

**Honest reading.** On protective stops the sequence model beats the cycle-phase baseline (+0.064, CI +0.029 to +0.112) but **not** the phase + recency baseline (+0.027, CI -0.027 to +0.085): knowing where the robot is in the cycle and how long since it last stopped captures most of what is predictable about stops at 1 Hz, which is why the stop story is the cycle window (section 3) and the fix (section 5.6), not an alarm. Where learning wins is **grip loss**: phase and time since the last down episode carry little there (0.13 against 0.08 for chance), and adding the time since the last grip loss lifts the baseline to 0.34, because grip losses come in runs. The CNN on joint currents, physics residuals and speeds reaches 0.55 with at least 3 s of lead, +0.21 over that grip-aware baseline, with the gripper channel removed and with causal features, and it still separates isolated grip losses (0.32 vs 0.09). This is one robot and 77 grip-loss onsets, and the CNN's configuration (window, epochs, learning rate) was set without a separate tuning split, so treat the size of the edge as a lead to confirm on a new recording, not a measured alarm performance. What in the joint signals precedes a grip loss (a payload swing, a slower transfer, a cycle variant) is not identified; that needs joint positions and a faster sample rate.

Note on the numbers in 5.4: there, "blocked" meant GroupKFold over shuffled blocks of 5 cycles, so training data came from both before and after the test block. Here each test fold is one contiguous stretch of time with a purge, which is stricter; the phase baseline drops (0.332 to 0.277 on the any-episode target) because stop-dense and stop-sparse stretches no longer mix.

**A ranking task that does not hold up.** One row per cycle at sample 1, target "this cycle throws a protective stop in samples 2-5" (247 cycles, 76 positive). The CNN ranks them at ROC-AUC 0.604 (0.549-0.657) against 0.554 for recency alone, a difference of CI -0.054 to +0.168: not significant. Which cycle will stop in the hot window is not predictable from the samples before it at 1 Hz.

### 5.6 What fixing the J2 swing would buy

If the stop rate in cycle samples 2-5 fell to the cycle-average rate (all protective-stop onsets over all up-time, 1.56 per 100 up-seconds), with up-time and the per-stop recovery (2.74 s, measured) held fixed:

$$n_\text{in}' = \bar\lambda\,E_\text{in}, \quad \bar\lambda = \frac{n_\text{in} + n_\text{out}}{E_\text{in} + E_\text{out}}, \quad \Delta n = n_\text{in} - n_\text{in}'$$

| | Now | After fix |
|---|---|---|
| Protective stops in samples 2-5 | 77 | 13.1 |
| Protective stops removed | | **63.9 of 108 (59 %, 95 % CI 50-69 %)** |
| Protective stops per up-hour | 56.3 | 23.0 |
| MTBF between down episodes | 37.5 s | 57.5 s |
| Availability (this rig) | 92.9 % | 95.1 % |

The interval comes from 1,000 resamples of 5-cycle blocks. It covers sampling noise only: samples 2-5 were chosen because they are the hottest window in this same data, so the in-sample share removed is biased upward (a hot window picked post hoc tends to look hotter than it will on new data). Read 59 % as an upper-end target for a trial, not a forecast. MTBF here uses up-time on rows with a known stop label (37.5 s), slightly different from the 37.8 s in section 2. Hand check (unit test): window 10 stops in 100 s, rest 10 in 900 s, average 0.02/s, so the window keeps 2 and loses 8 (40 % of stops), and availability with 3 s per stop goes from 1000/1060 to 1000/1036. The rig figures are not transferred to a production cell; the share of stops removed (about 60 %) is the number to test there, and section 3.1 sizes that trial.

## 6. Cost per stop against a contracted uptime budget

A palletizer at the end of a line is the bottleneck: every minute it is down, the line loses that minute's cases. The rig's availability (92.9 %) and its 2.9 s auto-resume are **not** transferred to a production cell; a production cell runs far fewer stops, and each needs a site recovery that is often a manual reset. The model is per stop:

$$L = N_\text{stops} \cdot \frac{\text{MTTR}_\text{site}}{60} \cdot c, \qquad B = 60\,S\,(1 - A_\text{target}), \qquad N_\text{max} = \frac{B}{\text{MTTR}_\text{site}/60}$$

with case rate $c$ (cases/min), shift $S$ (h) and target availability $A_\text{target}$. Inputs (assumptions, not measured): $c = 12$ cases/min, $S = 8$ h, $A_\text{target} = 99.6\,\%$, so the budget is $B = 1.92$ min of downtime per shift.

| Site recovery per stop | Cases lost per stop | Most stops per shift inside a 99.6 % budget | Cases lost at 5 stops/shift |
|---|---|---|---|
| 2.9 s (this rig's auto-resume) | 0.6 | 39.7 | 3 |
| 30 s | 6 | 3.8 | 30 |
| 2 min (manual reset) | 24 | **0.96** | 120 |
| 5 min | 60 | 0.38 | 300 |

Reading: with a manual reset, one protective stop per shift uses up the whole 99.6 % budget. On this rig, 71 % of protective stops start in one four-sample window of the cycle; if a fix halved the stop rate in samples 2-10, 41 % of protective stops would go (0.5 x 82 %). Whether a production cell shows the same concentration is exactly what step 1 of section 3.1 measures.

Hand check (unit test): 480 min x 0.004 = 1.92 min; a 2 min reset at 12 cases/min costs 24 cases; 1.92 / 2 = 0.96 stops; 5 stops x 30 s x 12/min = 30 cases.

## 7. Assumptions and limitations

- The UR3 is a pick-and-place test cobot, not a palletizer. Case rate, shift length, site recovery time and the 99.6 % target are inputs, not measurements.
- Stop and grip-loss labels are taken as given. Unknown labels are excluded.
- 1 Hz sampling hides sub-second transients; no joint positions, so no explicit gravity term and no way to name the waypoint, only the sample index in the cycle.
- The hot window was found on the pooled data; the per-segment re-check is not independent of that choice and segment 1 has only 14 stops. It has not been tested on an untouched recording or cell.
- The section 4 joint load models are fitted once on all clean rows (sections 5.1-5.4). In section 5.5 the physics residual model is refitted inside each fold on training rows only.
- Protective stops: the sequence model beats the cycle-phase baseline but not a phase + recency baseline (section 5.5). Grip loss is the task where the learned model wins: +0.21 PR-AUC over a grip-aware recency baseline (+0.42 over the weaker any-episode recency baseline); the precursor it uses is not identified, and part of the signal is the run structure of grip losses.
- The acceleration feature uses a centred filter (3 samples of look-ahead). The causal re-run in section 5.5 leaves the results unchanged, but the shipped `learned.json` still uses the centred filter.
- No separate tuning split: the CNN settings were fixed by hand and evaluated on the same folds that report the scores.
- One robot only, so the new-machine case cannot be tested.
- Only 2.08 h of observed time and one robot.
- The planned gripper force/torque screen on the Robot Execution Failures dataset was not built: that dataset is a separate robot with 15-sample force/torque windows and no shared cycle structure, so it could not be tied to these stop events. Grip losses here are analysed by cycle phase instead (section 3).

## 8. Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # includes CPU torch
mkdir -p downloads && curl -L -o downloads/ur3.zip "https://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip" && unzip -o downloads/ur3.zip -d downloads
OMP_NUM_THREADS=2 .venv/bin/python -I pipeline/run.py downloads/dataset_02052023.xlsx public/data
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/run_learned.py downloads/dataset_02052023.xlsx public/data   # about 7 min on 3 CPU threads
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/check_leakage.py downloads/dataset_02052023.xlsx public/data  # causal-feature and grip-aware-baseline checks
OMP_NUM_THREADS=3 .venv/bin/python -I pipeline/stop_map.py downloads/dataset_02052023.xlsx --window 2-5 --grip-window 15-30  # the stop map (SOLUTION.md)
.venv/bin/python -I -m pytest -q tests
```
