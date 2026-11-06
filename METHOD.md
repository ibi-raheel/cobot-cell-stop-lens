# Method

All numbers below are produced by `pipeline/run.py` from the raw dataset and written to `public/data/summary.json`. Nothing is tuned to a target result.

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

| Quantity | Value |
|---|---|
| Observed time T | 7,479 s (2.08 h) |
| Episodes n | 184 (108 stop onsets, 77 grip-loss onsets, 1 episode is both) |
| Down time D | 531 s |
| MTBF | 37.8 s |
| MTTR | 2.89 s |
| Availability A | 92.9 % |

The onset counts (108, 77) are rising edges of each label and match the counts in the dataset description.

**Validation (hand calculation).** `tests/test_pipeline.py` checks:
- a 12-sample toy log worked by hand: 11 s observed, 4 s down, 3 episodes, so MTBF = 7/3 s, MTTR = 4/3 s, A = 7/11;
- on the real data, an independent calculation (sum the forward interval of every down sample, $A = 1 - D/T$) agrees with the episode-table result to within 0.2 percentage points, and $A$ equals MTBF/(MTBF+MTTR) exactly.

Note: this is a test-cell recording where stops and grip losses are frequent by design. 92.9 % is the measured availability of this rig, not of a production cell.

## 3. Joint load model (expected current)

Motor current is proportional to joint torque, and joint torque is the sum of inertial, friction and gravity terms. Without joint positions in the dataset, the gravity term cannot be computed directly, so each joint $j$ gets a linear model fitted on **clean rows only** (not down, label known):

$$\hat I_j = \beta_0 + \beta_1 \omega_j + \beta_2 \dot\omega_j + \beta_3\,\mathrm{sign}(\omega_j) + \beta_4 T_j + \sum_{k\ne j}(\gamma_k \omega_k + \delta_k \dot\omega_k)$$

- $\omega_j$: joint speed; $\dot\omega_j$: acceleration from a Savitzky-Golay derivative (window 7, order 2) within each segment;
- $\mathrm{sign}(\omega_j)$: Coulomb friction; $T_j$: joint temperature (viscosity, winding resistance);
- other joints' speed and acceleration: dynamic coupling.

Ridge regression ($\alpha = 1$). The residual $z_j = (I_j - \hat I_j)/\sigma_j$ is standardised by the clean-row residual SD.

| Joint | R^2 (clean rows) | Residual SD (A) | mean abs z, 10 s before onset | mean abs z, other clean rows |
|---|---|---|---|---|
| J0 | 0.18 | 0.77 | 0.48 | 0.43 |
| J1 | 0.23 | 0.70 | 0.67 | 0.70 |
| J2 | 0.28 | 0.51 | 0.64 | 0.67 |
| J3 | 0.27 | 0.44 | 0.58 | 0.56 |
| J4 | 0.10 | 0.60 | 0.51 | 0.48 |
| J5 | 0.61 | 0.08 | 0.80 | 0.75 |

Honest reading: at 1 Hz, without positions or payload, a linear load model explains only 10 to 61 % of current variance, and the residual rises only slightly (J0, J3, J4, J5) or not at all (J1, J2) before an onset. A higher sample rate and joint angles (for the gravity term) are the obvious next steps.

## 4. Early warning

**Target.** Row $i$ is positive if an episode onset occurs in the next $H = 10$ samples (about 10 s) of the same segment. Rows that are already down, or have an unknown stop label, are excluded. 6,837 rows are scored; 23.2 % are positive (episodes are dense: one every 38 s).

**Features** (rolling window of 8 samples, per segment): mean and max of $|z_j|$ for each joint, 8-sample temperature change per joint, tool current, its rolling minimum and first difference (21 features).

**Model.** scikit-learn `HistGradientBoostingClassifier` (200 iterations, learning rate 0.05, 15 leaves, L2 = 1). **5-fold GroupKFold by cycle**: no cycle appears in both train and test (asserted in a test). All scores are out-of-fold.

**Baselines.** (a) single tool-current threshold (score = minus the 8-sample rolling minimum of tool current); (b) chance (uniform random score, seed 0).

**Metrics.** PR-AUC on rows. For a threshold set at a score quantile (so every scorer raises alerts on the same share of rows), an onset is **caught** if at least one alert falls in the 10 samples before it. False alarms per hour = alerted rows that are not followed by an onset within 10 s, divided by observed hours (each alerted second counts once).

| Scorer | PR-AUC | Alert on top 5 % of rows: precision | onsets caught | false alarms / h | Top 2 %: onsets caught | false alarms / h |
|---|---|---|---|---|---|---|
| Gradient-boosted model | **0.438** | **61.4 %** | **41.3 %** | **63.5** | **23.4 %** | **20.7** |
| Tool-current threshold | 0.248 | 31.8 % | 12.5 % | 113.6 | 3.8 % | 52.0 |
| Chance | 0.236 | 25.7 % | 37.5 % | 122.3 | 17.4 % | 49.6 |

Median lead time of the first alert for caught onsets: 8 s (model).

Honest reading: the model's alerts are right about 2.4 times as often as chance and the threshold rule, and it raises about half the false alarms for the same alert budget. But because episodes are so dense, chance alone touches many onsets, and at looser thresholds (top 10 %) chance catches more onsets (64.7 %) than the model (59.2 %) by spreading its alerts. The single tool-current threshold is barely better than chance. The model is a useful filter, not a reliable predictor at this sample rate.

## 5. Throughput translation

An end-of-line palletizer is the bottleneck: every minute it is down, the line loses that minute's cases. For case rate $c$ (cases/min), shift length $S$ (h) and availability $A$:

$$D_\text{shift} = 60\,S\,(1-A), \qquad L = c\,D_\text{shift}$$

With the measured $A = 92.9\%$, $c = 12$ cases/min and $S = 8$ h (both are **input assumptions**, not measured): $D_\text{shift} = 34.1$ min and $L = 409$ cases per shift. If every onset the model catches at the top-5 % threshold (41.3 %) were avoided entirely, an upper bound, $L$ falls to 240 cases.

Hand check (unit test): $c = 12$, $S = 8$, $A = 0.95$ gives 24 min and 288 cases; catching half gives 144.

## 6. Assumptions and limitations

- The UR3 is a pick-and-place test cobot, not a palletizer. The case rate and shift length are illustrative inputs.
- Stop and grip-loss labels are taken as given. Unknown labels are excluded.
- 1 Hz sampling hides sub-second transients; no joint positions, so no explicit gravity term.
- Only 2.08 h of observed time and one robot: results will not transfer to a fleet without retraining.
- "Avoided if caught" is an upper bound; a real alert may only shorten recovery.

## 7. Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p downloads && curl -L -o downloads/ur3.zip "https://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip" && unzip -o downloads/ur3.zip -d downloads
OMP_NUM_THREADS=2 .venv/bin/python -I pipeline/run.py downloads/dataset_02052023.xlsx public/data
.venv/bin/python -I -m pytest -q tests
```
