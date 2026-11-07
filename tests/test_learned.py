import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import learned as LM
import reliability as R
import phase as P
import warning as W

XLSX = os.path.join(ROOT, "downloads", "dataset_02052023.xlsx")


def toy_cycles(n_runs=20, per=10):
    n = n_runs * per
    t = pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC")
    df = pd.DataFrame(dict(t=t, cycle=np.repeat(np.arange(n_runs), per), segment=0))
    df["dt"] = 1.0
    return P.add_phase(df)


def test_blocked_folds_contiguous_and_purged():
    df = toy_cycles()
    folds = LM.blocked_folds(df, n_folds=5, purge=7)
    seen = []
    for tr, te in folds:
        assert np.all(np.diff(te) == 1)                      # one contiguous block of rows
        runs_te = set(df["cycle_run"].to_numpy()[te])
        assert runs_te.isdisjoint(df["cycle_run"].to_numpy()[tr])
        gap = np.min(np.abs(tr[:, None] - te[None, :]))
        assert gap > 7                                       # purge respected
        seen.extend(te)
    assert sorted(seen) == list(range(len(df)))              # every row tested exactly once


def test_integrated_angle_hand_calc():
    # speed 0.5 rad/s for 3 samples of 1 s, then a new cycle: angle 0.5, 1.0, 1.5, then resets to 0.5
    df = toy_cycles(n_runs=2, per=3)
    for j in range(6):
        df[f"Speed_J{j}"] = 0.5
    q = LM.integrated_angles(df)
    assert q[:, 0].tolist() == pytest.approx([0.5, 1.0, 1.5, 0.5, 1.0, 1.5])


def test_gravity_offset_absorbed_by_sin_cos():
    # current = 2*sin(q + 0.7): the unknown offset is linear in sin(q), cos(q) -> exact fit
    q = np.linspace(-3, 3, 200)
    y = 2 * np.sin(q + 0.7)
    A = np.column_stack([np.sin(q), np.cos(q)])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    assert coef == pytest.approx([2 * np.cos(0.7), 2 * np.sin(0.7)], abs=1e-9)


def test_windows_pad_and_relative_temperature():
    X = np.arange(5 * 25, dtype=float).reshape(5, 25)
    seg = np.array([0, 0, 0, 1, 1])
    Wn = LM.windows(X, seg, length=3)
    assert Wn.shape == (5, 25, 3)
    # row 1, channel 0: samples [0, 0, 1] (padded with the segment's first row) -> values [0, 0, 25]
    assert Wn[1, 0].tolist() == [0, 0, 25]
    # row 3 starts a new segment: no samples from segment 0 leak in
    assert Wn[3, 0].tolist() == [75, 75, 75]
    # temperature channels are changes relative to the last sample
    assert Wn[2, 18, -1] == 0 and Wn[2, 18, 0] == -50


def test_since_last_hand_calc():
    df = pd.DataFrame(dict(segment=[0, 0, 0, 0, 1, 1]))
    down = np.array([False, True, False, False, False, True])
    assert LM.since_last(df, down).tolist() == [300, 0, 1, 2, 300, 0]


def test_segment_fix_hand_calc():
    # window: 10 stops in 100 s; rest: 10 stops in 900 s. Average 20/1000 = 0.02/s -> window keeps 2.
    g = LM.segment_fix_gain(10, 100.0, 10, 900.0, mttr_s=3.0, observed_s=1060.0, n_other_episodes=0, other_down_s=0.0)
    assert g["stops_in_window_after"] == pytest.approx(2.0)
    assert g["stops_removed"] == pytest.approx(8.0)
    assert g["share_removed"] == pytest.approx(0.4)
    assert g["availability_before"] == pytest.approx(1000 / 1060)
    assert g["availability_after"] == pytest.approx(1000 / 1036)


def test_block_bootstrap_perfect_vs_random():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(int)
    s = {"perfect": y.astype(float), "rand": rng.random(400)}
    b = LM.block_bootstrap(y, s, np.ones(400, bool), np.arange(400) // 10, [("perfect", "rand")], "auc", n_boot=200)
    assert b["perfect"]["lo"] == pytest.approx(1.0)
    assert b["perfect-rand"]["lo"] > 0.3 and b["perfect-rand"]["p_le_0"] == 0.0


def test_cnn_learns_a_planted_pattern():
    # sanity check of the training loop only (synthetic): label = channel 0 spikes in the last sample
    rng = np.random.default_rng(0)
    Wn = rng.normal(size=(600, 25, 12)).astype(np.float32)
    y = (rng.random(600) < 0.3).astype(float)
    Wn[y == 1, 0, -1] += 4.0
    Pn = np.zeros((600, 17))
    s = LM.train_cnn(Wn[:400], Pn[:400], y[:400], Wn[400:], Pn[400:], epochs=60, seeds=(0,))
    assert LM.auc(y[400:].astype(int), s, np.ones(200, bool)) > 0.85


@pytest.mark.skipif(not os.path.exists(XLSX), reason="dataset not downloaded")
def test_physics_model_beats_linear_on_heldout_blocks():
    df = R.load(XLSX)
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    df = P.add_phase(W.add_kinematics(df))
    clean = ~down & df["stop"].notna().to_numpy()
    q = LM.integrated_angles(df)
    folds = LM.blocked_folds(df)
    phys = LM.residual_r2_blocked(df, q, clean, folds, "physics")
    lin = LM.residual_r2_blocked(df, q, clean, folds, "ridge")
    assert np.mean(phys) > np.mean(lin)


def test_lead_labels_hand_calc():
    # one onset at row 6, horizon 4, min lead 3: rows 2,3 are 4 and 3 samples ahead -> positive;
    # rows 4,5 are only 2 and 1 ahead -> dropped from the mask; row 6 itself is down -> dropped.
    df = pd.DataFrame(dict(segment=[0] * 10, stop=[0.0] * 10))
    down = np.zeros(10, bool)
    down[6] = True
    y, m = LM.lead_labels(df, np.array([6]), down, horizon=4, min_lead=3)
    assert y.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert m.tolist() == [True, True, True, True, False, False, False, True, True, True]


def test_lead_labels_min_lead_1_matches_plain_labels():
    df = pd.DataFrame(dict(segment=[0] * 6 + [1] * 6, stop=[0.0] * 12))
    down = np.zeros(12, bool)
    on = np.array([4, 7])
    y1, m1 = LM.lead_labels(df, on, down, horizon=3, min_lead=1)
    y2, m2 = W.labels(df, on, down, horizon=3)
    assert y1.tolist() == y2.tolist() and m1.tolist() == m2.tolist()
    # the onset at row 7 (segment 1) never labels row 5 (segment 0)
    assert y1[5] == 0


def test_cnn_drop_channel_removes_the_signal():
    # same planted pattern as above, but channel 0 is silenced: the model must fall back to chance
    rng = np.random.default_rng(0)
    Wn = rng.normal(size=(600, 25, 12)).astype(np.float32)
    y = (rng.random(600) < 0.3).astype(float)
    Wn[y == 1, 0, -1] += 4.0
    Pn = np.zeros((600, 17))
    s = LM.train_cnn(Wn[:400], Pn[:400], y[:400], Wn[400:], Pn[400:], epochs=30, seeds=(0,), drop=(0,))
    assert LM.auc(y[400:].astype(int), s, np.ones(200, bool)) < 0.65


def test_shipped_acceleration_looks_ahead_causal_one_does_not():
    """A speed step at row 10: the shipped centred filter already reacts at row 7 (3 samples early);
    the causal filter used in check_leakage.py stays exactly zero until the step arrives."""
    import check_leakage as CL
    n = 30
    sp = np.where(np.arange(n) >= 10, 1.0, 0.0)
    df = pd.DataFrame({f"Speed_J{j}": sp for j in range(6)})
    df["segment"] = 0
    shipped = W.add_kinematics(df)["Acc_J0"].to_numpy()
    causal = CL.causal_acc(df)["Acc_J0"].to_numpy()
    assert abs(shipped[7]) > 1e-6          # look-ahead: row 7 sees the step at row 10
    assert np.all(causal[:10] == 0.0)      # no look-ahead
    assert causal[10] > 0


def test_since_last_by_kind():
    """Recency per event kind: rows since the last grip row, reset at a segment boundary."""
    df = pd.DataFrame(dict(segment=[0] * 6 + [1] * 3))
    grip = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    out = LM.since_last(df, grip)
    assert out[0] == 300 and out[1] == 0 and out[5] == 4 and out[6] == 300
