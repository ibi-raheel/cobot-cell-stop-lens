import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import reliability as R
import warning as W
import throughput as T

XLSX = os.path.join(ROOT, "downloads", "dataset_02052023.xlsx")


def toy():
    t = pd.date_range("2024-01-01", periods=12, freq="1s", tz="UTC")
    stop = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    grip = [False] * 12
    grip[6] = True
    df = pd.DataFrame(dict(t=t, cycle=[1] * 6 + [2] * 6, stop=stop, grip=grip))
    df["dt"] = df["t"].diff().dt.total_seconds()
    df["segment"] = 0
    return df


def test_episodes_toy():
    ep = R.episodes(toy())
    assert list(ep["start"]) == [2, 6, 8]
    assert list(ep["duration_s"]) == [2.0, 1.0, 1.0]


def test_availability_hand_calc_toy():
    df = toy()
    rel = R.reliability(df, R.episodes(df))
    # hand: observed 11 s, down 4 s, 3 episodes -> MTBF 7/3, MTTR 4/3, A = 7/11
    assert rel["mtbf_s"] == pytest.approx(7 / 3)
    assert rel["mttr_s"] == pytest.approx(4 / 3)
    assert rel["availability"] == pytest.approx(7 / 11)


def test_labels_horizon_and_mask():
    df = toy()
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    y, mask = W.labels(df, np.array([2, 6, 8]), down, horizon=2)
    assert list(y) == [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # row 6 is down, masked out below
    assert not mask[2] and not mask[6] and mask[0]


def test_throughput_hand_calc():
    # 12 cases/min, 8 h shift, A = 0.95 -> 24 min down -> 288 cases; half caught -> 144
    out = T.cases_lost(12, 8, 0.95, 0.5)
    assert out["down_min"] == pytest.approx(24.0)
    assert out["cases_lost"] == pytest.approx(288.0)
    assert out["cases_lost_after"] == pytest.approx(144.0)


@pytest.mark.skipif(not os.path.exists(XLSX), reason="dataset not downloaded")
class TestRealData:
    @pytest.fixture(scope="class")
    def data(self):
        df = R.load(XLSX)
        return df, R.episodes(df)

    def test_shape_and_onsets_match_published_counts(self, data):
        df, ep = data
        assert df.shape[0] == 7409
        rel = R.reliability(df, ep)
        assert rel["n_stop_onsets"] == 108
        assert rel["n_grip_onsets"] == 77

    def test_availability_matches_independent_hand_calc(self, data):
        df, ep = data
        rel = R.reliability(df, ep)
        # independent: count down samples weighted by their forward interval
        down = ((df["stop"] == 1) | df["grip"]).to_numpy()
        fwd = df["t"].shift(-1).sub(df["t"]).dt.total_seconds().to_numpy()
        same = (df["segment"].shift(-1) == df["segment"]).to_numpy()
        fwd = np.where(same, fwd, np.nanmedian(df["dt"]))
        hand_down = float(fwd[down].sum())
        hand_A = 1 - hand_down / R.observed_seconds(df)
        assert rel["availability"] == pytest.approx(hand_A, abs=2e-3)
        assert rel["availability"] == pytest.approx(rel["mtbf_s"] / (rel["mtbf_s"] + rel["mttr_s"]))

    def test_no_cycle_in_both_train_and_test(self, data):
        from sklearn.model_selection import GroupKFold
        df, _ = data
        g = df["cycle"].to_numpy()
        for tr, te in GroupKFold(5).split(g, g, g):
            assert not set(g[tr]) & set(g[te])
