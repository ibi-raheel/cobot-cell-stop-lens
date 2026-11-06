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
    # 99.6% of an 8 h shift -> 480 * 0.004 = 1.92 min budget; a 2 min reset at 12 cases/min costs 24 cases,
    # so the budget fits 1.92 / 2 = 0.96 stops; 5 stops x 30 s x 12/min = 30 cases
    assert T.budget_min(8, 0.996) == pytest.approx(1.92)
    assert T.cases_lost(1, 120, 12) == pytest.approx(24.0)
    assert T.max_stops(8, 0.996, 120) == pytest.approx(0.96)
    assert T.cases_lost(5, 30, 12) == pytest.approx(30.0)
    sens = T.sensitivity()
    row = next(x for x in sens["rows"] if x["mttr_s"] == 120.0)
    assert row["availability"][0] == pytest.approx(1 - 2 / 480)


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


# ---------- cycle-phase hazard ----------
import phase as P


def phase_toy():
    # two cycles of 6 samples; onsets (rising edges of stop) at phase 2 in both cycles and phase 4 in cycle 2
    stop = [0, 0, 1, 0, 0, 0,  0, 0, 1, 0, 1, 0]
    df = pd.DataFrame(dict(cycle=[1] * 6 + [2] * 6, segment=0, stop=stop, grip=[False] * 12,
                           dt=1.0, **{f"Speed_J{j}": 0.0 for j in range(6)}))
    return P.add_phase(df)


def test_phase_index_resets_each_cycle():
    df = phase_toy()
    assert list(df["phase"]) == [0, 1, 2, 3, 4, 5] * 2


def test_hazard_hand_calc():
    df = phase_toy()
    down = (df["stop"] == 1).to_numpy()
    idx = R.rising_edges(df["stop"], df["segment"])
    assert list(idx) == [2, 8, 10]
    tab = P.hazard_table(df, dict(stop=idx), down, bins=[(0, 1), (2, 3), (4, 5)])
    # bin 2-3: rows 2,3,8,9; down rows 2,8 -> 2 up-seconds, onsets at 2 and 8 -> 2 per 2 s = 100 per 100 s
    b = tab[1]
    assert b["up_s"] == pytest.approx(2.0) and b["stop"]["n"] == 2
    assert b["stop"]["rate100"] == pytest.approx(100.0)
    # bin 4-5: rows 4,5,10,11; row 10 down -> 3 up-s, 1 onset
    assert tab[2]["up_s"] == pytest.approx(3.0) and tab[2]["stop"]["n"] == 1


def test_poisson_ci_known_values():
    lo, hi = P.poisson_ci(0, 1.0)
    assert lo == 0 and hi == pytest.approx(3.689, abs=1e-3)   # classic zero-event 95% upper bound
    lo, hi = P.poisson_ci(10, 1.0)
    assert lo == pytest.approx(4.795, abs=1e-3) and hi == pytest.approx(18.390, abs=1e-3)


def test_rate_ratio_equal_rates_is_one():
    rr = P.rate_ratio_test(20, 100.0, 20, 100.0)
    assert rr["rr"] == pytest.approx(1.0) and rr["lo"] < 1 < rr["hi"] and rr["p"] > 0.9


def test_events_needed_hand_calc():
    # (1.95996 + 0.84162)^2 * (1 + 1/0.5) / ln(0.5)^2 = 7.8489 * 3 / 0.48045 = 49.01
    assert P.events_needed(0.5) == pytest.approx(49.01, abs=0.02)
    assert P.run_hours_needed(0.25) == pytest.approx(49.01 / 0.25, rel=1e-3)


def test_pareto_toy():
    df = pd.DataFrame(dict(cycle=range(10)))
    ep = pd.DataFrame(dict(cycle=[0, 0, 1, 2], duration_s=[5.0, 3.0, 1.0, 1.0]))
    out = P.pareto(df, ep)
    assert out["top10pct_cycles"] == 1 and out["top10pct_share"] == pytest.approx(0.8)
    assert out["cycles_with_episode"] == 3 and out["cycles_for_half"] == 1


# ---------- reliability statistics ----------
def test_weibull_recovers_known_shape():
    rng = np.random.default_rng(1)
    assert R.weibull_fit(rng.exponential(40, 2000), n_boot=50)["shape"] == pytest.approx(1.0, abs=0.06)
    assert R.weibull_fit(40 * rng.weibull(2.0, 2000), n_boot=50)["shape"] == pytest.approx(2.0, abs=0.12)


def test_mtbf_ci_formula_and_brackets_point():
    from scipy.stats import chi2
    lo, hi = R.mtbf_ci(1000.0, 10)
    assert lo == pytest.approx(2000 / chi2.ppf(0.975, 22)) and hi == pytest.approx(2000 / chi2.ppf(0.025, 20))
    assert lo < 100 < hi


def test_interarrivals_drop_cross_segment_pairs():
    t = pd.to_datetime([0, 10, 25, 1000, 1030], unit="s", utc=True)
    df = pd.DataFrame(dict(t=t, segment=[0, 0, 0, 1, 1]))
    assert list(R.interarrivals(df, np.array([0, 1, 2, 3, 4]))) == [10.0, 15.0, 30.0]


# ---------- stronger baselines ----------
def test_phase_rate_fit_hand_calc():
    phase = np.array([0, 0, 1, 1, 1, 1])
    y = np.array([1, 0, 0, 0, 0, 1])
    tab = W.phase_rate_fit(phase, y, prior_n=0.0)
    assert tab[0] == pytest.approx(0.5) and tab[1] == pytest.approx(0.25)
    tab = W.phase_rate_fit(phase, y, prior_n=2.0)   # shrink toward overall 1/3
    assert tab[0] == pytest.approx((1 + 2 / 3) / 4)


def test_bootstrap_ap_perfect_scorer_and_diff():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    groups = np.repeat(np.arange(40), 10)
    mask = np.ones(400, bool)
    out = W.bootstrap_ap(y, dict(perfect=y.astype(float), noise=rng.random(400)), mask, groups,
                         n_boot=100, diffs=[("perfect", "noise")])
    assert out["perfect"]["lo"] == pytest.approx(1.0)
    assert out["perfect-noise"]["lo"] > 0.3 and out["perfect-noise"]["p_le_0"] == 0.0


@pytest.mark.skipif(not os.path.exists(XLSX), reason="dataset not downloaded")
def test_time_ordered_split_is_disjoint_in_time():
    df = R.load(XLSX)
    seg = df["segment"].to_numpy()
    assert df.loc[seg == 2, "t"].min() > df.loc[np.isin(seg, [0, 1]), "t"].max()


@pytest.mark.skipif(not os.path.exists(XLSX), reason="dataset not downloaded")
def test_phase_bins_account_for_every_stop_onset():
    df = P.add_phase(R.load(XLSX))
    down = ((df["stop"] == 1) | df["grip"]).to_numpy()
    idx = R.rising_edges(df["stop"], df["segment"])
    tab = P.hazard_table(df, dict(stop=idx), down)
    assert sum(b["stop"]["n"] for b in tab) == 108
    assert sum(b["share_up"] for b in tab) == pytest.approx(1.0)
