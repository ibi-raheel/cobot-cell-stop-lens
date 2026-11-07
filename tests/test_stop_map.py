import json, os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import stop_map as SM
import throughput as T

XLSX = os.path.join(ROOT, "downloads", "dataset_02052023.xlsx")
DATA = os.path.join(ROOT, "public", "data")


def test_cost_line_matches_hand_check():
    # 99.6 % of an 8 h shift leaves 0.004 x 480 = 1.92 min; a 2-min reset at 12 cases/min loses 24 cases
    assert T.budget_min(8, 0.996) == pytest.approx(1.92)
    assert T.cases_lost(1, 120, 12) == pytest.approx(24)
    assert T.max_stops(8, 0.996, 120) == pytest.approx(0.96)


@pytest.mark.skipif(not os.path.exists(XLSX), reason="dataset not downloaded")
def test_stop_map_reproduces_shipped_results():
    s = json.load(open(os.path.join(DATA, "summary.json")))
    l = json.load(open(os.path.join(DATA, "learned.json")))
    m = SM.stop_map(XLSX, window="2-5", grip_window="15-30")
    st, gr = m["kinds"]["stop"], m["kinds"]["grip"]
    ref = s["phase"]["hot"]["stop_2_5"]
    assert (st["hot"]["n_in"], st["hot"]["n_out"]) == (ref["n_in"], ref["n_out"]) == (77, 31)
    assert st["hot"]["rr"] == pytest.approx(ref["rr"], rel=1e-6)
    assert st["is_hot"] and gr["is_hot"]
    assert gr["hot"]["n_in"] == s["phase"]["hot"]["grip_15_30"]["n_in"]
    assert st["fix"]["share_removed"] == pytest.approx(l["j2_fix"]["share_removed"], abs=1e-4)
    assert st["fix"]["share_lo"] == pytest.approx(l["j2_fix"]["share_removed_lo"], abs=1e-3)
    assert m["cost"]["cases_per_stop"] == pytest.approx(s["throughput"]["rows"][2]["cases_per_stop"])
    assert m["trial"]["events_baseline_arm"] == pytest.approx(s["phase"]["power"]["events_baseline_arm"], abs=0.1)
    # auto mode picks the same windows on this data
    a = SM.stop_map(XLSX)
    assert a["kinds"]["stop"]["window"] == [2, 5] and a["kinds"]["grip"]["window"] == [15, 30]
    assert "HOT" in SM.report(m)
