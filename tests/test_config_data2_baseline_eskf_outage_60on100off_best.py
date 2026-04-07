from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


def test_formal_best_config_matches_recovered_60on100off_bias_contract():
    cfg = load_yaml(REPO_ROOT / "config_data2_baseline_eskf_outage_60on100off_best.yaml")
    fusion = cfg["fusion"]
    noise = fusion["noise"]
    init_cfg = fusion["init"]
    gnss_schedule = fusion["gnss_schedule"]

    assert noise["sigma_ba"] == pytest.approx(0.000125)
    assert noise["sigma_bg"] == pytest.approx(5.825e-05)
    assert noise["sigma_ba_vec"] == pytest.approx([0.000125, 0.000125, 0.000125])
    assert noise["sigma_bg_vec"] == pytest.approx([5.825e-05, 5.825e-05, 5.825e-05])
    assert init_cfg["std_ba"] == pytest.approx([0.005625, 0.0005625, 0.00028125])
    assert init_cfg["std_bg"] == pytest.approx(
        [0.000181805130416076, 0.000136353847812057, 0.000181805130416076]
    )
    assert init_cfg["P0_diag"][9:12] == pytest.approx([3.1640625e-05, 3.1640625e-07, 7.91015625e-08])
    assert init_cfg["P0_diag"][12:15] == pytest.approx(
        [3.30531054456064e-08, 1.85923718131536e-08, 3.30531054456064e-08]
    )
    assert gnss_schedule["enabled"] is True
    assert gnss_schedule["enabled_windows"][0] == {"start_time": 528076.0, "end_time": 528836.0}
    assert gnss_schedule["enabled_windows"][1] == {"start_time": 528936.0, "end_time": 528996.0}
    assert gnss_schedule["enabled_windows"][-1] == {"start_time": 530376.0, "end_time": 530436.0}

