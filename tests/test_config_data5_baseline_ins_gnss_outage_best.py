from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


def test_data5_pure_ins_gnss_outage_best_config_keeps_validated_noise_and_contract():
    cfg = load_yaml(REPO_ROOT / "config_data5_baseline_ins_gnss_outage_best.yaml")
    fusion = cfg["fusion"]
    noise = fusion["noise"]
    init_cfg = fusion["init"]
    constraints = fusion["constraints"]
    ablation = fusion["ablation"]
    runtime_phases = fusion["runtime_phases"]
    gnss_schedule = fusion["gnss_schedule"]

    assert init_cfg["use_truth_pva"] is False
    assert constraints["enable_odo"] is False
    assert constraints["enable_nhc"] is False
    assert constraints["enable_zupt"] is False
    assert ablation["disable_odo_scale"] is True
    assert ablation["disable_mounting"] is True
    assert ablation["disable_odo_lever_arm"] is True
    assert ablation["disable_gnss_lever_arm"] is True

    assert noise["sigma_acc"] == pytest.approx(0.0005)
    assert noise["sigma_gyro"] == pytest.approx(8.726646259971648e-07)
    assert noise["sigma_ba"] == pytest.approx(0.00015000000000000001)
    assert noise["sigma_bg"] == pytest.approx(1.308996938995747e-07)
    assert noise["sigma_sg"] == pytest.approx(0.0003)
    assert noise["sigma_sa"] == pytest.approx(0.0003)
    assert noise["markov_corr_time"] == pytest.approx(14400.0)
    assert noise["sigma_ba_vec"] == pytest.approx([0.00015000000000000001] * 3)
    assert noise["sigma_bg_vec"] == pytest.approx([1.308996938995747e-07] * 3)
    assert noise["sigma_sg_vec"] == pytest.approx([0.0003] * 3)
    assert noise["sigma_sa_vec"] == pytest.approx([0.0003] * 3)

    assert init_cfg["gnss_lever_arm0"] == pytest.approx([0.136, -0.301, -0.184])
    assert noise["sigma_gnss_lever_arm"] == pytest.approx(0.0)
    assert noise["sigma_gnss_lever_arm_vec"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["std_gnss_lever_arm"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["P0_diag"][28:31] == pytest.approx([0.0, 0.0, 0.0])

    assert gnss_schedule["enabled"] is True
    assert gnss_schedule["enabled_windows"][0] == {"start_time": 456300.0, "end_time": 457060.0}
    assert gnss_schedule["enabled_windows"][1] == {"start_time": 457120.0, "end_time": 457180.0}
    assert gnss_schedule["enabled_windows"][-1] == {"start_time": 459640.0, "end_time": 459664.620611019}

    assert [phase["name"] for phase in runtime_phases] == [
        "phase1_ins_gnss_freeze_odo_states",
        "phase2_ins_gnss_freeze_odo_states",
        "phase3_periodic_gnss_outage_freeze_gnss_lever",
    ]
    for phase in runtime_phases:
        assert phase["constraints"]["enable_odo"] is False
        assert phase["constraints"]["enable_nhc"] is False
        assert phase["ablation"]["disable_odo_scale"] is True
        assert phase["ablation"]["disable_mounting"] is True
        assert phase["ablation"]["disable_odo_lever_arm"] is True
        assert phase["ablation"]["disable_gnss_lever_arm"] is True
