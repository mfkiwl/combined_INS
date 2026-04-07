from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


def test_pure_ins_gnss_best_config_keeps_2x_bias_noise_and_pure_ins_gnss_contract():
    cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_eskf_outage_60on100off_best.yaml")
    fusion = cfg["fusion"]
    noise = fusion["noise"]
    init_cfg = fusion["init"]
    constraints = fusion["constraints"]
    ablation = fusion["ablation"]
    runtime_phases = fusion["runtime_phases"]
    gnss_schedule = fusion["gnss_schedule"]

    assert constraints["enable_odo"] is False
    assert constraints["enable_nhc"] is False
    assert ablation["disable_odo_scale"] is True
    assert ablation["disable_mounting"] is True
    assert ablation["disable_odo_lever_arm"] is True
    assert ablation["disable_gnss_lever_arm"] is True
    assert ablation["disable_gyro_scale"] is False
    assert ablation["disable_accel_scale"] is False

    assert noise["sigma_acc"] == pytest.approx(0.0016666666666666668)
    assert noise["sigma_gyro"] == pytest.approx(1.454441043328608e-05)
    assert noise["sigma_ba"] == pytest.approx(0.0005)
    assert noise["sigma_bg"] == pytest.approx(4.84813681109536e-06)
    assert noise["sigma_sg"] == pytest.approx(0.0003)
    assert noise["sigma_sa"] == pytest.approx(0.0003)
    assert noise["markov_corr_time"] == pytest.approx(14400.0)
    assert noise["sigma_ba_vec"] == pytest.approx([0.0005, 0.0005, 0.0005])
    assert noise["sigma_bg_vec"] == pytest.approx([4.84813681109536e-06] * 3)
    assert noise["sigma_sg_vec"] == pytest.approx([0.0003, 0.0003, 0.0003])
    assert noise["sigma_sa_vec"] == pytest.approx([0.0003, 0.0003, 0.0003])
    for field in ["ba0", "bg0", "sg0", "sa0", "std_ba", "std_bg", "std_sg", "std_sa"]:
        assert field not in init_cfg
    assert init_cfg["P0_diag"][9:12] == pytest.approx([2.5e-07, 2.5e-07, 2.5e-07])
    assert init_cfg["P0_diag"][12:15] == pytest.approx([2.3504430539097885e-11] * 3)
    assert init_cfg["P0_diag"][15:18] == pytest.approx([8.999999999999999e-08] * 3)
    assert init_cfg["P0_diag"][18:21] == pytest.approx([8.999999999999999e-08] * 3)

    assert init_cfg["gnss_lever_arm0"] == pytest.approx([0.15, -0.22, -1.15])
    assert noise["sigma_gnss_lever_arm"] == pytest.approx(0.0)
    assert noise["sigma_gnss_lever_arm_vec"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["std_gnss_lever_arm"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["P0_diag"][28:31] == pytest.approx([0.0, 0.0, 0.0])

    assert gnss_schedule["enabled"] is True
    assert gnss_schedule["enabled_windows"][0] == {"start_time": 528076.0, "end_time": 528836.0}
    assert gnss_schedule["enabled_windows"][1] == {"start_time": 528936.0, "end_time": 528996.0}
    assert gnss_schedule["enabled_windows"][-1] == {"start_time": 530376.0, "end_time": 530436.0}

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
