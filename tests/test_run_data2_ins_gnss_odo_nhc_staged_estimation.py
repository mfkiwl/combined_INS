import importlib.util
import math
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "run_data2_ins_gnss_odo_nhc_staged_estimation.py"


def load_module():
    assert MODULE_PATH.exists(), f"missing staged INS/GNSS/ODO/NHC runner: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "run_data2_ins_gnss_odo_nhc_staged_estimation",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_match_formal_data2_staged_eskf_contract():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")
    output_dir = Path("output/_tmp_data2_ins_gnss_odo_nhc_staged_estimation_contract")

    assert module.EXP_ID_DEFAULT == "EXP-20260408-data2-ins-gnss-odo-nhc-staged-estimation-r1"
    assert module.OUTPUT_DIR_DEFAULT == Path("output/data2_ins_gnss_odo_nhc_staged_estimation_r1")
    assert module.BASE_CONFIG_DEFAULT == Path("config_data2_baseline_ins_gnss_outage_best.yaml")
    assert module.CASE_ID == "data2_ins_gnss_odo_nhc_staged_estimation_eskf"

    cfg, metadata = module.build_case_config(base_cfg, output_dir)
    fusion = cfg["fusion"]
    init_cfg = fusion["init"]
    noise = fusion["noise"]
    runtime_phases = fusion["runtime_phases"]

    assert fusion["constraints"]["enable_odo"] is True
    assert fusion["constraints"]["enable_nhc"] is True
    assert fusion["ablation"]["disable_mounting_roll"] is True
    assert fusion["ablation"]["disable_mounting"] is False
    assert fusion["ablation"]["disable_odo_scale"] is False
    assert fusion["ablation"]["disable_odo_lever_arm"] is False
    assert fusion["ablation"]["disable_gnss_lever_arm"] is False

    assert init_cfg["gnss_lever_arm0"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["odo_scale"] == pytest.approx(1.0)
    assert init_cfg["mounting_pitch0"] == pytest.approx(0.0)
    assert init_cfg["mounting_yaw0"] == pytest.approx(0.0)
    assert init_cfg["lever_arm0"] == pytest.approx([0.0, 0.0, 0.0])
    assert init_cfg["P0_diag"][9:21] == pytest.approx(base_cfg["fusion"]["init"]["P0_diag"][9:21])
    assert init_cfg["P0_diag"][28:31] == pytest.approx([1.0, 1.0, 1.0])

    assert noise["sigma_ba"] == pytest.approx(base_cfg["fusion"]["noise"]["sigma_ba"])
    assert noise["sigma_bg"] == pytest.approx(base_cfg["fusion"]["noise"]["sigma_bg"])
    assert noise["sigma_sg"] == pytest.approx(base_cfg["fusion"]["noise"]["sigma_sg"])
    assert noise["sigma_sa"] == pytest.approx(base_cfg["fusion"]["noise"]["sigma_sa"])
    assert noise["sigma_lever_arm"] == pytest.approx(base_cfg["fusion"]["noise"]["sigma_lever_arm"])
    assert noise["sigma_gnss_lever_arm"] == pytest.approx(module.PHASE1_GNSS_LEVER_SIGMA_DEFAULT)
    assert noise["sigma_gnss_lever_arm_vec"] == pytest.approx([module.PHASE1_GNSS_LEVER_SIGMA_DEFAULT] * 3)

    assert [phase["name"] for phase in runtime_phases] == [
        "phase1_ins_gnss_estimate_gnss_lever",
        "phase2_ins_gnss_odo_nhc_activate_calibration",
        "phase3_periodic_gnss_outage_keep_calibrating",
    ]
    phase1, phase2, phase3 = runtime_phases
    assert phase1["constraints"]["enable_odo"] is False
    assert phase1["constraints"]["enable_nhc"] is False
    assert phase1["ablation"]["disable_odo_scale"] is True
    assert phase1["ablation"]["disable_mounting"] is True
    assert phase1["ablation"]["disable_odo_lever_arm"] is True
    assert phase1["ablation"]["disable_gnss_lever_arm"] is False

    assert phase2["constraints"]["enable_odo"] is True
    assert phase2["constraints"]["enable_nhc"] is True
    assert phase2["ablation"]["disable_mounting_roll"] is True
    assert phase2["ablation"]["disable_mounting"] is False
    assert phase2["ablation"]["disable_odo_scale"] is False
    assert phase2["ablation"]["disable_odo_lever_arm"] is False
    assert phase2["ablation"]["disable_gnss_lever_arm"] is True
    assert "phase_entry_init_overrides" not in phase2
    assert phase2["phase_entry_std_overrides"]["std_odo_scale"] == pytest.approx(module.PHASE2_ODO_SCALE_STD_DEFAULT)
    assert phase2["phase_entry_std_overrides"]["std_mounting_pitch"] == pytest.approx(module.PHASE2_MOUNTING_STD_DEG_DEFAULT)
    assert phase2["phase_entry_std_overrides"]["std_mounting_yaw"] == pytest.approx(module.PHASE2_MOUNTING_STD_DEG_DEFAULT)
    assert phase2["phase_entry_std_overrides"]["std_lever_arm"] == pytest.approx(
        [module.PHASE2_ODO_LEVER_STD_DEFAULT] * 3
    )

    assert phase3["constraints"]["enable_odo"] is True
    assert phase3["constraints"]["enable_nhc"] is True
    assert phase3["ablation"]["disable_mounting_roll"] is True
    assert phase3["ablation"]["disable_gnss_lever_arm"] is True
    assert "phase_entry_init_overrides" not in phase3
    assert "phase_entry_std_overrides" not in phase3

    assert metadata["case_id"] == module.CASE_ID
    assert metadata["phase1_window"] == [528076.0, 528276.0]
    assert metadata["phase2_window"] == [528276.0, 528776.0]
    assert metadata["phase3_window"] == [528776.0, 530488.9]
    assert metadata["gnss_on_windows"][0] == [528076.0, 528836.0]
    assert metadata["gnss_on_windows"][1] == [528896.0, 528956.0]
    assert metadata["gnss_off_windows"][0] == [528836.0, 528896.0]
    assert metadata["gnss_off_windows"][1] == [528956.0, 529016.0]


def test_build_case_config_supports_custom_phase3_gnss_schedule():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    _, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_staged_estimation_60on100off"),
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
    )

    assert metadata["gnss_on_windows"][0] == [528076.0, 528836.0]
    assert metadata["gnss_on_windows"][1] == [528936.0, 528996.0]
    assert metadata["gnss_on_windows"][2] == [529096.0, 529156.0]
    assert metadata["gnss_off_windows"][0] == [528836.0, 528936.0]
    assert metadata["gnss_off_windows"][1] == [528996.0, 529096.0]


def test_build_case_config_supports_inekf_filter_mode():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_staged_estimation_inekf"),
        filter_mode="InEKF",
    )

    assert metadata["filter_mode"] == "InEKF"
    assert metadata["case_id"] == "data2_ins_gnss_odo_nhc_staged_estimation_inekf"
    assert cfg["fusion"]["inekf"]["enable"] is True
    assert cfg["fusion"]["output_path"].endswith("SOL_data2_ins_gnss_odo_nhc_staged_estimation_inekf.txt")
    assert cfg["fusion"]["state_series_output_path"].endswith(
        "state_series_data2_ins_gnss_odo_nhc_staged_estimation_inekf.csv"
    )


def test_build_case_config_supports_odo_lever_process_noise_override():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_staged_estimation_odo_lever_q_override"),
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
        odo_lever_process_noise=1.0e-4,
    )

    noise = cfg["fusion"]["noise"]
    assert noise["sigma_lever_arm"] == pytest.approx(1.0e-4)
    assert noise["sigma_lever_arm_vec"] == pytest.approx([1.0e-4, 1.0e-4, 1.0e-4])
    assert metadata["odo_lever_process_noise"] == pytest.approx(1.0e-4)


def test_build_case_config_supports_phase2_bgz_std_and_process_noise_scale():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_staged_estimation_bgz_noise_override"),
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
        phase2_bgz_std_degh=30.0,
        bgz_process_noise_scale=0.1,
    )

    noise = cfg["fusion"]["noise"]
    phase2 = cfg["fusion"]["runtime_phases"][1]
    base_bg_std = math.sqrt(float(base_cfg["fusion"]["init"]["P0_diag"][12]))
    base_sigma_bg_vec = list(map(float, base_cfg["fusion"]["noise"]["sigma_bg_vec"]))

    assert phase2["phase_entry_std_overrides"]["std_bg"] == pytest.approx(
        [
            base_bg_std,
            base_bg_std,
            module.degph_to_radps(30.0),
        ]
    )
    assert noise["sigma_bg_vec"] == pytest.approx(
        [
            base_sigma_bg_vec[0],
            base_sigma_bg_vec[1],
            base_sigma_bg_vec[2] * 0.1,
        ]
    )
    assert noise["sigma_bg"] == pytest.approx(base_sigma_bg_vec[0])
    assert metadata["phase2_bgz_std_degh"] == pytest.approx(30.0)
    assert metadata["bgz_process_noise_scale"] == pytest.approx(0.1)


def test_build_case_config_supports_fullrun_odo_lever_truth_fix():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_fix_odo_lever_truth_fullrun"),
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
        truth_fix_mode="fix_odo_lever_truth_fullrun",
    )

    fusion = cfg["fusion"]
    init_cfg = fusion["init"]
    runtime_phases = fusion["runtime_phases"]

    assert metadata["truth_fix_mode"] == "fix_odo_lever_truth_fullrun"
    assert init_cfg["lever_arm0"] == pytest.approx([0.2, -1.0, 0.6])
    assert fusion["constraints"]["odo_lever_arm"] == pytest.approx([0.2, -1.0, 0.6])
    assert all(phase["ablation"]["disable_odo_lever_arm"] is True for phase in runtime_phases)
    assert runtime_phases[1]["phase_entry_std_overrides"] == {
        "std_odo_scale": pytest.approx(module.PHASE2_ODO_SCALE_STD_DEFAULT),
        "std_mounting_pitch": pytest.approx(module.PHASE2_MOUNTING_STD_DEG_DEFAULT),
        "std_mounting_yaw": pytest.approx(module.PHASE2_MOUNTING_STD_DEG_DEFAULT),
    }


def test_build_case_config_supports_fullrun_mounting_truth_fix():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_fix_mounting_truth_fullrun"),
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
        truth_fix_mode="fix_mounting_truth_fullrun",
    )

    fusion = cfg["fusion"]
    init_cfg = fusion["init"]
    runtime_phases = fusion["runtime_phases"]

    assert metadata["truth_fix_mode"] == "fix_mounting_truth_fullrun"
    assert init_cfg["mounting_roll0"] == pytest.approx(0.0)
    assert init_cfg["mounting_pitch0"] == pytest.approx(0.36)
    assert init_cfg["mounting_yaw0"] == pytest.approx(1.37)
    assert all(phase["ablation"]["disable_mounting"] is True for phase in runtime_phases)
    assert runtime_phases[1]["phase_entry_std_overrides"] == {
        "std_odo_scale": pytest.approx(module.PHASE2_ODO_SCALE_STD_DEFAULT),
        "std_lever_arm": pytest.approx([module.PHASE2_ODO_LEVER_STD_DEFAULT] * 3),
    }


def test_build_case_config_supports_joint300_layout_with_frozen_gnss_lever_after_joint():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_ins_gnss_outage_best.yaml")

    cfg, metadata = module.build_case_config(
        base_cfg,
        Path("output/_tmp_data2_ins_gnss_odo_nhc_joint300_covonly_60on100off"),
        phase_layout="joint300",
        joint_estimation_duration=300.0,
        freeze_gnss_lever_after_joint=True,
        phase3_gnss_on_duration=60.0,
        phase3_gnss_off_duration=100.0,
    )

    runtime_phases = cfg["fusion"]["runtime_phases"]
    assert metadata["phase_layout"] == "joint300"
    assert metadata["joint_estimation_duration"] == pytest.approx(300.0)
    assert metadata["freeze_gnss_lever_after_joint"] is True
    assert [phase["name"] for phase in runtime_phases] == [
        "phase1_joint_ins_gnss_odo_nhc_estimate_all_calibration",
        "phase2_periodic_gnss_outage_freeze_gnss_lever_keep_odo_nhc_calibrating",
    ]

    phase1, phase2 = runtime_phases
    assert phase1["start_time"] == pytest.approx(528076.0)
    assert phase1["end_time"] == pytest.approx(528376.0)
    assert phase1["constraints"]["enable_odo"] is True
    assert phase1["constraints"]["enable_nhc"] is True
    assert phase1["ablation"]["disable_odo_scale"] is False
    assert phase1["ablation"]["disable_mounting"] is False
    assert phase1["ablation"]["disable_mounting_roll"] is True
    assert phase1["ablation"]["disable_odo_lever_arm"] is False
    assert phase1["ablation"]["disable_gnss_lever_arm"] is False
    assert "phase_entry_init_overrides" not in phase1
    assert "phase_entry_std_overrides" not in phase1

    assert phase2["start_time"] == pytest.approx(528376.0)
    assert phase2["end_time"] == pytest.approx(530488.9)
    assert phase2["constraints"]["enable_odo"] is True
    assert phase2["constraints"]["enable_nhc"] is True
    assert phase2["ablation"]["disable_odo_scale"] is False
    assert phase2["ablation"]["disable_mounting"] is False
    assert phase2["ablation"]["disable_mounting_roll"] is True
    assert phase2["ablation"]["disable_odo_lever_arm"] is False
    assert phase2["ablation"]["disable_gnss_lever_arm"] is True
    assert "phase_entry_init_overrides" not in phase2
    assert "phase_entry_std_overrides" not in phase2

    assert metadata["phase1_window"] == [528076.0, 528376.0]
    assert metadata["phase2_window"] == [528376.0, 530488.9]
    assert "phase3_window" not in metadata
    assert metadata["gnss_on_windows"][0] == [528076.0, 528436.0]
    assert metadata["gnss_on_windows"][1] == [528536.0, 528596.0]
    assert metadata["gnss_off_windows"][0] == [528436.0, 528536.0]
    assert metadata["gnss_off_windows"][1] == [528596.0, 528696.0]


def test_add_error_columns_falls_back_to_truth_reference_bg_z():
    module = load_module()
    plot_df = pd.DataFrame({"bg_z_degh": [12.5, 13.5]})
    truth_reference = {
        "states": {
            "bg_z": {
                "reference_value": 10.0,
            }
        }
    }

    out = module.add_error_columns(plot_df, truth_reference)

    assert out["bg_z_degh_err"].tolist() == pytest.approx([2.5, 3.5])


def test_compute_case_metrics_row_uses_staged_plot_columns():
    module = load_module()
    case_row = {
        "case_id": module.CASE_ID,
        "odo_accept_ratio": 0.75,
        "nhc_accept_ratio": 0.5,
    }
    plot_df = pd.DataFrame(
        {
            "yaw_err_deg": [-1.0, 2.5, -0.5],
            "bg_z_degh_err": [10.0, -15.0, 5.0],
        }
    )

    metrics = module.compute_case_metrics_row(case_row, plot_df)

    assert metrics["case_id"] == module.CASE_ID
    assert metrics["yaw_err_max_abs_deg"] == pytest.approx(2.5)
    assert metrics["bg_z_degh_err_max_abs"] == pytest.approx(15.0)
    assert metrics["odo_accept_ratio"] == pytest.approx(0.75)
    assert metrics["nhc_accept_ratio"] == pytest.approx(0.5)


def test_plot_outputs_follow_mainline_outage_style():
    module = load_module()

    assert module.MAINLINE_PLOT_KEYS == (
        "all_states_overview",
        "key_coupling_states",
        "position",
        "velocity",
        "attitude",
        "ba",
        "bg",
        "odo_scale",
        "mounting",
        "odo_lever",
        "gnss_lever",
    )


def test_calibration_plot_column_groups_cover_plotted_calibration_states():
    module = load_module()

    assert module.CALIBRATION_DELTA_COLUMNS["mounting_roll_deg"] == "truth_mounting_roll_deg"
    assert module.CALIBRATION_DELTA_COLUMNS["mounting_pitch_deg"] == "truth_mounting_pitch_deg"
    assert module.CALIBRATION_DELTA_COLUMNS["mounting_yaw_deg"] == "truth_mounting_yaw_deg"
    assert module.CALIBRATION_DELTA_COLUMNS["odo_lever_x_m"] == "truth_odo_lever_x_m"
    assert module.CALIBRATION_DELTA_COLUMNS["odo_lever_y_m"] == "truth_odo_lever_y_m"
    assert module.CALIBRATION_DELTA_COLUMNS["odo_lever_z_m"] == "truth_odo_lever_z_m"
    assert module.CALIBRATION_DELTA_COLUMNS["gnss_lever_x_m"] == "truth_gnss_lever_x_m"
    assert module.CALIBRATION_DELTA_COLUMNS["gnss_lever_y_m"] == "truth_gnss_lever_y_m"
    assert module.CALIBRATION_DELTA_COLUMNS["gnss_lever_z_m"] == "truth_gnss_lever_z_m"
    assert "odo_scale_state" not in module.CALIBRATION_DELTA_COLUMNS
    assert module.ABSOLUTE_CALIBRATION_COLUMNS["odo_scale_state"] == "truth_odo_scale_state"


def test_plot_frame_shifts_calibration_states_from_initial_value():
    module = load_module()
    state_frame = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "mounting_pitch_deg": [0.0, 0.2],
            "truth_mounting_pitch_deg": [0.36, 0.36],
            "odo_lever_y_m": [0.0, -0.1],
            "truth_odo_lever_y_m": [-1.0, -1.0],
            "odo_scale_state": [1.0, 1.01],
            "truth_odo_scale_state": [1.0, 1.0],
            "ba_x_mgal": [0.0, 1.0],
            "truth_ba_x_mgal": [10.0, 10.0],
        }
    )

    plot_frame = module.build_plot_frame_from_initial_delta(state_frame)

    assert plot_frame["mounting_pitch_deg"].tolist() == pytest.approx([0.0, 0.2])
    assert plot_frame["truth_mounting_pitch_deg"].tolist() == pytest.approx([0.36, 0.36])
    assert plot_frame["odo_lever_y_m"].tolist() == pytest.approx([0.0, -0.1])
    assert plot_frame["truth_odo_lever_y_m"].tolist() == pytest.approx([-1.0, -1.0])
    assert plot_frame["odo_scale_state"].tolist() == pytest.approx([1.0, 1.01])
    assert plot_frame["truth_odo_scale_state"].tolist() == pytest.approx([1.0, 1.0])
    assert plot_frame["ba_x_mgal"].tolist() == pytest.approx([0.0, 1.0])
    assert plot_frame["truth_ba_x_mgal"].tolist() == pytest.approx([10.0, 10.0])


def test_build_staged_truth_keys_to_hide_hides_calibration_truth_overlays():
    module = load_module()

    hidden = module.build_staged_truth_keys_to_hide({"bg_x_degh"})

    assert "bg_x_degh" in hidden
    for key in module.CALIBRATION_DELTA_COLUMNS:
        assert key in hidden


def test_build_staged_truth_keys_to_hide_can_restore_selected_calibration_truth_overlays():
    module = load_module()

    hidden = module.build_staged_truth_keys_to_hide(
        {"bg_x_degh"},
        visible_keys={"odo_scale_state", "mounting_pitch_deg", "gnss_lever_y_m"},
    )

    assert "bg_x_degh" in hidden
    assert "odo_scale_state" not in hidden
    assert "mounting_pitch_deg" not in hidden
    assert "gnss_lever_y_m" not in hidden
    assert "mounting_yaw_deg" in hidden
    assert "odo_lever_y_m" in hidden


def test_build_all_states_export_frame_preserves_state_columns():
    module = load_module()
    state_frame = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "mounting_pitch_deg": [0.0, 0.2],
            "truth_mounting_pitch_deg": [0.36, 0.36],
            "odo_lever_y_m": [0.0, -0.1],
            "truth_odo_lever_y_m": [-1.0, -1.0],
        }
    )

    export_frame = module.build_all_states_export_frame(state_frame)

    assert list(export_frame.columns) == list(state_frame.columns)
    assert export_frame["mounting_pitch_deg"].tolist() == pytest.approx([0.0, 0.2])
    assert export_frame["odo_lever_y_m"].tolist() == pytest.approx([0.0, -0.1])
