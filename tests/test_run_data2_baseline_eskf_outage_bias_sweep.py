import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "run_data2_baseline_eskf_outage_bias_sweep.py"


def load_module():
    assert MODULE_PATH.exists(), f"missing data2 outage bias sweep runner: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("run_data2_baseline_eskf_outage_bias_sweep", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_match_data2_outage_bias_sweep_contract():
    module = load_module()
    base_cfg = load_yaml(REPO_ROOT / "config_data2_baseline_eskf.yaml")
    output_dir = Path("output/_tmp_data2_outage_bias_sweep_contract")

    assert module.EXP_ID_DEFAULT == "EXP-20260407-data2-baseline-eskf-outage-60on100off-bias-sweep-r1"
    assert module.OUTPUT_DIR_DEFAULT == Path("output/data2_baseline_eskf_outage_60on100off_bias_sweep_r1")
    assert module.BASE_CONFIG_DEFAULT == Path("config_data2_baseline_eskf.yaml")
    assert module.PHASE3_GNSS_ON_DEFAULT == 60.0
    assert module.PHASE3_GNSS_OFF_DEFAULT == 100.0
    assert module.COARSE_SCALES_DEFAULT == (0.5, 1.0, 2.0)

    cfg, metadata = module.build_case_config(
        base_cfg,
        output_dir,
        scale=0.5,
        case_id="bias_scale_0p5x",
    )

    fusion = cfg["fusion"]
    noise = fusion["noise"]
    init_cfg = fusion["init"]
    enabled_windows = fusion["gnss_schedule"]["enabled_windows"]

    assert fusion["output_path"] == "output/_tmp_data2_outage_bias_sweep_contract/SOL_bias_scale_0p5x.txt"
    assert fusion["state_series_output_path"] == "output/_tmp_data2_outage_bias_sweep_contract/state_series_bias_scale_0p5x.csv"
    assert metadata["scale"] == pytest.approx(0.5)
    assert metadata["phase3_gnss_on_s"] == pytest.approx(60.0)
    assert metadata["phase3_gnss_off_s"] == pytest.approx(100.0)
    assert enabled_windows[0] == {"start_time": 528076.0, "end_time": 528836.0}
    assert enabled_windows[1] == {"start_time": 528936.0, "end_time": 528996.0}
    assert metadata["gnss_off_windows"][0] == [528836.0, 528936.0]
    assert metadata["gnss_off_windows"][1] == [528996.0, 529096.0]
    assert noise["sigma_ba"] == pytest.approx(0.00025)
    assert noise["sigma_bg"] == pytest.approx(0.0001165)
    assert noise["sigma_ba_vec"] == pytest.approx([0.00025, 0.00025, 0.00025])
    assert noise["sigma_bg_vec"] == pytest.approx([0.0001165, 0.0001165, 0.0001165])
    assert init_cfg["std_ba"] == pytest.approx([0.01125, 0.001125, 0.0005625])
    assert init_cfg["std_bg"] == pytest.approx(
        [0.000363610260832152, 0.000272707695624114, 0.000363610260832152]
    )
    assert init_cfg["P0_diag"][9:12] == pytest.approx(
        [0.0001265625, 1.265625e-06, 3.1640625e-07]
    )
    assert init_cfg["P0_diag"][12:15] == pytest.approx(
        [1.322124217824256e-07, 7.43694872526144e-08, 1.322124217824256e-07]
    )


def test_followup_scale_selection_expands_only_toward_improving_direction():
    module = load_module()

    assert module.build_followup_scale_candidates(0.5) == [0.25, 0.125]
    assert module.build_followup_scale_candidates(2.0) == [4.0, 8.0]
    assert module.build_followup_scale_candidates(1.0) == [0.75, 1.25]


def test_stop_rule_requires_material_phase3_improvement():
    module = load_module()

    assert module.is_material_improvement(
        current_best_rmse=1.60,
        candidate_rmse=1.57,
        abs_threshold_m=0.02,
        rel_threshold=0.015,
    )
    assert not module.is_material_improvement(
        current_best_rmse=1.60,
        candidate_rmse=1.585,
        abs_threshold_m=0.02,
        rel_threshold=0.015,
    )


def test_reference_comparison_labels_match_acceptance_goal():
    module = load_module()

    assert module.classify_reference_gap(1.58, 1.5911200711249691) == "better_than_reference"
    assert module.classify_reference_gap(1.60, 1.5911200711249691) == "near_reference"
    assert module.classify_reference_gap(1.70, 1.5911200711249691) == "worse_than_reference"
