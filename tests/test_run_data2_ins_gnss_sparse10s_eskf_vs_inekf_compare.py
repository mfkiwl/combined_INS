import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "run_data2_ins_gnss_sparse10s_eskf_vs_inekf_compare.py"
BASE_CONFIG_PATH = REPO_ROOT / "config_data2_baseline_ins_gnss_eskf_outage_60on100off_best.yaml"


def load_module():
    assert MODULE_PATH.exists(), f"missing sparse10s compare runner: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("run_data2_ins_gnss_sparse10s_eskf_vs_inekf_compare", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_sparse_gnss_file_keeps_stride_samples(tmp_path):
    module = load_module()
    source_path = tmp_path / "rtk.txt"
    output_path = tmp_path / "rtk_sparse.txt"
    source_path.write_text("\n".join(["0 30 114 10 1 1 1", "1 30 114 10 1 1 1", "2 30 114 10 1 1 1", "10 30 114 10 1 1 1", "11 30 114 10 1 1 1", "20 30 114 10 1 1 1"]) + "\n", encoding="utf-8")
    stats = module.write_sparse_gnss_file(source_path, output_path, start_time=0.0, final_time=20.0, stride_seconds=10.0)
    assert output_path.read_text(encoding="utf-8").strip().splitlines() == ["0 30 114 10 1 1 1", "10 30 114 10 1 1 1", "20 30 114 10 1 1 1"]
    assert stats["rows_raw"] == 6
    assert stats["rows_window"] == 6
    assert stats["rows_kept"] == 3


def test_build_case_config_uses_pure_ins_gnss_contract_and_unified_inekf_switch(tmp_path):
    module = load_module()
    base_cfg = load_yaml(BASE_CONFIG_PATH)
    sparse_gnss_path = REPO_ROOT / "output" / "_tmp_test_sparse10s_compare" / "rtk_sparse_10s.txt"
    eskf_cfg = module.build_case_config(base_cfg, case_spec=module.CASE_SPECS[0], case_output_dir=tmp_path / "eskf", sparse_gnss_path=sparse_gnss_path)
    inekf_cfg = module.build_case_config(base_cfg, case_spec=module.CASE_SPECS[1], case_output_dir=tmp_path / "inekf", sparse_gnss_path=sparse_gnss_path)
    for cfg, enabled in ((eskf_cfg, False), (inekf_cfg, True)):
        fusion = cfg["fusion"]
        assert fusion["runtime_phases"] == []
        assert fusion["enable_gnss_velocity"] is False
        assert fusion["gnss_schedule"]["enabled"] is True
        assert fusion["gnss_schedule"]["enabled_windows"] == [{"start_time": 528076.0, "end_time": 530488.9}]
        assert fusion["constraints"]["enable_odo"] is False
        assert fusion["constraints"]["enable_nhc"] is False
        assert fusion["constraints"]["enable_zupt"] is False
        assert fusion["ablation"]["disable_odo_scale"] is True
        assert fusion["ablation"]["disable_mounting"] is True
        assert fusion["ablation"]["disable_odo_lever_arm"] is True
        assert fusion["ablation"]["disable_gnss_lever_arm"] is True
        assert fusion["inekf"]["enable"] is enabled
        assert "fej" not in fusion
        assert "filter" not in fusion
    assert eskf_cfg["fusion"]["noise"]["sigma_ba"] == base_cfg["fusion"]["noise"]["sigma_ba"]
    assert eskf_cfg["fusion"]["noise"]["sigma_bg"] == base_cfg["fusion"]["noise"]["sigma_bg"]
    assert eskf_cfg["fusion"]["noise"]["sigma_gnss_lever_arm"] == 0.0
    assert eskf_cfg["fusion"]["init"]["gnss_lever_arm0"] == [0.15, -0.22, -1.15]
    assert eskf_cfg["fusion"]["init"]["P0_diag"] == base_cfg["fusion"]["init"]["P0_diag"]


def test_state_labels_are_explicit_and_clean():
    module = load_module()
    assert module.STATE_LABELS["ba_y_mgal"] == "ba_y"
    assert module.STATE_LABELS["ba_x_mgal"] == "ba_x"
    assert module.STATE_LABELS["bg_z_degh"] == "bg_z"
    assert module.STATE_LABELS["sg_x_ppm"] == "sg_x"


def test_sparse_plot_helper_disables_truth_overlay_for_bias_and_scale_states():
    module = load_module()
    plot_module = module
    assert plot_module.STATE_LABELS["sa_x_ppm"] == "sa_x"

    from scripts.analysis import generate_sparse_compare_plots as plots

    assert plots.should_draw_truth("p_n_m", error_mode=False) is True
    assert plots.should_draw_truth("yaw_deg", error_mode=False) is True
    assert plots.should_draw_truth("ba_x_mgal", error_mode=False) is False
    assert plots.should_draw_truth("bg_z_degh", error_mode=False) is False
    assert plots.should_draw_truth("sg_y_ppm", error_mode=False) is False
    assert plots.should_draw_truth("sa_z_ppm", error_mode=False) is False
    assert plots.should_draw_truth("p_n_m", error_mode=True) is False
