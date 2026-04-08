from pathlib import Path


def test_stage_runner_module_path_is_reserved():
    path = Path("scripts/analysis/run_data2_ins_gnss_odo_nhc_staged_estimation.py")
    assert path.exists(), "formal staged INS/GNSS/ODO/NHC runner must exist at the reserved path"
