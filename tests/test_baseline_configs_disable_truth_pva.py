from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import load_yaml


BASELINE_CONFIGS = [
    Path("config_data2_baseline_ins_gnss_outage_best.yaml"),
    Path("config_data5_baseline_ins_gnss_outage_best.yaml"),
]


def test_canonical_baseline_configs_disable_truth_pva():
    for rel_path in BASELINE_CONFIGS:
        cfg = load_yaml(REPO_ROOT / rel_path)
        assert cfg["fusion"]["init"]["use_truth_pva"] is False, (
            f"{rel_path} should keep fusion.init.use_truth_pva=false "
            "so baseline experiments do not inherit truth-PVA initialization"
        )
