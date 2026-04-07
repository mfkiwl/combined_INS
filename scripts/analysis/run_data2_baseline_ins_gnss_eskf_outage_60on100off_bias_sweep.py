from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.odo_nhc_update_sweep import ensure_dir, load_yaml, rel_from_root, save_yaml
from scripts.analysis.run_data2_baseline_ins_gnss_outage_no_odo_nhc import (
    build_run_config as build_pure_run_config,
    mtime_text,
    normalize_repo_path,
    run_case as run_pure_case,
)
from scripts.analysis.run_data2_fullwindow_attitude_bias_coupling import build_state_frame, compute_case_metrics
from scripts.analysis.run_data2_staged_g5_no_imu_scale import compute_phase_metrics, format_metric, render_table
from scripts.analysis.run_data2_state_sanity_matrix import build_truth_reference, json_safe
from scripts.analysis.run_nhc_state_convergence_research import (
    build_truth_interp,
    load_pos_dataframe,
    merge_case_outputs,
)


EXP_ID_DEFAULT = "EXP-20260407-data2-baseline-ins-gnss-eskf-outage-60on100off-bias-sweep-r1"
OUTPUT_DIR_DEFAULT = Path("output/data2_baseline_ins_gnss_eskf_outage_60on100off_bias_sweep_r1")
BASE_CONFIG_DEFAULT = Path("config_data2_baseline_eskf.yaml")
SOLVER_DEFAULT = Path("build/Release/eskf_fusion.exe")
REFERENCE_OUTPUT_DIR_DEFAULT = Path("output/data2_baseline_eskf_outage_60on100off_r1")
PHASE3_GNSS_ON_DEFAULT = 60.0
PHASE3_GNSS_OFF_DEFAULT = 100.0
COARSE_SCALES_DEFAULT = (0.5, 1.0, 2.0)
EXTEND_LOWER_SCALES_DEFAULT = (0.25, 0.125)
EXTEND_UPPER_SCALES_DEFAULT = (4.0, 8.0)
INTERIOR_FOLLOWUP_SCALES_DEFAULT = (0.75, 1.25)
PHASE3_IMPROVEMENT_ABS_THRESHOLD_M_DEFAULT = 0.02
PHASE3_IMPROVEMENT_REL_THRESHOLD_DEFAULT = 0.015
REFERENCE_NEAR_THRESHOLD_M = 0.02
PHASE3_WINDOW_NAME = "phase3_periodic_gnss_outage"


@dataclass(frozen=True)
class CaseSpec:
    scale: float

    @property
    def case_id(self) -> str:
        return f"data2_baseline_ins_gnss_eskf_outage_60on100off_bias_{scale_slug(self.scale)}"

    @property
    def case_label(self) -> str:
        return f"data2 baseline ins gnss eskf outage 60on100off bias scale {self.scale:g}x"


def scale_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p") + "x"


def parse_scale_list(raw: str) -> tuple[float, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("scale list must not be empty")
    parsed = tuple(float(item) for item in values)
    for value in parsed:
        if value <= 0.0:
            raise ValueError("scale list values must be positive")
    return parsed


def build_followup_scale_candidates(best_scale: float) -> list[float]:
    best_scale = float(best_scale)
    if math.isclose(best_scale, min(COARSE_SCALES_DEFAULT), rel_tol=0.0, abs_tol=1.0e-12):
        return [float(value) for value in EXTEND_LOWER_SCALES_DEFAULT]
    if math.isclose(best_scale, max(COARSE_SCALES_DEFAULT), rel_tol=0.0, abs_tol=1.0e-12):
        return [float(value) for value in EXTEND_UPPER_SCALES_DEFAULT]
    return [float(value) for value in INTERIOR_FOLLOWUP_SCALES_DEFAULT]


def is_material_improvement(
    current_best_rmse: float,
    candidate_rmse: float,
    abs_threshold_m: float,
    rel_threshold: float,
) -> bool:
    current_best_rmse = float(current_best_rmse)
    candidate_rmse = float(candidate_rmse)
    improvement = current_best_rmse - candidate_rmse
    if improvement <= 0.0:
        return False
    if improvement >= float(abs_threshold_m):
        return True
    return (improvement / max(current_best_rmse, 1.0e-12)) >= float(rel_threshold)


def classify_reference_gap(phase3_rmse_m: float, reference_phase3_rmse_m: float) -> str:
    phase3_rmse_m = float(phase3_rmse_m)
    reference_phase3_rmse_m = float(reference_phase3_rmse_m)
    if phase3_rmse_m < reference_phase3_rmse_m - 1.0e-12:
        return "better_than_reference"
    if abs(phase3_rmse_m - reference_phase3_rmse_m) <= REFERENCE_NEAR_THRESHOLD_M:
        return "near_reference"
    return "worse_than_reference"


def sigma_ba_internal_to_mgal(value: float) -> float:
    return float(value) / 1.0e-5


def sigma_bg_internal_to_degh(value: float) -> float:
    return math.degrees(float(value)) * 3600.0


def phase3_reference_metrics(reference_output_dir: Path) -> dict[str, float]:
    phase_df = pd.read_csv(reference_output_dir / "phase_metrics.csv")
    phase3_row = phase_df.loc[phase_df["window_name"] == PHASE3_WINDOW_NAME].iloc[0]
    case_df = pd.read_csv(reference_output_dir / "case_metrics.csv")
    case_row = case_df.iloc[0]
    return {
        "phase3_rmse_3d_m": float(phase3_row["rmse_3d_m"]),
        "phase3_final_err_3d_m": float(phase3_row["final_err_3d_m"]),
        "overall_rmse3d_m": float(case_row["overall_rmse3d_m"]),
    }


def build_case_config(
    base_cfg: dict[str, Any],
    output_dir: Path,
    scale: float,
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_cfg, baseline_metadata = build_pure_run_config(
        base_cfg,
        output_dir,
        fix_gnss_lever_truth=True,
        phase3_gnss_on=PHASE3_GNSS_ON_DEFAULT,
        phase3_gnss_off=PHASE3_GNSS_OFF_DEFAULT,
    )
    base_sigma_ba_mgal = sigma_ba_internal_to_mgal(baseline_cfg["fusion"]["noise"]["sigma_ba"])
    base_sigma_bg_degh = sigma_bg_internal_to_degh(baseline_cfg["fusion"]["noise"]["sigma_bg"])

    scaled_sigma_ba_mgal = float(base_sigma_ba_mgal) * float(scale)
    scaled_sigma_bg_degh = float(base_sigma_bg_degh) * float(scale)

    cfg, metadata = build_pure_run_config(
        base_cfg,
        output_dir,
        fix_gnss_lever_truth=True,
        sigma_ba_mgal_override=scaled_sigma_ba_mgal,
        sigma_bg_degh_override=scaled_sigma_bg_degh,
        phase3_gnss_on=PHASE3_GNSS_ON_DEFAULT,
        phase3_gnss_off=PHASE3_GNSS_OFF_DEFAULT,
    )
    output_dir_abs = normalize_repo_path(output_dir)
    cfg["fusion"]["output_path"] = rel_from_root(output_dir_abs / f"SOL_{case_id}.txt", REPO_ROOT)
    cfg["fusion"]["state_series_output_path"] = rel_from_root(
        output_dir_abs / f"state_series_{case_id}.csv",
        REPO_ROOT,
    )
    metadata["case_id"] = case_id
    metadata["scale"] = float(scale)
    metadata["base_sigma_ba_mgal"] = float(base_sigma_ba_mgal)
    metadata["base_sigma_bg_degh"] = float(base_sigma_bg_degh)
    metadata["scaled_sigma_ba_mgal"] = float(scaled_sigma_ba_mgal)
    metadata["scaled_sigma_bg_degh"] = float(scaled_sigma_bg_degh)
    return cfg, metadata


def phase3_row_from_df(phase_metrics_df: pd.DataFrame) -> pd.Series:
    return phase_metrics_df.loc[phase_metrics_df["window_name"] == PHASE3_WINDOW_NAME].iloc[0]


def run_case_spec(
    base_cfg: dict[str, Any],
    output_dir: Path,
    exe_path: Path,
    spec: CaseSpec,
    reference_metrics: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    case_dir = output_dir / "artifacts" / "cases" / spec.case_id
    ensure_dir(case_dir)
    cfg, metadata = build_case_config(base_cfg, output_dir, spec.scale, spec.case_id)
    cfg_path = case_dir / f"config_{spec.case_id}.yaml"
    save_yaml(cfg, cfg_path)

    case_row = run_pure_case(cfg_path, output_dir, case_dir, exe_path, spec.case_id, spec.case_label)
    sol_path = output_dir / f"SOL_{spec.case_id}.txt"
    state_series_path = output_dir / f"state_series_{spec.case_id}.csv"
    merged_df = merge_case_outputs(sol_path, state_series_path)
    truth_df = load_pos_dataframe((REPO_ROOT / cfg["fusion"]["pos_path"]).resolve())
    truth_interp_df = build_truth_interp(merged_df["timestamp"].to_numpy(dtype=float), truth_df)
    truth_reference = build_truth_reference(cfg)
    state_frame = build_state_frame(merged_df, truth_interp_df, truth_reference)

    all_states_path = case_dir / f"all_states_{spec.case_id}.csv"
    state_frame.to_csv(all_states_path, index=False, encoding="utf-8-sig")

    phase_windows = [
        ("phase1_ins_gnss", *metadata["phase1_window"]),
        ("phase2_ins_gnss", *metadata["phase2_window"]),
        (PHASE3_WINDOW_NAME, *metadata["phase3_window"]),
    ]
    phase_metrics_df = compute_phase_metrics(
        state_frame,
        spec.case_id,
        phase_windows,
        [tuple(window) for window in metadata["gnss_off_windows"]],
    )
    metrics_row = compute_case_metrics(case_row, state_frame)
    phase3_row = phase3_row_from_df(phase_metrics_df)

    result_row = {
        "case_id": spec.case_id,
        "case_label": spec.case_label,
        "scale": float(spec.scale),
        "config_path": rel_from_root(cfg_path, REPO_ROOT),
        "all_states_path": rel_from_root(all_states_path, REPO_ROOT),
        "all_states_mtime": mtime_text(all_states_path),
        "phase3_rmse_3d_m": float(phase3_row["rmse_3d_m"]),
        "phase3_p95_3d_m": float(phase3_row["p95_3d_m"]),
        "phase3_final_err_3d_m": float(phase3_row["final_err_3d_m"]),
        "overall_rmse3d_m": float(metrics_row["overall_rmse_3d_m_aux"]),
        "overall_p95_3d_m": float(metrics_row["overall_p95_3d_m_aux"]),
        "final_3d_m": float(metrics_row["overall_final_err_3d_m_aux"]),
        "reference_phase3_rmse_3d_m": float(reference_metrics["phase3_rmse_3d_m"]),
        "reference_phase3_final_err_3d_m": float(reference_metrics["phase3_final_err_3d_m"]),
        "reference_gap_m": float(phase3_row["rmse_3d_m"]) - float(reference_metrics["phase3_rmse_3d_m"]),
        "reference_gap_label": classify_reference_gap(
            float(phase3_row["rmse_3d_m"]),
            float(reference_metrics["phase3_rmse_3d_m"]),
        ),
        "phase3_gnss_on_s": metadata["phase3_gnss_on_s"],
        "phase3_gnss_off_s": metadata["phase3_gnss_off_s"],
        "gnss_off_window_count": len(metadata["gnss_off_windows"]),
        "base_sigma_ba_mgal": float(metadata["base_sigma_ba_mgal"]),
        "base_sigma_bg_degh": float(metadata["base_sigma_bg_degh"]),
        "scaled_sigma_ba_mgal": float(metadata["scaled_sigma_ba_mgal"]),
        "scaled_sigma_bg_degh": float(metadata["scaled_sigma_bg_degh"]),
        "process_sigma_ba": float(cfg["fusion"]["noise"]["sigma_ba"]),
        "process_sigma_bg": float(cfg["fusion"]["noise"]["sigma_bg"]),
        "sol_path": case_row["sol_path"],
        "state_series_path": case_row["state_series_path"],
        "diag_path": case_row["diag_path"],
        "stdout_path": case_row["stdout_path"],
        "config_mtime": case_row["config_mtime"],
        "sol_mtime": case_row["sol_mtime"],
        "state_series_mtime": case_row["state_series_mtime"],
        "diag_mtime": case_row["diag_mtime"],
        "stdout_mtime": case_row["stdout_mtime"],
    }
    return result_row, phase_metrics_df


def rank_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=["phase3_rmse_3d_m", "phase3_final_err_3d_m", "overall_rmse3d_m", "scale"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def write_summary(
    output_path: Path,
    manifest: dict[str, Any],
    ranked_df: pd.DataFrame,
    evaluated_scales: list[float],
    stop_reason: str,
) -> None:
    rows = [
        [
            str(row["case_id"]),
            format_metric(row["scale"]),
            format_metric(row["scaled_sigma_ba_mgal"]),
            format_metric(row["scaled_sigma_bg_degh"]),
            format_metric(row["phase3_rmse_3d_m"]),
            format_metric(row["phase3_final_err_3d_m"]),
            format_metric(row["overall_rmse3d_m"]),
            format_metric(row["reference_gap_m"]),
            str(row["reference_gap_label"]),
        ]
        for _, row in ranked_df.iterrows()
    ]
    best_row = ranked_df.iloc[0]
    lines = [
        "# data2 baseline pure ins gnss eskf outage 60on100off bias sweep summary",
        "",
        f"- exp_id: `{manifest['exp_id']}`",
        f"- base_config: `{manifest['base_config']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- historical_reference_dir: `{manifest['reference_output_dir']}`",
        "- contract: pure `INS/GNSS`, `ODO/NHC` disabled, GNSS lever fixed to truth",
        (
            f"- target schedule: `phase3 on={manifest['phase3_gnss_on_s']:.1f}s`, "
            f"`off={manifest['phase3_gnss_off_s']:.1f}s`"
        ),
        (
            f"- historical target: `phase3_rmse3d={manifest['reference_metrics']['phase3_rmse_3d_m']:.12f} m`, "
            f"`phase3_final_3d={manifest['reference_metrics']['phase3_final_err_3d_m']:.12f} m`"
        ),
        f"- evaluated_scales: `{evaluated_scales}`",
        f"- stop_reason: `{stop_reason}`",
        (
            f"- best_case: `{best_row['case_id']}` with `scale={best_row['scale']:g}x`, "
            f"`sigma_ba={best_row['scaled_sigma_ba_mgal']:.6f} mGal`, "
            f"`sigma_bg={best_row['scaled_sigma_bg_degh']:.6f} deg/h`, "
            f"`phase3_rmse3d={best_row['phase3_rmse_3d_m']:.12f} m`, "
            f"`phase3_final_3d={best_row['phase3_final_err_3d_m']:.12f} m`, "
            f"`status={best_row['reference_gap_label']}`"
        ),
        "",
        "## Ranking",
    ]
    lines.extend(
        render_table(
            [
                "case_id",
                "scale",
                "sigma_ba_mgal",
                "sigma_bg_degh",
                "phase3_rmse3d_m",
                "phase3_final_3d_m",
                "overall_rmse3d_m",
                "gap_vs_ref_m",
                "status",
            ],
            rows,
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover the best pure INS/GNSS data2 ESKF outage 60on100off ba/bg noise scale."
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_DEFAULT)
    parser.add_argument("--exe", type=Path, default=SOLVER_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--reference-output-dir", type=Path, default=REFERENCE_OUTPUT_DIR_DEFAULT)
    parser.add_argument("--exp-id", default=EXP_ID_DEFAULT)
    parser.add_argument("--coarse-scales", default="0.5,1.0,2.0")
    parser.add_argument("--extend-lower-scales", default="0.25,0.125")
    parser.add_argument("--extend-upper-scales", default="4.0,8.0")
    parser.add_argument("--improvement-abs-threshold-m", type=float, default=PHASE3_IMPROVEMENT_ABS_THRESHOLD_M_DEFAULT)
    parser.add_argument("--improvement-rel-threshold", type=float, default=PHASE3_IMPROVEMENT_REL_THRESHOLD_DEFAULT)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    args.base_config = normalize_repo_path(args.base_config)
    args.exe = normalize_repo_path(args.exe)
    args.output_dir = normalize_repo_path(args.output_dir)
    args.reference_output_dir = normalize_repo_path(args.reference_output_dir)
    args.coarse_scales = parse_scale_list(args.coarse_scales)
    args.extend_lower_scales = parse_scale_list(args.extend_lower_scales)
    args.extend_upper_scales = parse_scale_list(args.extend_upper_scales)
    return args


def main() -> None:
    args = parse_args()
    if not args.base_config.exists():
        raise FileNotFoundError(f"missing base config: {args.base_config}")
    if not args.exe.exists():
        raise FileNotFoundError(f"missing solver executable: {args.exe}")
    if not args.reference_output_dir.exists():
        raise FileNotFoundError(f"missing historical reference output dir: {args.reference_output_dir}")

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "artifacts")
    ensure_dir(args.output_dir / "artifacts" / "cases")
    ensure_dir(args.output_dir / "artifacts" / "generated")

    reference_metrics = phase3_reference_metrics(args.reference_output_dir)
    base_cfg = load_yaml(args.base_config)

    evaluated_rows: list[dict[str, Any]] = []
    evaluated_phase_frames: list[pd.DataFrame] = []
    evaluated_scales: list[float] = []

    def evaluate_scale(scale: float) -> None:
        if scale in evaluated_scales:
            return
        if len(evaluated_scales) >= args.max_cases:
            raise RuntimeError(f"max cases reached before evaluating requested scale {scale:g}x")
        spec = CaseSpec(scale=float(scale))
        result_row, phase_metrics_df = run_case_spec(
            base_cfg=base_cfg,
            output_dir=args.output_dir,
            exe_path=args.exe,
            spec=spec,
            reference_metrics=reference_metrics,
        )
        evaluated_scales.append(float(scale))
        evaluated_rows.append(result_row)
        evaluated_phase_frames.append(phase_metrics_df)

    for scale in args.coarse_scales:
        evaluate_scale(scale)

    ranked_df = rank_cases(pd.DataFrame(evaluated_rows))
    best_scale = float(ranked_df.iloc[0]["scale"])

    if math.isclose(best_scale, min(args.coarse_scales), rel_tol=0.0, abs_tol=1.0e-12):
        followup_scales = [float(value) for value in args.extend_lower_scales]
    elif math.isclose(best_scale, max(args.coarse_scales), rel_tol=0.0, abs_tol=1.0e-12):
        followup_scales = [float(value) for value in args.extend_upper_scales]
    else:
        followup_scales = build_followup_scale_candidates(best_scale)

    stop_reason = "coarse_stage_only"
    current_best_rmse = float(ranked_df.iloc[0]["phase3_rmse_3d_m"])
    if math.isclose(best_scale, min(args.coarse_scales), rel_tol=0.0, abs_tol=1.0e-12) or math.isclose(
        best_scale,
        max(args.coarse_scales),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        for scale in followup_scales:
            evaluate_scale(scale)
            ranked_df = rank_cases(pd.DataFrame(evaluated_rows))
            candidate_row = ranked_df.loc[ranked_df["scale"] == float(scale)].iloc[0]
            candidate_rmse = float(candidate_row["phase3_rmse_3d_m"])
            if is_material_improvement(
                current_best_rmse=current_best_rmse,
                candidate_rmse=candidate_rmse,
                abs_threshold_m=args.improvement_abs_threshold_m,
                rel_threshold=args.improvement_rel_threshold,
            ):
                current_best_rmse = candidate_rmse
                stop_reason = f"boundary_extension_improved_at_{scale:g}x"
                continue
            stop_reason = f"boundary_extension_stopped_after_{scale:g}x_non_material_gain"
            break
    else:
        for scale in followup_scales:
            evaluate_scale(scale)
        ranked_df = rank_cases(pd.DataFrame(evaluated_rows))
        stop_reason = "interior_refinement_completed"

    ranked_df = rank_cases(pd.DataFrame(evaluated_rows))
    phase_metrics_df = pd.concat(evaluated_phase_frames, ignore_index=True)
    ranking_path = args.output_dir / "phase3_ranking.csv"
    case_metrics_path = args.output_dir / "sweep_results.csv"
    phase_metrics_path = args.output_dir / "phase_metrics.csv"
    ranked_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(evaluated_rows).to_csv(case_metrics_path, index=False, encoding="utf-8-sig")
    phase_metrics_df.to_csv(phase_metrics_path, index=False, encoding="utf-8-sig")

    best_row = ranked_df.iloc[0]
    best_cfg_src = REPO_ROOT / str(best_row["config_path"])
    best_cfg_dst = args.output_dir / "artifacts" / "generated" / "config_data2_baseline_ins_gnss_eskf_outage_60on100off_best_recovered.yaml"
    shutil.copy2(best_cfg_src, best_cfg_dst)

    summary_path = args.output_dir / "summary.md"
    manifest = {
        "exp_id": args.exp_id,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "base_config": rel_from_root(args.base_config, REPO_ROOT),
        "output_dir": rel_from_root(args.output_dir, REPO_ROOT),
        "reference_output_dir": rel_from_root(args.reference_output_dir, REPO_ROOT),
        "reference_metrics": reference_metrics,
        "phase3_gnss_on_s": PHASE3_GNSS_ON_DEFAULT,
        "phase3_gnss_off_s": PHASE3_GNSS_OFF_DEFAULT,
        "evaluated_scales": evaluated_scales,
        "ranking_path": rel_from_root(ranking_path, REPO_ROOT),
        "sweep_results_path": rel_from_root(case_metrics_path, REPO_ROOT),
        "phase_metrics_path": rel_from_root(phase_metrics_path, REPO_ROOT),
        "best_config_path": rel_from_root(best_cfg_dst, REPO_ROOT),
        "best_case_id": str(best_row["case_id"]),
        "best_scale": float(best_row["scale"]),
        "best_phase3_rmse_3d_m": float(best_row["phase3_rmse_3d_m"]),
        "best_phase3_final_err_3d_m": float(best_row["phase3_final_err_3d_m"]),
        "best_reference_gap_label": str(best_row["reference_gap_label"]),
        "stop_reason": stop_reason,
        "freshness": {
            "base_config_mtime": mtime_text(args.base_config),
            "solver_mtime": mtime_text(args.exe),
            "reference_phase_metrics_mtime": mtime_text(args.reference_output_dir / "phase_metrics.csv"),
            "reference_case_metrics_mtime": mtime_text(args.reference_output_dir / "case_metrics.csv"),
            "ranking_mtime": mtime_text(ranking_path),
            "sweep_results_mtime": mtime_text(case_metrics_path),
            "phase_metrics_mtime": mtime_text(phase_metrics_path),
            "best_config_mtime": mtime_text(best_cfg_dst),
        },
    }
    write_summary(summary_path, manifest, ranked_df, evaluated_scales, stop_reason)
    manifest["freshness"]["summary_md_mtime"] = mtime_text(summary_path)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(rel_from_root(manifest_path, REPO_ROOT))


if __name__ == "__main__":
    main()
