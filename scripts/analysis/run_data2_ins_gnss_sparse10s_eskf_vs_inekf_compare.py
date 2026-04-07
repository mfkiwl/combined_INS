from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.generate_sparse_compare_plots import (  # noqa: E402
    GROUPS,
    PLOT_FILENAMES,
    STATE_FIELDS,
    load_key_fields,
    merge_states,
    plot_group,
    plot_overview,
    read_current_states,
    read_kf_states,
    thin_for_plot,
)
from scripts.analysis.odo_nhc_update_sweep import ensure_dir, load_yaml, rel_from_root, save_yaml  # noqa: E402
from scripts.analysis.run_data2_baseline_ins_gnss_outage_no_odo_nhc import run_case  # noqa: E402
from scripts.analysis.run_data2_fullwindow_attitude_bias_coupling import (  # noqa: E402
    build_state_frame,
    compute_case_metrics,
)
from scripts.analysis.run_data2_ins_gnss_odo_nhc_pva_anchor_compare import mtime_text  # noqa: E402
from scripts.analysis.run_data2_state_sanity_matrix import build_truth_reference, json_safe  # noqa: E402
from scripts.analysis.run_nhc_state_convergence_research import (  # noqa: E402
    build_truth_interp,
    load_pos_dataframe,
    merge_case_outputs,
    wrap_deg,
)


EXP_ID_DEFAULT = "EXP-20260407-data2-ins-gnss-sparse10s-eskf-vs-inekf-r2"
OUTPUT_DIR_DEFAULT = Path("output/data2_ins_gnss_sparse10s_eskf_vs_inekf_r2")
BASE_CONFIG_DEFAULT = Path("config_data2_baseline_ins_gnss_eskf_outage_60on100off_best.yaml")
SOLVER_DEFAULT = Path("build/Release/eskf_fusion.exe")
GNSS_SOURCE_DEFAULT = Path("dataset/data2/rtk.txt")


@dataclass(frozen=True)
class CaseSpec:
    system: str
    case_id: str
    case_label: str
    inekf_enable: bool


CASE_SPECS = (
    CaseSpec("eskf", "data2_ins_gnss_sparse10s_eskf", "ESKF sparse 10s GNSS", False),
    CaseSpec("inekf", "data2_ins_gnss_sparse10s_inekf", "InEKF sparse 10s GNSS", True),
)

PLOT_CURRENT_LABEL = "ESKF"
PLOT_KF_LABEL = "InEKF"
PLOT_TRUTH_LABEL = "Truth"

STATE_LABELS = {
    "p_n_m": "p_n",
    "p_e_m": "p_e",
    "p_u_m": "p_u",
    "v_n_mps": "v_n",
    "v_e_mps": "v_e",
    "v_u_mps": "v_u",
    "roll_deg": "roll",
    "pitch_deg": "pitch",
    "yaw_deg": "yaw",
    "ba_x_mgal": "ba_x",
    "ba_y_mgal": "ba_y",
    "ba_z_mgal": "ba_z",
    "bg_x_degh": "bg_x",
    "bg_y_degh": "bg_y",
    "bg_z_degh": "bg_z",
    "sg_x_ppm": "sg_x",
    "sg_y_ppm": "sg_y",
    "sg_z_ppm": "sg_z",
    "sa_x_ppm": "sa_x",
    "sa_y_ppm": "sa_y",
    "sa_z_ppm": "sa_z",
}


def normalize_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data2 pure INS/GNSS sparse-10s ESKF-vs-InEKF comparison with shared canonical noise."
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_DEFAULT)
    parser.add_argument("--gnss-source", type=Path, default=GNSS_SOURCE_DEFAULT)
    parser.add_argument("--exe", type=Path, default=SOLVER_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--exp-id", default=EXP_ID_DEFAULT)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--min-plot-dt", type=float, default=0.1)
    args = parser.parse_args()
    args.base_config = normalize_repo_path(args.base_config)
    args.gnss_source = normalize_repo_path(args.gnss_source)
    args.exe = normalize_repo_path(args.exe)
    args.output_dir = normalize_repo_path(args.output_dir)
    args.artifacts_dir = args.output_dir / "artifacts"
    args.generated_dir = args.artifacts_dir / "generated"
    args.plot_dir = args.output_dir / "plots"
    return args


def write_sparse_gnss_file(
    source_path: Path,
    output_path: Path,
    *,
    start_time: float,
    final_time: float,
    stride_seconds: float,
) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    rows_raw = 0
    rows_window = 0
    rows_kept = 0
    last_kept_time: float | None = None
    kept_lines: list[str] = []
    for raw in source_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows_raw += 1
        timestamp = float(parts[0])
        if timestamp < start_time or timestamp > final_time:
            continue
        rows_window += 1
        if last_kept_time is None or timestamp >= last_kept_time + stride_seconds - 1.0e-9:
            kept_lines.append(line)
            last_kept_time = timestamp
            rows_kept += 1
    output_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return {
        "rows_raw": rows_raw,
        "rows_window": rows_window,
        "rows_kept": rows_kept,
        "ratio_kept_in_window": float(rows_kept / rows_window) if rows_window else 0.0,
        "start_time": float(start_time),
        "end_time": float(final_time),
        "stride_seconds": float(stride_seconds),
    }


def build_case_config(base_cfg: dict[str, Any], *, case_spec: CaseSpec, case_output_dir: Path, sparse_gnss_path: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    fusion = cfg.setdefault("fusion", {})
    constraints = fusion.setdefault("constraints", {})
    ablation = fusion.setdefault("ablation", {})
    start_time = float(fusion["starttime"])
    final_time = float(fusion["finaltime"])
    fusion["gnss_path"] = rel_from_root(sparse_gnss_path, REPO_ROOT)
    fusion["output_path"] = rel_from_root(case_output_dir / f"SOL_{case_spec.case_id}.txt", REPO_ROOT)
    fusion["state_series_output_path"] = rel_from_root(case_output_dir / f"state_series_{case_spec.case_id}.csv", REPO_ROOT)
    fusion["gnss_update_debug_output_path"] = rel_from_root(
        case_output_dir / "artifacts" / "cases" / case_spec.case_id / f"gnss_updates_{case_spec.case_id}.csv",
        REPO_ROOT,
    )
    fusion["enable_gnss_velocity"] = False
    fusion["runtime_phases"] = []
    fusion["gnss_schedule"] = {"enabled": True, "enabled_windows": [{"start_time": start_time, "end_time": final_time}]}
    fusion.pop("post_gnss_ablation", None)
    fusion.pop("fej", None)
    fusion.pop("filter", None)
    fusion["inekf"] = {"enable": bool(case_spec.inekf_enable)}
    constraints["enable_odo"] = False
    constraints["enable_nhc"] = False
    constraints["enable_zupt"] = False
    constraints["enable_diagnostics"] = True
    constraints["enable_consistency_log"] = True
    constraints["enable_mechanism_log"] = False
    ablation["disable_odo_scale"] = True
    ablation["disable_mounting"] = True
    ablation["disable_odo_lever_arm"] = True
    ablation["disable_gnss_lever_arm"] = True
    ablation["disable_gnss_lever_z"] = False
    return cfg


def run_solver_case(cfg: dict[str, Any], *, case_spec: CaseSpec, case_output_dir: Path, exe_path: Path) -> dict[str, Any]:
    case_dir = case_output_dir / "artifacts" / "cases" / case_spec.case_id
    ensure_dir(case_dir)
    cfg_path = case_dir / f"config_{case_spec.case_id}.yaml"
    save_yaml(cfg, cfg_path)
    truth_reference = build_truth_reference(cfg)
    case_row = run_case(cfg_path, case_output_dir, case_dir, exe_path, case_spec.case_id, case_spec.case_label)
    sol_path = case_output_dir / f"SOL_{case_spec.case_id}.txt"
    state_series_path = case_output_dir / f"state_series_{case_spec.case_id}.csv"
    merged_df = merge_case_outputs(sol_path, state_series_path)
    truth_df = load_pos_dataframe((REPO_ROOT / cfg["fusion"]["pos_path"]).resolve())
    truth_interp_df = build_truth_interp(merged_df["timestamp"].to_numpy(dtype=float), truth_df)
    state_frame = build_state_frame(merged_df, truth_interp_df, truth_reference)
    all_states_path = case_dir / f"all_states_{case_spec.case_id}.csv"
    state_frame.to_csv(all_states_path, index=False, encoding="utf-8-sig")
    metrics_row = compute_case_metrics(case_row, state_frame)
    metrics_row["system"] = case_spec.system
    metrics_row["all_states_path"] = rel_from_root(all_states_path, REPO_ROOT)
    metrics_row["all_states_mtime"] = mtime_text(all_states_path)
    case_metrics_df = pd.DataFrame([metrics_row])
    case_metrics_df.to_csv(case_output_dir / "case_metrics.csv", index=False, encoding="utf-8-sig")
    return {
        "spec": case_spec,
        "config_path": cfg_path,
        "sol_path": sol_path,
        "state_series_path": state_series_path,
        "all_states_path": all_states_path,
        "case_metrics_df": case_metrics_df,
        "truth_reference": truth_reference,
    }


def build_state_delta_summary(eskf_all_states_path: Path, inekf_all_states_path: Path, artifacts_dir: Path) -> tuple[pd.DataFrame, Path]:
    ensure_dir(artifacts_dir)
    eskf_shared = read_current_states(eskf_all_states_path)
    inekf_shared = read_current_states(inekf_all_states_path)
    eskf_shared.to_csv(artifacts_dir / "eskf_shared_states.csv", index=False, encoding="utf-8-sig")
    inekf_shared.to_csv(artifacts_dir / "inekf_shared_states.csv", index=False, encoding="utf-8-sig")
    merged = merge_states(eskf_shared, read_kf_states(inekf_all_states_path))
    unit_map = {
        "p_n_m": "m", "p_e_m": "m", "p_u_m": "m",
        "v_n_mps": "m/s", "v_e_mps": "m/s", "v_u_mps": "m/s",
        "roll_deg": "deg", "pitch_deg": "deg", "yaw_deg": "deg",
        "ba_x_mgal": "mGal", "ba_y_mgal": "mGal", "ba_z_mgal": "mGal",
        "bg_x_degh": "deg/h", "bg_y_degh": "deg/h", "bg_z_degh": "deg/h",
        "sg_x_ppm": "ppm", "sg_y_ppm": "ppm", "sg_z_ppm": "ppm",
        "sa_x_ppm": "ppm", "sa_y_ppm": "ppm", "sa_z_ppm": "ppm",
    }
    rows: list[dict[str, Any]] = []
    for field in STATE_FIELDS:
        delta = merged[f"{field}_kf"].to_numpy(dtype=float) - merged[f"{field}_current"].to_numpy(dtype=float)
        if field in {"roll_deg", "pitch_deg", "yaw_deg"}:
            delta = wrap_deg(delta)
        rows.append(
            {
                "state_key": field,
                "label": STATE_LABELS[field],
                "unit": unit_map[field],
                "mean_abs_delta_full": float(np.mean(np.abs(delta))),
                "p95_abs_delta_full": float(np.percentile(np.abs(delta), 95.0)),
                "max_abs_delta_full": float(np.max(np.abs(delta))),
                "final_delta_full_current_minus_kf": float(
                    merged[f"{field}_current"].to_numpy(dtype=float)[-1] - merged[f"{field}_kf"].to_numpy(dtype=float)[-1]
                ),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        by=["mean_abs_delta_full", "p95_abs_delta_full", "max_abs_delta_full"],
        ascending=False,
    )
    summary_path = artifacts_dir / "state_delta_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df, summary_path


def generate_plots(run_dir: Path, state_delta_path: Path, min_plot_dt: float) -> dict[str, str]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    eskf_path = (REPO_ROOT / Path(manifest["current_case"]["all_states_path"])).resolve()
    inekf_path = (REPO_ROOT / Path(manifest["kf_case"]["all_states_path"])).resolve()
    eskf_df = thin_for_plot(read_current_states(eskf_path), min_plot_dt)
    inekf_df = thin_for_plot(read_kf_states(inekf_path), min_plot_dt)
    merged = merge_states(eskf_df, inekf_df)
    x = merged["timestamp"].to_numpy(dtype=np.float64)
    x = x - x[0]
    key_fields = load_key_fields(state_delta_path)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(
        merged,
        x,
        STATE_FIELDS,
        plots_dir / PLOT_FILENAMES["all_shared_states_overview"],
        "All Shared States Overview: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_overview(
        merged,
        x,
        key_fields,
        plots_dir / PLOT_FILENAMES["key_shared_states"],
        "Key Shared States: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["position_state"],
        plots_dir / PLOT_FILENAMES["position_state"],
        "Position State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["velocity_state"],
        plots_dir / PLOT_FILENAMES["velocity_state"],
        "Velocity State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["attitude_state"],
        plots_dir / PLOT_FILENAMES["attitude_state"],
        "Attitude State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["ba_state"],
        plots_dir / PLOT_FILENAMES["ba_state"],
        "Accelerometer Bias State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["bg_state"],
        plots_dir / PLOT_FILENAMES["bg_state"],
        "Gyro Bias State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["sg_state"],
        plots_dir / PLOT_FILENAMES["sg_state"],
        "Gyro Scale State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["sa_state"],
        plots_dir / PLOT_FILENAMES["sa_state"],
        "Accelerometer Scale State: ESKF vs InEKF",
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["position_error"],
        plots_dir / PLOT_FILENAMES["position_error"],
        "Position Error: ESKF vs InEKF",
        error_mode=True,
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["velocity_error"],
        plots_dir / PLOT_FILENAMES["velocity_error"],
        "Velocity Error: ESKF vs InEKF",
        error_mode=True,
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    plot_group(
        merged,
        x,
        GROUPS["attitude_error"],
        plots_dir / PLOT_FILENAMES["attitude_error"],
        "Attitude Error: ESKF vs InEKF",
        error_mode=True,
        current_label=PLOT_CURRENT_LABEL,
        kf_label=PLOT_KF_LABEL,
        truth_label=PLOT_TRUTH_LABEL,
    )
    return {name: rel_from_root(run_dir / "plots" / filename, REPO_ROOT) for name, filename in PLOT_FILENAMES.items()}


def format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return str(value)


def render_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_summary(path: Path, *, exp_id: str, output_dir: Path, stats: dict[str, Any], comparison_df: pd.DataFrame, case_metrics_df: pd.DataFrame, state_delta_df: pd.DataFrame, plot_paths: dict[str, str]) -> None:
    headline_rows = [
        [
            str(row["system"]),
            str(row["case_id"]),
            format_metric(row["overall_rmse_3d_m_aux"]),
            format_metric(row["overall_p95_3d_m_aux"]),
            format_metric(row["overall_final_err_3d_m_aux"]),
            format_metric(row["yaw_err_max_abs_deg"]),
            format_metric(row["bg_z_degh_err_max_abs"]),
        ]
        for _, row in case_metrics_df.sort_values("system").iterrows()
    ]
    full_rows = [
        [
            str(row["window_name"]),
            format_metric(row["rmse_3d_m_eskf"]),
            format_metric(row["rmse_3d_m_inekf"]),
            format_metric(row["rmse_3d_m_delta_inekf_minus_eskf"]),
            format_metric(row["final_err_3d_m_eskf"]),
            format_metric(row["final_err_3d_m_inekf"]),
            format_metric(row["final_err_3d_m_delta_inekf_minus_eskf"]),
        ]
        for _, row in comparison_df.iterrows()
    ]
    delta_rows = [
        [
            str(row["label"]),
            str(row["unit"]),
            format_metric(row["mean_abs_delta_full"]),
            format_metric(row["p95_abs_delta_full"]),
            format_metric(row["max_abs_delta_full"]),
            format_metric(-float(row["final_delta_full_current_minus_kf"])),
        ]
        for _, row in state_delta_df.head(10).iterrows()
    ]
    lines = [
        "# data2 sparse-10s INS/GNSS ESKF-vs-InEKF comparison",
        "",
        f"- exp_id: `{exp_id}`",
        "- comparison contract: `full-window INS/GNSS only`, `ODO/NHC/UWB off`, `enable_gnss_velocity=false`, fixed true `gnss_lever=[0.15,-0.22,-1.15] m`.",
        f"- sparse_gnss_stats: `rows_window={int(stats['rows_window'])}`, `rows_kept={int(stats['rows_kept'])}`, `stride={float(stats['stride_seconds']):.1f}s`.",
        f"- output_dir: `{rel_from_root(output_dir, REPO_ROOT)}`",
        f"- generated_at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Headline Metrics",
        *render_table(["system", "case_id", "rmse3d_m", "p95_3d_m", "final_3d_m", "yaw_err_max_abs_deg", "bg_z_err_max_abs_degh"], headline_rows),
        "",
        "## Full-Window Metrics",
        *render_table(["window", "rmse3d_eskf", "rmse3d_inekf", "delta_rmse3d", "final3d_eskf", "final3d_inekf", "delta_final3d"], full_rows),
        "",
        "## Largest Shared-State Deltas",
        *render_table(["state", "unit", "mean_abs_delta_full", "p95_abs_delta_full", "max_abs_delta_full", "final_delta_inekf_minus_eskf"], delta_rows),
        "",
        "## Plot Outputs",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in plot_paths.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.base_config.exists():
        raise FileNotFoundError(f"missing base config: {args.base_config}")
    if not args.gnss_source.exists():
        raise FileNotFoundError(f"missing GNSS source: {args.gnss_source}")
    if not args.exe.exists():
        raise FileNotFoundError(f"missing solver executable: {args.exe}")
    reset_directory(args.output_dir)
    ensure_dir(args.generated_dir)
    base_cfg = load_yaml(args.base_config)
    start_time = float(base_cfg["fusion"]["starttime"])
    final_time = float(base_cfg["fusion"]["finaltime"])
    sparse_gnss_path = args.generated_dir / "rtk_full_window_sparse_10s.txt"
    gnss_filter_stats = write_sparse_gnss_file(args.gnss_source, sparse_gnss_path, start_time=start_time, final_time=final_time, stride_seconds=float(args.stride_seconds))
    results: dict[str, dict[str, Any]] = {}
    for case_spec in CASE_SPECS:
        case_output_dir = args.output_dir / case_spec.system
        ensure_dir(case_output_dir)
        cfg = build_case_config(base_cfg, case_spec=case_spec, case_output_dir=case_output_dir, sparse_gnss_path=sparse_gnss_path)
        results[case_spec.system] = run_solver_case(cfg, case_spec=case_spec, case_output_dir=case_output_dir, exe_path=args.exe)
    truth_reference_path = args.output_dir / "truth_reference.json"
    truth_reference_path.write_text(json.dumps(json_safe(results["eskf"]["truth_reference"]), ensure_ascii=False, indent=2), encoding="utf-8")
    state_delta_df, state_delta_path = build_state_delta_summary(results["eskf"]["all_states_path"], results["inekf"]["all_states_path"], args.artifacts_dir)
    eskf_row = results["eskf"]["case_metrics_df"].iloc[0]
    inekf_row = results["inekf"]["case_metrics_df"].iloc[0]
    comparison_df = pd.DataFrame([{
        "window_name": "full_window_sparse10s",
        "rmse_3d_m_eskf": float(eskf_row["overall_rmse_3d_m_aux"]),
        "rmse_3d_m_inekf": float(inekf_row["overall_rmse_3d_m_aux"]),
        "rmse_3d_m_delta_inekf_minus_eskf": float(inekf_row["overall_rmse_3d_m_aux"] - eskf_row["overall_rmse_3d_m_aux"]),
        "final_err_3d_m_eskf": float(eskf_row["overall_final_err_3d_m_aux"]),
        "final_err_3d_m_inekf": float(inekf_row["overall_final_err_3d_m_aux"]),
        "final_err_3d_m_delta_inekf_minus_eskf": float(inekf_row["overall_final_err_3d_m_aux"] - eskf_row["overall_final_err_3d_m_aux"]),
    }])
    comparison_phase_metrics_path = args.output_dir / "comparison_phase_metrics.csv"
    comparison_df.to_csv(comparison_phase_metrics_path, index=False, encoding="utf-8-sig")
    manifest = {
        "exp_id": args.exp_id,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_dir": rel_from_root(args.output_dir, REPO_ROOT),
        "solver": rel_from_root(args.exe, REPO_ROOT),
        "base_config": rel_from_root(args.base_config, REPO_ROOT),
        "eskf_config_base": rel_from_root(args.base_config, REPO_ROOT),
        "inekf_config_base": rel_from_root(args.base_config, REPO_ROOT),
        "gnss_source": rel_from_root(args.gnss_source, REPO_ROOT),
        "filtered_gnss_path": rel_from_root(sparse_gnss_path, REPO_ROOT),
        "gnss_filter_stats": gnss_filter_stats,
        "truth_reference_path": rel_from_root(truth_reference_path, REPO_ROOT),
        "comparison_phase_metrics_path": rel_from_root(comparison_phase_metrics_path, REPO_ROOT),
        "state_delta_summary_path": rel_from_root(state_delta_path, REPO_ROOT),
        "summary_path": rel_from_root(args.output_dir / "summary.md", REPO_ROOT),
        "current_case": {"case_id": results["eskf"]["spec"].case_id, "all_states_path": rel_from_root(results["eskf"]["all_states_path"], REPO_ROOT)},
        "kf_case": {"case_id": results["inekf"]["spec"].case_id, "all_states_path": rel_from_root(results["inekf"]["all_states_path"], REPO_ROOT)},
        "artifacts": {
            "state_delta_summary": rel_from_root(state_delta_path, REPO_ROOT),
            "eskf_shared_states": rel_from_root(args.artifacts_dir / "eskf_shared_states.csv", REPO_ROOT),
            "inekf_shared_states": rel_from_root(args.artifacts_dir / "inekf_shared_states.csv", REPO_ROOT),
        },
        "cases": {
            name: {
                "case_id": result["spec"].case_id,
                "config_path": rel_from_root(result["config_path"], REPO_ROOT),
                "sol_path": rel_from_root(result["sol_path"], REPO_ROOT),
                "state_series_path": rel_from_root(result["state_series_path"], REPO_ROOT),
                "all_states_path": rel_from_root(result["all_states_path"], REPO_ROOT),
                "gnss_update_debug_path": rel_from_root(result["config_path"].parent / f"gnss_updates_{result['spec'].case_id}.csv", REPO_ROOT),
            }
            for name, result in results.items()
        },
        "freshness": {
            "solver_mtime": mtime_text(args.exe),
            "filtered_gnss_mtime": mtime_text(sparse_gnss_path),
            "eskf_case_config_mtime": mtime_text(results["eskf"]["config_path"]),
            "inekf_case_config_mtime": mtime_text(results["inekf"]["config_path"]),
            "truth_reference_mtime": mtime_text(truth_reference_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_paths = generate_plots(args.output_dir, state_delta_path, float(args.min_plot_dt))
    combined_case_metrics_df = pd.concat([results["eskf"]["case_metrics_df"], results["inekf"]["case_metrics_df"]], ignore_index=True)
    summary_path = args.output_dir / "summary.md"
    write_summary(summary_path, exp_id=args.exp_id, output_dir=args.output_dir, stats=gnss_filter_stats, comparison_df=comparison_df, case_metrics_df=combined_case_metrics_df, state_delta_df=state_delta_df, plot_paths=plot_paths)
    manifest["plot_paths"] = plot_paths
    manifest["freshness"]["plots_generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["freshness"]["summary_md_mtime"] = mtime_text(summary_path)
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
