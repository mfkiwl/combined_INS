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

from scripts.analysis.odo_nhc_update_sweep import (
    ensure_dir,
    load_yaml,
    parse_consistency_summary,
    rel_from_root,
    save_yaml,
)
from scripts.analysis.run_data2_fullwindow_attitude_bias_coupling import build_state_frame
from scripts.analysis.run_data2_ins_gnss_odo_nhc_pva_anchor_compare import mtime_text
from scripts.analysis.run_data2_staged_g5_no_imu_scale import (
    KEY_COUPLING_STATES,
    PVA_ERROR_GROUP_SPECS,
    build_mainline_plot_config,
    plot_state_grid,
    remove_obsolete_mainline_plot_files,
)
from scripts.analysis.run_data2_state_sanity_matrix import (
    build_truth_reference,
    downsample_for_plot,
    evaluate_navigation_metrics,
    json_safe,
    reset_directory,
    run_command,
)
from scripts.analysis.run_nhc_state_convergence_research import (
    build_motion_frame,
    build_plot_frame,
    build_truth_interp,
    load_imu_dataframe,
    load_pos_dataframe,
    merge_case_outputs,
)


EXP_ID_DEFAULT = "EXP-20260408-data2-ins-gnss-odo-nhc-staged-estimation-r1"
OUTPUT_DIR_DEFAULT = Path("output/data2_ins_gnss_odo_nhc_staged_estimation_r1")
BASE_CONFIG_DEFAULT = Path("config_data2_baseline_ins_gnss_outage_best.yaml")
SOLVER_DEFAULT = Path("build/Release/eskf_fusion.exe")
CASE_ID = "data2_ins_gnss_odo_nhc_staged_estimation_eskf"
FILTER_MODE_ESKF = "ESKF"
FILTER_MODE_INEKF = "InEKF"
FILTER_MODE_CHOICES = (FILTER_MODE_ESKF, FILTER_MODE_INEKF)

PHASE1_END_OFFSET_DEFAULT = 200.0
PHASE2_END_OFFSET_DEFAULT = 700.0
PHASE1_GNSS_LEVER_STD_DEFAULT = 1.0
PHASE1_GNSS_LEVER_SIGMA_DEFAULT = 1.0e-6
PHASE2_ODO_SCALE_STD_DEFAULT = 0.2
PHASE2_MOUNTING_STD_DEG_DEFAULT = 10.0
PHASE2_ODO_LEVER_STD_DEFAULT = 1.0
PHASE3_GNSS_ON_DURATION_DEFAULT = 60.0
PHASE3_GNSS_OFF_DURATION_DEFAULT = 60.0
PHASE_LAYOUT_DEFAULT = "staged"
PHASE_LAYOUT_JOINT300 = "joint300"
PHASE_LAYOUT_CHOICES = (PHASE_LAYOUT_DEFAULT, PHASE_LAYOUT_JOINT300)
JOINT_ESTIMATION_DURATION_DEFAULT = 300.0
FREEZE_GNSS_LEVER_AFTER_JOINT_DEFAULT = True
TRUTH_FIX_MODE_NONE = "none"
TRUTH_FIX_MODE_FIX_ODO_LEVER_FULLRUN = "fix_odo_lever_truth_fullrun"
TRUTH_FIX_MODE_FIX_MOUNTING_FULLRUN = "fix_mounting_truth_fullrun"
TRUTH_FIX_MODE_CHOICES = (
    TRUTH_FIX_MODE_NONE,
    TRUTH_FIX_MODE_FIX_ODO_LEVER_FULLRUN,
    TRUTH_FIX_MODE_FIX_MOUNTING_FULLRUN,
)
MAINLINE_PLOT_KEYS = (
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
CALIBRATION_DELTA_COLUMNS = {
    "mounting_roll_deg": "truth_mounting_roll_deg",
    "mounting_pitch_deg": "truth_mounting_pitch_deg",
    "mounting_yaw_deg": "truth_mounting_yaw_deg",
    "odo_lever_x_m": "truth_odo_lever_x_m",
    "odo_lever_y_m": "truth_odo_lever_y_m",
    "odo_lever_z_m": "truth_odo_lever_z_m",
    "gnss_lever_x_m": "truth_gnss_lever_x_m",
    "gnss_lever_y_m": "truth_gnss_lever_y_m",
    "gnss_lever_z_m": "truth_gnss_lever_z_m",
}
ABSOLUTE_CALIBRATION_COLUMNS = {
    "odo_scale_state": "truth_odo_scale_state",
}
CALIBRATION_TRUTH_VISIBLE_KEYS_BY_GROUP = {
    "odo_scale": {"odo_scale_state"},
    "mounting": {"mounting_roll_deg", "mounting_pitch_deg", "mounting_yaw_deg"},
    "odo_lever": {"odo_lever_x_m", "odo_lever_y_m", "odo_lever_z_m"},
    "gnss_lever": {"gnss_lever_x_m", "gnss_lever_y_m", "gnss_lever_z_m"},
}


@dataclass(frozen=True)
class PlotColumn:
    key: str
    label: str
    unit: str


@dataclass(frozen=True)
class PlotCase:
    case_id: str
    label: str
    color: str


def build_staged_truth_keys_to_hide(
    base_keys: set[str] | None = None,
    *,
    visible_keys: set[str] | None = None,
) -> set[str]:
    hidden = set() if base_keys is None else set(base_keys)
    hidden.update(CALIBRATION_DELTA_COLUMNS.keys())
    hidden.update(ABSOLUTE_CALIBRATION_COLUMNS.keys())
    if visible_keys is not None:
        hidden.difference_update(visible_keys)
    return hidden


def build_all_states_export_frame(state_frame: pd.DataFrame) -> pd.DataFrame:
    return state_frame.copy()


NAV_ERROR_COLUMNS: tuple[PlotColumn, ...] = (
    PlotColumn("p_n_err_m", "p_n error", "m"),
    PlotColumn("p_e_err_m", "p_e error", "m"),
    PlotColumn("p_u_err_m", "p_u error", "m"),
    PlotColumn("v_n_err_mps", "v_n error", "m/s"),
    PlotColumn("v_e_err_mps", "v_e error", "m/s"),
    PlotColumn("v_u_err_mps", "v_u error", "m/s"),
    PlotColumn("roll_err_deg", "roll error", "deg"),
    PlotColumn("pitch_err_deg", "pitch error", "deg"),
    PlotColumn("yaw_err_deg", "yaw error", "deg"),
)

EXTRINSIC_ERROR_COLUMNS: tuple[PlotColumn, ...] = (
    PlotColumn("odo_scale_err", "odo_scale error", "1"),
    PlotColumn("mounting_pitch_err_deg", "mounting_pitch error", "deg"),
    PlotColumn("mounting_yaw_err_deg", "mounting_yaw error", "deg"),
    PlotColumn("odo_lever_x_err_m", "odo_lever_x error", "m"),
    PlotColumn("odo_lever_y_err_m", "odo_lever_y error", "m"),
    PlotColumn("odo_lever_z_err_m", "odo_lever_z error", "m"),
    PlotColumn("gnss_lever_x_err_m", "gnss_lever_x error", "m"),
    PlotColumn("gnss_lever_y_err_m", "gnss_lever_y error", "m"),
    PlotColumn("gnss_lever_z_err_m", "gnss_lever_z error", "m"),
)

IMU_STATE_COLUMNS: tuple[PlotColumn, ...] = (
    PlotColumn("ba_x_mgal", "ba_x", "mGal"),
    PlotColumn("ba_y_mgal", "ba_y", "mGal"),
    PlotColumn("ba_z_mgal", "ba_z", "mGal"),
    PlotColumn("bg_x_degh", "bg_x", "deg/h"),
    PlotColumn("bg_y_degh", "bg_y", "deg/h"),
    PlotColumn("bg_z_degh", "bg_z", "deg/h"),
    PlotColumn("sg_x_ppm", "sg_x", "ppm"),
    PlotColumn("sg_y_ppm", "sg_y", "ppm"),
    PlotColumn("sg_z_ppm", "sg_z", "ppm"),
    PlotColumn("sa_x_ppm", "sa_x", "ppm"),
    PlotColumn("sa_y_ppm", "sa_y", "ppm"),
    PlotColumn("sa_z_ppm", "sa_z", "ppm"),
)

KEY_STATE_COLUMNS: tuple[PlotColumn, ...] = (
    PlotColumn("yaw_err_deg", "yaw error", "deg"),
    PlotColumn("bg_z_degh_err", "bg_z error", "deg/h"),
    PlotColumn("odo_scale_err", "odo_scale error", "1"),
    PlotColumn("mounting_pitch_err_deg", "mounting_pitch error", "deg"),
    PlotColumn("mounting_yaw_err_deg", "mounting_yaw error", "deg"),
    PlotColumn("odo_lever_y_err_m", "odo_lever_y error", "m"),
    PlotColumn("gnss_lever_y_err_m", "gnss_lever_y error", "m"),
)

STATE_METRIC_COLUMNS: tuple[PlotColumn, ...] = KEY_STATE_COLUMNS + (
    PlotColumn("gnss_lever_x_err_m", "gnss_lever_x error", "m"),
    PlotColumn("gnss_lever_z_err_m", "gnss_lever_z error", "m"),
    PlotColumn("odo_lever_x_err_m", "odo_lever_x error", "m"),
    PlotColumn("odo_lever_z_err_m", "odo_lever_z error", "m"),
)


def normalize_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def case_id_for_filter_mode(filter_mode: str) -> str:
    suffix = "eskf" if filter_mode == FILTER_MODE_ESKF else "inekf"
    return f"data2_ins_gnss_odo_nhc_staged_estimation_{suffix}"


def case_label_for_filter_mode(filter_mode: str) -> str:
    return f"data2 ins gnss odo nhc staged estimation {filter_mode}"


def degph_to_radps(value_degh: float) -> float:
    return math.radians(float(value_degh) / 3600.0)


def build_base_bg_std_vector(p0_diag: list[float]) -> list[float]:
    return [math.sqrt(float(p0_diag[12])), math.sqrt(float(p0_diag[13])), math.sqrt(float(p0_diag[14]))]


def build_base_sigma_bg_vector(noise_cfg: dict[str, Any]) -> list[float]:
    if "sigma_bg_vec" in noise_cfg:
        return [float(x) for x in noise_cfg["sigma_bg_vec"]]
    sigma_bg = float(noise_cfg["sigma_bg"])
    return [sigma_bg, sigma_bg, sigma_bg]


def invert_enabled_windows(
    start_time: float,
    final_time: float,
    enabled_windows: list[list[float]],
) -> list[list[float]]:
    off_windows: list[list[float]] = []
    cursor = float(start_time)
    for win_start, win_end in enabled_windows:
        start = float(win_start)
        end = float(win_end)
        if start > cursor:
            off_windows.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < final_time:
        off_windows.append([cursor, float(final_time)])
    return off_windows


def build_periodic_enabled_windows(
    start_time: float,
    phase3_start_time: float,
    final_time: float,
    *,
    on_duration: float,
    off_duration: float,
) -> list[list[float]]:
    enabled_windows: list[list[float]] = []
    first_end_time = min(float(final_time), float(phase3_start_time + on_duration))
    enabled_windows.append([float(start_time), first_end_time])
    cursor = first_end_time + float(off_duration)
    while cursor < final_time:
        end_time = min(cursor + float(on_duration), float(final_time))
        enabled_windows.append([float(cursor), float(end_time)])
        cursor = end_time + float(off_duration)
    return enabled_windows


def build_phase_window_records(runtime_phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, phase in enumerate(runtime_phases, start=1):
        phase_key = f"phase{idx}"
        records.append(
            {
                "phase_key": phase_key,
                "window_name": phase_key,
                "runtime_name": str(phase["name"]),
                "start_time": float(phase["start_time"]),
                "end_time": float(phase["end_time"]),
            }
        )
    return records


def apply_truth_fix_mode(
    cfg: dict[str, Any],
    runtime_phases: list[dict[str, Any]],
    *,
    truth_fix_mode: str,
) -> dict[str, Any]:
    if truth_fix_mode == TRUTH_FIX_MODE_NONE:
        return {"truth_fix_mode": truth_fix_mode, "truth_initialized": [], "truth_frozen_blocks": []}

    truth_reference = build_truth_reference(cfg)
    fusion = cfg["fusion"]
    init_cfg = fusion["init"]
    constraints = fusion["constraints"]
    ablation = fusion["ablation"]
    truth_initialized: list[str] = []
    truth_frozen_blocks: list[str] = []

    if truth_fix_mode == TRUTH_FIX_MODE_FIX_ODO_LEVER_FULLRUN:
        odo_lever_truth = [float(x) for x in truth_reference["sources"]["odo_lever_truth"]["value_m"]]
        init_cfg["lever_arm0"] = odo_lever_truth
        constraints["odo_lever_arm"] = odo_lever_truth
        ablation["disable_odo_lever_arm"] = True
        for phase in runtime_phases:
            phase["ablation"]["disable_odo_lever_arm"] = True
        phase2_overrides = runtime_phases[1].get("phase_entry_std_overrides")
        if phase2_overrides is not None:
            phase2_overrides.pop("std_lever_arm", None)
        truth_initialized.append("lever_odo(25-27)")
        truth_frozen_blocks.append("lever_odo(25-27)")
    elif truth_fix_mode == TRUTH_FIX_MODE_FIX_MOUNTING_FULLRUN:
        init_cfg["mounting_roll0"] = float(truth_reference["states"]["mounting_roll"]["reference_value"])
        init_cfg["mounting_pitch0"] = float(truth_reference["states"]["mounting_pitch"]["reference_value"])
        init_cfg["mounting_yaw0"] = float(truth_reference["states"]["mounting_yaw"]["reference_value"])
        ablation["disable_mounting"] = True
        for phase in runtime_phases:
            phase["ablation"]["disable_mounting"] = True
        phase2_overrides = runtime_phases[1].get("phase_entry_std_overrides")
        if phase2_overrides is not None:
            phase2_overrides.pop("std_mounting_pitch", None)
            phase2_overrides.pop("std_mounting_yaw", None)
        truth_initialized.append("mounting(22-24)")
        truth_frozen_blocks.append("mounting(22-24)")
    else:
        raise ValueError(f"unsupported truth_fix_mode={truth_fix_mode!r}, expected one of {TRUTH_FIX_MODE_CHOICES}")

    return {
        "truth_fix_mode": truth_fix_mode,
        "truth_initialized": truth_initialized,
        "truth_frozen_blocks": truth_frozen_blocks,
    }


def build_case_config(
    base_cfg: dict[str, Any],
    output_dir: Path,
    *,
    filter_mode: str = FILTER_MODE_ESKF,
    phase1_end_offset: float = PHASE1_END_OFFSET_DEFAULT,
    phase2_end_offset: float = PHASE2_END_OFFSET_DEFAULT,
    phase1_gnss_lever_std: float = PHASE1_GNSS_LEVER_STD_DEFAULT,
    phase1_gnss_lever_sigma: float = PHASE1_GNSS_LEVER_SIGMA_DEFAULT,
    phase2_odo_scale_std: float = PHASE2_ODO_SCALE_STD_DEFAULT,
    phase2_mounting_std_deg: float = PHASE2_MOUNTING_STD_DEG_DEFAULT,
    phase2_odo_lever_std: float = PHASE2_ODO_LEVER_STD_DEFAULT,
    phase2_bgz_std_degh: float | None = None,
    bgz_process_noise_scale: float | None = None,
    odo_lever_process_noise: float | None = None,
    phase3_gnss_on_duration: float = PHASE3_GNSS_ON_DURATION_DEFAULT,
    phase3_gnss_off_duration: float = PHASE3_GNSS_OFF_DURATION_DEFAULT,
    phase_layout: str = PHASE_LAYOUT_DEFAULT,
    joint_estimation_duration: float = JOINT_ESTIMATION_DURATION_DEFAULT,
    freeze_gnss_lever_after_joint: bool = FREEZE_GNSS_LEVER_AFTER_JOINT_DEFAULT,
    truth_fix_mode: str = TRUTH_FIX_MODE_NONE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(base_cfg)
    output_dir_abs = normalize_repo_path(output_dir)
    case_id = case_id_for_filter_mode(filter_mode)
    fusion = cfg.setdefault("fusion", {})
    constraints = fusion.setdefault("constraints", {})
    ablation = fusion.setdefault("ablation", {})
    init_cfg = fusion.setdefault("init", {})
    noise = fusion.setdefault("noise", {})

    if filter_mode not in FILTER_MODE_CHOICES:
        raise ValueError(f"unsupported filter_mode={filter_mode!r}, expected one of {FILTER_MODE_CHOICES}")
    fusion["output_path"] = rel_from_root(output_dir_abs / f"SOL_{case_id}.txt", REPO_ROOT)
    fusion["state_series_output_path"] = rel_from_root(
        output_dir_abs / f"state_series_{case_id}.csv",
        REPO_ROOT,
    )
    fusion["inekf"] = {"enable": filter_mode == FILTER_MODE_INEKF}
    constraints["enable_odo"] = True
    constraints["enable_nhc"] = True
    constraints["enable_diagnostics"] = True
    constraints["enable_consistency_log"] = True
    ablation["disable_mounting_roll"] = True
    ablation["disable_mounting"] = False
    ablation["disable_odo_scale"] = False
    ablation["disable_odo_lever_arm"] = False
    ablation["disable_gnss_lever_arm"] = False
    ablation["disable_gnss_lever_z"] = False

    init_cfg["odo_scale"] = 1.0
    init_cfg["mounting_roll0"] = 0.0
    init_cfg["mounting_pitch0"] = 0.0
    init_cfg["mounting_yaw0"] = 0.0
    init_cfg["lever_arm0"] = [0.0, 0.0, 0.0]
    init_cfg["gnss_lever_arm0"] = [0.0, 0.0, 0.0]
    init_cfg["std_gnss_lever_arm"] = [float(phase1_gnss_lever_std)] * 3
    p0_diag = [float(x) for x in init_cfg["P0_diag"]]
    p0_diag[28:31] = [float(phase1_gnss_lever_std * phase1_gnss_lever_std)] * 3
    init_cfg["P0_diag"] = p0_diag

    noise["sigma_gnss_lever_arm"] = float(phase1_gnss_lever_sigma)
    noise["sigma_gnss_lever_arm_vec"] = [float(phase1_gnss_lever_sigma)] * 3
    if bgz_process_noise_scale is not None:
        sigma_bg_vec = build_base_sigma_bg_vector(noise)
        sigma_bg_vec[2] = float(sigma_bg_vec[2]) * float(bgz_process_noise_scale)
        noise["sigma_bg_vec"] = sigma_bg_vec
        noise["sigma_bg"] = float(max(sigma_bg_vec))
    if odo_lever_process_noise is not None:
        noise["sigma_lever_arm"] = float(odo_lever_process_noise)
        noise["sigma_lever_arm_vec"] = [float(odo_lever_process_noise)] * 3

    start_time = float(fusion["starttime"])
    final_time = float(fusion["finaltime"])
    if phase_layout not in PHASE_LAYOUT_CHOICES:
        raise ValueError(f"unsupported phase_layout={phase_layout!r}, expected one of {PHASE_LAYOUT_CHOICES}")
    if truth_fix_mode not in TRUTH_FIX_MODE_CHOICES:
        raise ValueError(f"unsupported truth_fix_mode={truth_fix_mode!r}, expected one of {TRUTH_FIX_MODE_CHOICES}")

    runtime_phases: list[dict[str, Any]]
    if phase_layout == PHASE_LAYOUT_DEFAULT:
        phase1_end_time = start_time + phase1_end_offset
        phase2_end_time = start_time + phase2_end_offset
        schedule_start_time = phase2_end_time
        runtime_phases = [
            {
                "name": "phase1_ins_gnss_estimate_gnss_lever",
                "start_time": start_time,
                "end_time": phase1_end_time,
                "ablation": {
                    "disable_odo_scale": True,
                    "disable_mounting": True,
                    "disable_mounting_roll": True,
                    "disable_odo_lever_arm": True,
                    "disable_gnss_lever_arm": False,
                },
                "constraints": {"enable_odo": False, "enable_nhc": False},
            },
            {
                "name": "phase2_ins_gnss_odo_nhc_activate_calibration",
                "start_time": phase1_end_time,
                "end_time": phase2_end_time,
                "ablation": {
                    "disable_odo_scale": False,
                    "disable_mounting": False,
                    "disable_mounting_roll": True,
                    "disable_odo_lever_arm": False,
                    "disable_gnss_lever_arm": True,
                },
                "constraints": {"enable_odo": True, "enable_nhc": True},
                "phase_entry_std_overrides": {
                    "std_odo_scale": float(phase2_odo_scale_std),
                    "std_mounting_pitch": float(phase2_mounting_std_deg),
                    "std_mounting_yaw": float(phase2_mounting_std_deg),
                    "std_lever_arm": [float(phase2_odo_lever_std)] * 3,
                },
            },
            {
                "name": "phase3_periodic_gnss_outage_keep_calibrating",
                "start_time": phase2_end_time,
                "end_time": final_time,
                "ablation": {
                    "disable_odo_scale": False,
                    "disable_mounting": False,
                    "disable_mounting_roll": True,
                    "disable_odo_lever_arm": False,
                    "disable_gnss_lever_arm": True,
                },
                "constraints": {"enable_odo": True, "enable_nhc": True},
            },
        ]
        if phase2_bgz_std_degh is not None:
            runtime_phases[1]["phase_entry_std_overrides"]["std_bg"] = build_base_bg_std_vector(p0_diag)
            runtime_phases[1]["phase_entry_std_overrides"]["std_bg"][2] = degph_to_radps(phase2_bgz_std_degh)
        policy_summary = (
            "phase1 only estimates `gnss_lever`; phase2/3 enable `ODO/NHC` calibration states "
            "except `mounting_roll`; `gnss_lever` freezes after phase1"
        )
    else:
        phase1_end_time = start_time + joint_estimation_duration
        schedule_start_time = phase1_end_time
        runtime_phases = [
            {
                "name": "phase1_joint_ins_gnss_odo_nhc_estimate_all_calibration",
                "start_time": start_time,
                "end_time": phase1_end_time,
                "ablation": {
                    "disable_odo_scale": False,
                    "disable_mounting": False,
                    "disable_mounting_roll": True,
                    "disable_odo_lever_arm": False,
                    "disable_gnss_lever_arm": False,
                },
                "constraints": {"enable_odo": True, "enable_nhc": True},
            },
            {
                "name": "phase2_periodic_gnss_outage_freeze_gnss_lever_keep_odo_nhc_calibrating",
                "start_time": phase1_end_time,
                "end_time": final_time,
                "ablation": {
                    "disable_odo_scale": False,
                    "disable_mounting": False,
                    "disable_mounting_roll": True,
                    "disable_odo_lever_arm": False,
                    "disable_gnss_lever_arm": bool(freeze_gnss_lever_after_joint),
                },
                "constraints": {"enable_odo": True, "enable_nhc": True},
            },
        ]
        policy_summary = (
            "phase1 jointly estimates `gnss_lever`, `odo_scale`, `mounting_pitch/yaw`, and "
            "`odo_lever` under `INS/GNSS/ODO/NHC` for the initial window; phase2 switches to "
            "periodic GNSS outage, keeps `ODO/NHC` calibration active, and freezes `gnss_lever`"
        )
    truth_fix_metadata = apply_truth_fix_mode(cfg, runtime_phases, truth_fix_mode=truth_fix_mode)
    fusion["gnss_schedule"]["enabled_windows"] = [
        {"start_time": window_start, "end_time": window_end}
        for window_start, window_end in build_periodic_enabled_windows(
            start_time,
            schedule_start_time,
            final_time,
            on_duration=phase3_gnss_on_duration,
            off_duration=phase3_gnss_off_duration,
        )
    ]
    fusion["runtime_phases"] = runtime_phases

    phase_records = build_phase_window_records(runtime_phases)
    enabled_windows = [
        [float(window["start_time"]), float(window["end_time"])]
        for window in fusion["gnss_schedule"]["enabled_windows"]
    ]
    metadata = {
        "case_id": case_id,
        "filter_mode": filter_mode,
        "phase_layout": phase_layout,
        "joint_estimation_duration": float(joint_estimation_duration),
        "freeze_gnss_lever_after_joint": bool(freeze_gnss_lever_after_joint),
        "truth_fix_mode": truth_fix_metadata["truth_fix_mode"],
        "truth_initialized": truth_fix_metadata["truth_initialized"],
        "truth_frozen_blocks": truth_fix_metadata["truth_frozen_blocks"],
        "runtime_phase_windows": phase_records,
        "plot_boundary_times": [float(phase_records[0]["end_time"]), float(phase_records[-1]["start_time"])],
        "transition_times": [
            {
                "transition_name": f"{phase_records[idx]['phase_key']}_to_{phase_records[idx + 1]['phase_key']}",
                "center_time": float(phase_records[idx]["end_time"]),
            }
            for idx in range(len(phase_records) - 1)
        ],
        "policy_summary": policy_summary,
        "phase1_gnss_lever_std": float(phase1_gnss_lever_std),
        "phase1_gnss_lever_sigma": float(phase1_gnss_lever_sigma),
        "phase2_odo_scale_std": float(phase2_odo_scale_std),
        "phase2_mounting_std_deg": float(phase2_mounting_std_deg),
        "phase2_odo_lever_std": float(phase2_odo_lever_std),
        "phase2_bgz_std_degh": None if phase2_bgz_std_degh is None else float(phase2_bgz_std_degh),
        "bgz_process_noise_scale": 1.0 if bgz_process_noise_scale is None else float(bgz_process_noise_scale),
        "odo_lever_process_noise": float(noise["sigma_lever_arm"]),
        "phase3_gnss_on_duration": float(phase3_gnss_on_duration),
        "phase3_gnss_off_duration": float(phase3_gnss_off_duration),
        "gnss_on_windows": enabled_windows,
        "gnss_off_windows": invert_enabled_windows(start_time, final_time, enabled_windows),
    }
    for phase_record in phase_records:
        metadata[f"{phase_record['phase_key']}_window"] = [
            float(phase_record["start_time"]),
            float(phase_record["end_time"]),
        ]
    return cfg, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal data2 three-phase INS/GNSS/ODO/NHC staged experiment."
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_DEFAULT)
    parser.add_argument("--exe", type=Path, default=SOLVER_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--exp-id", default=EXP_ID_DEFAULT)
    parser.add_argument("--filter-mode", choices=FILTER_MODE_CHOICES, default=FILTER_MODE_ESKF)
    parser.add_argument("--phase1-end-offset", type=float, default=PHASE1_END_OFFSET_DEFAULT)
    parser.add_argument("--phase2-end-offset", type=float, default=PHASE2_END_OFFSET_DEFAULT)
    parser.add_argument("--phase1-gnss-lever-std", type=float, default=PHASE1_GNSS_LEVER_STD_DEFAULT)
    parser.add_argument("--phase1-gnss-lever-sigma", type=float, default=PHASE1_GNSS_LEVER_SIGMA_DEFAULT)
    parser.add_argument("--phase2-odo-scale-std", type=float, default=PHASE2_ODO_SCALE_STD_DEFAULT)
    parser.add_argument("--phase2-mounting-std-deg", type=float, default=PHASE2_MOUNTING_STD_DEG_DEFAULT)
    parser.add_argument("--phase2-odo-lever-std", type=float, default=PHASE2_ODO_LEVER_STD_DEFAULT)
    parser.add_argument("--phase2-bgz-std-degh", type=float, default=None)
    parser.add_argument("--bgz-process-noise-scale", type=float, default=None)
    parser.add_argument("--odo-lever-process-noise", type=float, default=None)
    parser.add_argument("--phase3-gnss-on-duration", type=float, default=PHASE3_GNSS_ON_DURATION_DEFAULT)
    parser.add_argument("--phase3-gnss-off-duration", type=float, default=PHASE3_GNSS_OFF_DURATION_DEFAULT)
    parser.add_argument("--phase-layout", choices=PHASE_LAYOUT_CHOICES, default=PHASE_LAYOUT_DEFAULT)
    parser.add_argument("--joint-estimation-duration", type=float, default=JOINT_ESTIMATION_DURATION_DEFAULT)
    parser.add_argument(
        "--freeze-gnss-lever-after-joint",
        action=argparse.BooleanOptionalAction,
        default=FREEZE_GNSS_LEVER_AFTER_JOINT_DEFAULT,
    )
    parser.add_argument("--truth-fix-mode", choices=TRUTH_FIX_MODE_CHOICES, default=TRUTH_FIX_MODE_NONE)
    args = parser.parse_args()
    args.base_config = normalize_repo_path(args.base_config)
    args.exe = normalize_repo_path(args.exe)
    args.output_dir = normalize_repo_path(args.output_dir)
    args.case_id = case_id_for_filter_mode(args.filter_mode)
    args.artifacts_dir = args.output_dir / "artifacts"
    args.case_dir = args.artifacts_dir / "cases" / args.case_id
    args.plot_dir = args.output_dir / "plots"
    return args


def run_case(cfg_path: Path, output_dir: Path, case_dir: Path, exe_path: Path, case_id: str) -> dict[str, Any]:
    sol_path = output_dir / f"SOL_{case_id}.txt"
    state_series_path = output_dir / f"state_series_{case_id}.csv"
    stdout_path = case_dir / f"solver_stdout_{case_id}.txt"
    diag_path = case_dir / f"DIAG_{case_id}.txt"
    root_diag = REPO_ROOT / "DIAG.txt"
    if root_diag.exists():
        root_diag.unlink()
    stdout_text = run_command([str(exe_path.resolve()), "--config", str(cfg_path.resolve())], REPO_ROOT)
    stdout_path.write_text(stdout_text, encoding="utf-8")
    if not sol_path.exists():
        raise RuntimeError(f"missing solver output: {sol_path}")
    if not state_series_path.exists():
        raise RuntimeError(f"missing state series output: {state_series_path}")
    if not root_diag.exists():
        raise RuntimeError("missing DIAG.txt after staged estimation run")
    shutil.copy2(root_diag, diag_path)

    nav_metrics, segment_rows = evaluate_navigation_metrics(cfg_path, sol_path)
    row: dict[str, Any] = {
        "case_id": case_id,
        "case_label": case_label_for_filter_mode(FILTER_MODE_INEKF if case_id.endswith("_inekf") else FILTER_MODE_ESKF),
        "config_path": rel_from_root(cfg_path, REPO_ROOT),
        "sol_path": rel_from_root(sol_path, REPO_ROOT),
        "state_series_path": rel_from_root(state_series_path, REPO_ROOT),
        "diag_path": rel_from_root(diag_path, REPO_ROOT),
        "stdout_path": rel_from_root(stdout_path, REPO_ROOT),
        "config_mtime": mtime_text(cfg_path),
        "sol_mtime": mtime_text(sol_path),
        "state_series_mtime": mtime_text(state_series_path),
        "diag_mtime": mtime_text(diag_path),
        "stdout_mtime": mtime_text(stdout_path),
        "segment_rows": segment_rows,
    }
    row.update(nav_metrics)
    for sensor_name, metrics in parse_consistency_summary(stdout_text).items():
        prefix = sensor_name.lower()
        for metric_name, metric_value in metrics.items():
            row[f"{prefix}_{metric_name}"] = float(metric_value)
    return row


def add_error_columns(plot_df: pd.DataFrame, truth_reference: dict[str, Any]) -> pd.DataFrame:
    out = plot_df.copy()
    if "truth_bg_z_degh" in out.columns:
        truth_bg_z = out["truth_bg_z_degh"]
    else:
        truth_bg_z_ref = float(truth_reference["states"]["bg_z"]["reference_value"])
        truth_bg_z = pd.Series(truth_bg_z_ref, index=out.index, dtype=float)
    out["bg_z_degh_err"] = out["bg_z_degh"] - truth_bg_z
    return out


def compute_phase_metrics(
    plot_df: pd.DataFrame,
    phase_windows: list[tuple[str, float, float]],
    gnss_off_windows: list[tuple[float, float]],
    case_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_type, windows in (
        ("phase", phase_windows),
        ("gnss_off", [(f"gnss_off_{idx:02d}", start, end) for idx, (start, end) in enumerate(gnss_off_windows, start=1)]),
    ):
        for window_name, start_time, end_time in windows:
            subset = plot_df.loc[(plot_df["timestamp"] >= start_time) & (plot_df["timestamp"] <= end_time)]
            if subset.empty:
                continue
            err3 = np.linalg.norm(
                subset[["p_n_err_m", "p_e_err_m", "p_u_err_m"]].to_numpy(dtype=float),
                axis=1,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "window_type": window_type,
                    "window_name": window_name,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "samples": int(len(subset)),
                    "rmse_3d_m": float(np.sqrt(np.mean(err3 * err3))),
                    "p95_3d_m": float(np.percentile(err3, 95)),
                    "final_err_3d_m": float(err3[-1]),
                    "yaw_err_abs_max_deg": float(np.max(np.abs(subset["yaw_err_deg"]))),
                    "bg_z_err_abs_max_degh": float(np.max(np.abs(subset["bg_z_degh_err"]))),
                    "odo_scale_err_abs_max": float(np.max(np.abs(subset["odo_scale_err"]))),
                    "mounting_pitch_err_abs_max_deg": float(np.max(np.abs(subset["mounting_pitch_err_deg"]))),
                    "mounting_yaw_err_abs_max_deg": float(np.max(np.abs(subset["mounting_yaw_err_deg"]))),
                    "odo_lever_y_err_abs_max_m": float(np.max(np.abs(subset["odo_lever_y_err_m"]))),
                    "gnss_lever_y_err_abs_max_m": float(np.max(np.abs(subset["gnss_lever_y_err_m"]))),
                }
            )
    return pd.DataFrame(rows)


def compute_transition_metrics(
    plot_df: pd.DataFrame,
    transitions: list[tuple[str, float]],
    case_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    t = plot_df["timestamp"].to_numpy(dtype=float)
    sec_grid = np.arange(math.ceil(float(t[0])), math.floor(float(t[-1])) + 1.0, 1.0, dtype=float)
    if sec_grid.size < 2:
        return pd.DataFrame()
    for transition_name, center_time in transitions:
        mask = (sec_grid >= center_time - 10.0) & (sec_grid <= center_time + 30.0)
        sec_window = sec_grid[mask]
        if sec_window.size < 2:
            continue
        for column in KEY_STATE_COLUMNS:
            sec_values = np.interp(sec_grid, t, plot_df[column.key].to_numpy(dtype=float))
            sec_subset = sec_values[mask]
            sec_diff = np.diff(sec_subset)
            diff_times = sec_window[1:]
            max_idx = int(np.argmax(np.abs(sec_diff)))
            rows.append(
                {
                    "case_id": case_id,
                    "transition": transition_name,
                    "state_key": column.key,
                    "center_time": float(center_time),
                    "window_start": float(sec_window[0]),
                    "window_end": float(sec_window[-1]),
                    "max_abs_jump_1hz": float(np.max(np.abs(sec_diff))),
                    "max_jump_t_1hz": float(diff_times[max_idx]),
                    "total_variation_1hz": float(np.sum(np.abs(sec_diff))),
                }
            )
    return pd.DataFrame(rows)


def compute_state_metrics(
    plot_df: pd.DataFrame,
    phase_windows: list[tuple[str, float, float]],
    case_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_name, start_time, end_time in phase_windows:
        subset = plot_df.loc[(plot_df["timestamp"] >= start_time) & (plot_df["timestamp"] <= end_time)]
        if subset.empty:
            continue
        for column in STATE_METRIC_COLUMNS:
            values = subset[column.key].to_numpy(dtype=float)
            rows.append(
                {
                    "case_id": case_id,
                    "window_name": window_name,
                    "state_key": column.key,
                    "label": column.label,
                    "unit": column.unit,
                    "final_value": float(values[-1]),
                    "max_abs": float(np.max(np.abs(values))),
                    "rmse": float(np.sqrt(np.mean(values * values))),
                }
            )
    return pd.DataFrame(rows)


def compute_case_metrics_row(case_row: dict[str, Any], plot_df: pd.DataFrame) -> dict[str, Any]:
    metrics = dict(case_row)
    metrics["yaw_err_max_abs_deg"] = float(np.max(np.abs(plot_df["yaw_err_deg"].to_numpy(dtype=float))))
    metrics["bg_z_degh_err_max_abs"] = float(np.max(np.abs(plot_df["bg_z_degh_err"].to_numpy(dtype=float))))
    return metrics


def build_plot_frame_from_initial_delta(state_frame: pd.DataFrame) -> pd.DataFrame:
    plot_frame = state_frame.copy()
    for key, truth_key in CALIBRATION_DELTA_COLUMNS.items():
        if key not in plot_frame.columns or truth_key not in plot_frame.columns:
            continue
        init_value = float(plot_frame[key].iloc[0])
        plot_frame[key] = plot_frame[key].to_numpy(dtype=float) - init_value
        plot_frame[truth_key] = plot_frame[truth_key].to_numpy(dtype=float) - init_value
    return plot_frame


def render_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    return str(value)


def write_summary(
    output_path: Path,
    manifest: dict[str, Any],
    case_metrics_df: pd.DataFrame,
    phase_metrics_df: pd.DataFrame,
    state_metrics_df: pd.DataFrame,
    plot_paths: dict[str, str],
) -> None:
    case_rows = [
        [
            str(row["case_id"]),
            format_metric(row.get("overall_rmse_3d_m_aux")),
            format_metric(row.get("overall_p95_3d_m_aux")),
            format_metric(row.get("overall_final_err_3d_m_aux")),
            format_metric(row.get("yaw_err_max_abs_deg")),
            format_metric(row.get("bg_z_degh_err_max_abs")),
            format_metric(row.get("odo_accept_ratio")),
            format_metric(row.get("nhc_accept_ratio")),
        ]
        for _, row in case_metrics_df.iterrows()
    ]
    phase_rows = [
        [
            str(row["window_name"]),
            format_metric(row["rmse_3d_m"]),
            format_metric(row["p95_3d_m"]),
            format_metric(row["final_err_3d_m"]),
            format_metric(row["yaw_err_abs_max_deg"]),
            format_metric(row["bg_z_err_abs_max_degh"]),
            format_metric(row["gnss_lever_y_err_abs_max_m"]),
            format_metric(row["odo_lever_y_err_abs_max_m"]),
        ]
        for _, row in phase_metrics_df.loc[phase_metrics_df["window_type"] == "phase"].iterrows()
    ]
    state_rows = [
        [
            str(row["window_name"]),
            str(row["state_key"]),
            format_metric(row["final_value"]),
            format_metric(row["max_abs"]),
            format_metric(row["rmse"]),
        ]
        for _, row in state_metrics_df.loc[
            state_metrics_df["state_key"].isin(
                [
                    "odo_scale_err",
                    "mounting_pitch_err_deg",
                    "mounting_yaw_err_deg",
                    "odo_lever_y_err_m",
                    "gnss_lever_y_err_m",
                    "bg_z_degh_err",
                ]
            )
        ].iterrows()
    ]

    lines = [
        "# data2 INS/GNSS/ODO/NHC staged estimation summary",
        "",
        f"- exp_id: `{manifest['exp_id']}`",
        f"- base_config: `{manifest['base_config']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- filter_mode: `{manifest['filter_mode']}`",
        f"- phase_layout: `{manifest['phase_layout']}`",
        f"- truth_fix_mode: `{manifest['truth_fix_mode']}`",
        f"- phase2_bgz_std_degh: `{manifest['phase_entry_defaults']['phase2_bgz_std_degh']}`",
        f"- bgz_process_noise_scale: `{manifest['phase_entry_defaults']['bgz_process_noise_scale']}`",
        f"- phase2_odo_lever_std: `{manifest['phase_entry_defaults']['phase2_odo_lever_std']}`",
        f"- odo_lever_process_noise: `{manifest['phase_entry_defaults']['odo_lever_process_noise']}`",
        f"- policy: {manifest['policy_summary']}",
        "",
        "## Case Metrics",
    ]
    lines.extend(
        render_table(
            [
                "case_id",
                "rmse3d_m",
                "p95_3d_m",
                "final_3d_m",
                "yaw_err_max_abs_deg",
                "bg_z_err_max_abs_degh",
                "odo_accept_ratio",
                "nhc_accept_ratio",
            ],
            case_rows,
        )
    )
    lines.extend(["", "## Phase Metrics"])
    lines.extend(
        render_table(
            [
                "window",
                "rmse3d_m",
                "p95_3d_m",
                "final_3d_m",
                "yaw_abs_max_deg",
                "bg_z_err_abs_max_degh",
                "gnss_lever_y_abs_max_m",
                "odo_lever_y_abs_max_m",
            ],
            phase_rows,
        )
    )
    lines.extend(["", "## Key State Metrics"])
    lines.extend(render_table(["window", "state_key", "final_value", "max_abs", "rmse"], state_rows))
    lines.extend(["", "## Outputs"])
    for key, path in plot_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.base_config.exists():
        raise FileNotFoundError(f"missing base config: {args.base_config}")
    if not args.exe.exists():
        raise FileNotFoundError(f"missing solver executable: {args.exe}")

    reset_directory(args.output_dir)
    ensure_dir(args.artifacts_dir)
    ensure_dir(args.case_dir)
    ensure_dir(args.plot_dir)

    base_cfg = load_yaml(args.base_config)
    cfg, metadata = build_case_config(
        base_cfg,
        args.output_dir,
        filter_mode=args.filter_mode,
        phase1_end_offset=args.phase1_end_offset,
        phase2_end_offset=args.phase2_end_offset,
        phase1_gnss_lever_std=args.phase1_gnss_lever_std,
        phase1_gnss_lever_sigma=args.phase1_gnss_lever_sigma,
        phase2_odo_scale_std=args.phase2_odo_scale_std,
        phase2_mounting_std_deg=args.phase2_mounting_std_deg,
        phase2_odo_lever_std=args.phase2_odo_lever_std,
        phase2_bgz_std_degh=args.phase2_bgz_std_degh,
        bgz_process_noise_scale=args.bgz_process_noise_scale,
        odo_lever_process_noise=args.odo_lever_process_noise,
        phase3_gnss_on_duration=args.phase3_gnss_on_duration,
        phase3_gnss_off_duration=args.phase3_gnss_off_duration,
        phase_layout=args.phase_layout,
        joint_estimation_duration=args.joint_estimation_duration,
        freeze_gnss_lever_after_joint=args.freeze_gnss_lever_after_joint,
        truth_fix_mode=args.truth_fix_mode,
    )

    cfg_path = args.case_dir / f"config_{args.case_id}.yaml"
    save_yaml(cfg, cfg_path)

    truth_reference = build_truth_reference(cfg)
    truth_reference_path = args.output_dir / "truth_reference.json"
    truth_reference_path.write_text(
        json.dumps(json_safe(truth_reference), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    case_row = run_case(cfg_path, args.output_dir, args.case_dir, args.exe, args.case_id)
    sol_path = args.output_dir / f"SOL_{args.case_id}.txt"
    state_series_path = args.output_dir / f"state_series_{args.case_id}.csv"
    merged_df = merge_case_outputs(sol_path, state_series_path)

    truth_df = load_pos_dataframe((REPO_ROOT / cfg["fusion"]["pos_path"]).resolve())
    imu_df = load_imu_dataframe((REPO_ROOT / cfg["fusion"]["imu_path"]).resolve())
    truth_interp_df = build_truth_interp(merged_df["timestamp"].to_numpy(dtype=float), truth_df)
    state_frame = build_state_frame(merged_df, truth_interp_df, truth_reference)
    motion_df = build_motion_frame(merged_df, truth_interp_df, imu_df, truth_reference)
    plot_df = add_error_columns(
        build_plot_frame(merged_df, truth_interp_df, truth_reference, motion_df),
        truth_reference,
    )

    all_states_path = args.case_dir / f"all_states_{args.case_id}.csv"
    build_all_states_export_frame(state_frame).to_csv(all_states_path, index=False, encoding="utf-8-sig")

    phase_windows = [
        (str(phase_record["window_name"]), float(phase_record["start_time"]), float(phase_record["end_time"]))
        for phase_record in metadata["runtime_phase_windows"]
    ]
    gnss_off_windows = [tuple(window) for window in metadata["gnss_off_windows"]]
    phase_metrics_df = compute_phase_metrics(plot_df, phase_windows, gnss_off_windows, args.case_id)
    transition_metrics_df = compute_transition_metrics(
        plot_df,
        [
            (str(item["transition_name"]), float(item["center_time"]))
            for item in metadata["transition_times"]
        ],
        args.case_id,
    )
    state_metrics_df = compute_state_metrics(plot_df, phase_windows, args.case_id)

    metrics_row = compute_case_metrics_row(case_row, plot_df)
    metrics_row["all_states_path"] = rel_from_root(all_states_path, REPO_ROOT)
    metrics_row["all_states_mtime"] = mtime_text(all_states_path)
    case_metrics_df = pd.DataFrame([metrics_row])

    case_metrics_path = args.output_dir / "case_metrics.csv"
    phase_metrics_path = args.output_dir / "phase_metrics.csv"
    transition_metrics_path = args.output_dir / "transition_metrics.csv"
    state_metrics_path = args.output_dir / "state_metrics.csv"
    case_metrics_df.to_csv(case_metrics_path, index=False, encoding="utf-8-sig")
    phase_metrics_df.to_csv(phase_metrics_path, index=False, encoding="utf-8-sig")
    transition_metrics_df.to_csv(transition_metrics_path, index=False, encoding="utf-8-sig")
    state_metrics_df.to_csv(state_metrics_path, index=False, encoding="utf-8-sig")

    plot_boundary_times = [float(value) for value in metadata["plot_boundary_times"]]
    plot_paths: dict[str, str] = {}
    plot_case = PlotCase(case_id=args.case_id, label=case_label_for_filter_mode(args.filter_mode), color="#d62728")
    case_frames = {args.case_id: build_plot_frame_from_initial_delta(state_frame)}
    plot_config = build_mainline_plot_config()
    truth_keys_to_hide = build_staged_truth_keys_to_hide(plot_config.truth_keys_to_hide)
    remove_obsolete_mainline_plot_files(args.plot_dir)

    overview_path = args.plot_dir / "all_states_overview.png"
    plot_state_grid(
        case_frames,
        [plot_case],
        plot_config.overview_states,
        overview_path,
        f"data2 ins gnss odo nhc staged estimation {args.filter_mode} all-state overview",
        plot_boundary_times[0],
        plot_boundary_times[1],
        gnss_off_windows,
        truth_keys_to_hide=truth_keys_to_hide,
    )
    plot_paths["all_states_overview"] = rel_from_root(overview_path, REPO_ROOT)

    key_states_path = args.plot_dir / "key_coupling_states.png"
    plot_state_grid(
        case_frames,
        [plot_case],
        KEY_COUPLING_STATES,
        key_states_path,
        f"data2 ins gnss odo nhc staged estimation {args.filter_mode} key coupling states",
        plot_boundary_times[0],
        plot_boundary_times[1],
        gnss_off_windows,
        truth_keys_to_hide=truth_keys_to_hide,
    )
    plot_paths["key_coupling_states"] = rel_from_root(key_states_path, REPO_ROOT)

    for group_spec in plot_config.group_specs:
        group_path = args.plot_dir / f"{group_spec.group_id}.png"
        selected_group = group_spec
        plot_mode = "state"
        group_truth_keys_to_hide = truth_keys_to_hide
        if group_spec.group_id in {"position", "velocity", "attitude"}:
            plot_mode = "error"
            selected_group = next(item for item in PVA_ERROR_GROUP_SPECS if item.group_id == group_spec.group_id)
        elif group_spec.group_id in CALIBRATION_TRUTH_VISIBLE_KEYS_BY_GROUP:
            group_truth_keys_to_hide = build_staged_truth_keys_to_hide(
                plot_config.truth_keys_to_hide,
                visible_keys=CALIBRATION_TRUTH_VISIBLE_KEYS_BY_GROUP[group_spec.group_id],
            )
        plot_state_grid(
            case_frames,
            [plot_case],
            selected_group.states,
            group_path,
            f"data2 ins gnss odo nhc staged estimation {args.filter_mode} {selected_group.title}",
            plot_boundary_times[0],
            plot_boundary_times[1],
            gnss_off_windows,
            plot_mode=plot_mode,
            truth_keys_to_hide=group_truth_keys_to_hide,
        )
        plot_paths[group_spec.group_id] = rel_from_root(group_path, REPO_ROOT)

    summary_path = args.output_dir / "summary.md"
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "exp_id": args.exp_id,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "base_config": rel_from_root(args.base_config, REPO_ROOT),
        "solver_exe": rel_from_root(args.exe, REPO_ROOT),
        "output_dir": rel_from_root(args.output_dir, REPO_ROOT),
        "filter_mode": args.filter_mode,
        "case_config": rel_from_root(cfg_path, REPO_ROOT),
        "case_metrics_csv": rel_from_root(case_metrics_path, REPO_ROOT),
        "phase_metrics_csv": rel_from_root(phase_metrics_path, REPO_ROOT),
        "transition_metrics_csv": rel_from_root(transition_metrics_path, REPO_ROOT),
        "state_metrics_csv": rel_from_root(state_metrics_path, REPO_ROOT),
        "truth_reference_json": rel_from_root(truth_reference_path, REPO_ROOT),
        "plots": plot_paths,
        "phase_layout": metadata["phase_layout"],
        "truth_fix_mode": metadata["truth_fix_mode"],
        "truth_initialized": metadata["truth_initialized"],
        "truth_frozen_blocks": metadata["truth_frozen_blocks"],
        "policy_summary": metadata["policy_summary"],
        "phase_windows": {
            str(phase_record["phase_key"]): [float(phase_record["start_time"]), float(phase_record["end_time"])]
            for phase_record in metadata["runtime_phase_windows"]
        },
        "runtime_phase_windows": metadata["runtime_phase_windows"],
        "phase_entry_defaults": {
            "filter_mode": metadata["filter_mode"],
            "phase1_gnss_lever_std": metadata["phase1_gnss_lever_std"],
            "phase1_gnss_lever_sigma": metadata["phase1_gnss_lever_sigma"],
            "phase2_odo_scale_std": metadata["phase2_odo_scale_std"],
            "phase2_mounting_std_deg": metadata["phase2_mounting_std_deg"],
            "phase2_odo_lever_std": metadata["phase2_odo_lever_std"],
            "phase2_bgz_std_degh": metadata["phase2_bgz_std_degh"],
            "bgz_process_noise_scale": metadata["bgz_process_noise_scale"],
            "odo_lever_process_noise": metadata["odo_lever_process_noise"],
            "phase3_gnss_on_duration": metadata["phase3_gnss_on_duration"],
            "phase3_gnss_off_duration": metadata["phase3_gnss_off_duration"],
            "joint_estimation_duration": metadata["joint_estimation_duration"],
            "freeze_gnss_lever_after_joint": metadata["freeze_gnss_lever_after_joint"],
            "truth_fix_mode": metadata["truth_fix_mode"],
        },
        "gnss_on_windows": metadata["gnss_on_windows"],
        "gnss_off_windows": metadata["gnss_off_windows"],
        "freshness": {
            "base_config_mtime": mtime_text(args.base_config),
            "solver_exe_mtime": mtime_text(args.exe),
            "case_config_mtime": mtime_text(cfg_path),
            "summary_mtime": None,
        },
    }
    write_summary(summary_path, manifest, case_metrics_df, phase_metrics_df, state_metrics_df, plot_paths)
    manifest["summary_md"] = rel_from_root(summary_path, REPO_ROOT)
    manifest["freshness"]["summary_mtime"] = mtime_text(summary_path)
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(rel_from_root(manifest_path, REPO_ROOT))


if __name__ == "__main__":
    main()
