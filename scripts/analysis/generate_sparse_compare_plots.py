#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_FIELDS = [
    "p_n_m",
    "p_e_m",
    "p_u_m",
    "v_n_mps",
    "v_e_mps",
    "v_u_mps",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "ba_x_mgal",
    "ba_y_mgal",
    "ba_z_mgal",
    "bg_x_degh",
    "bg_y_degh",
    "bg_z_degh",
    "sg_x_ppm",
    "sg_y_ppm",
    "sg_z_ppm",
    "sa_x_ppm",
    "sa_y_ppm",
    "sa_z_ppm",
]

TRUTH_FIELDS = [f"truth_{field}" for field in STATE_FIELDS]

GROUPS = {
    "position_state": ["p_n_m", "p_e_m", "p_u_m"],
    "velocity_state": ["v_n_mps", "v_e_mps", "v_u_mps"],
    "attitude_state": ["roll_deg", "pitch_deg", "yaw_deg"],
    "ba_state": ["ba_x_mgal", "ba_y_mgal", "ba_z_mgal"],
    "bg_state": ["bg_x_degh", "bg_y_degh", "bg_z_degh"],
    "sg_state": ["sg_x_ppm", "sg_y_ppm", "sg_z_ppm"],
    "sa_state": ["sa_x_ppm", "sa_y_ppm", "sa_z_ppm"],
    "position_error": ["p_n_m", "p_e_m", "p_u_m"],
    "velocity_error": ["v_n_mps", "v_e_mps", "v_u_mps"],
    "attitude_error": ["roll_deg", "pitch_deg", "yaw_deg"],
}

PLOT_FILENAMES = {
    "all_shared_states_overview": "all_shared_states_overview.png",
    "key_shared_states": "key_shared_states.png",
    "position_state": "position_state_compare.png",
    "velocity_state": "velocity_state_compare.png",
    "attitude_state": "attitude_state_compare.png",
    "ba_state": "ba_state_compare.png",
    "bg_state": "bg_state_compare.png",
    "sg_state": "sg_state_compare.png",
    "sa_state": "sa_state_compare.png",
    "position_error": "position_error_compare.png",
    "velocity_error": "velocity_error_compare.png",
    "attitude_error": "attitude_error_compare.png",
}

FIELD_LABELS = {
    "p_n_m": "p_n (m)",
    "p_e_m": "p_e (m)",
    "p_u_m": "p_u (m)",
    "v_n_mps": "v_n (m/s)",
    "v_e_mps": "v_e (m/s)",
    "v_u_mps": "v_u (m/s)",
    "roll_deg": "roll (deg)",
    "pitch_deg": "pitch (deg)",
    "yaw_deg": "yaw (deg)",
    "ba_x_mgal": "ba_x (mGal)",
    "ba_y_mgal": "ba_y (mGal)",
    "ba_z_mgal": "ba_z (mGal)",
    "bg_x_degh": "bg_x (deg/h)",
    "bg_y_degh": "bg_y (deg/h)",
    "bg_z_degh": "bg_z (deg/h)",
    "sg_x_ppm": "sg_x (ppm)",
    "sg_y_ppm": "sg_y (ppm)",
    "sg_z_ppm": "sg_z (ppm)",
    "sa_x_ppm": "sa_x (ppm)",
    "sa_y_ppm": "sa_y (ppm)",
    "sa_z_ppm": "sa_z (ppm)",
}

ANGLE_FIELDS = {"roll_deg", "pitch_deg", "yaw_deg"}
NO_TRUTH_OVERLAY_FIELDS = {
    "ba_x_mgal",
    "ba_y_mgal",
    "ba_z_mgal",
    "bg_x_degh",
    "bg_y_degh",
    "bg_z_degh",
    "sg_x_ppm",
    "sg_y_ppm",
    "sg_z_ppm",
    "sa_x_ppm",
    "sa_y_ppm",
    "sa_z_ppm",
}

CURRENT_STYLE = dict(color="#d95f02", linewidth=1.2, alpha=0.95, linestyle="-", zorder=3)
KF_STYLE = dict(color="#1f77b4", linewidth=1.25, alpha=0.95, linestyle="--", zorder=4)
TRUTH_STYLE = dict(color="#111111", linewidth=0.9, linestyle=":", alpha=0.82, zorder=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for sparse GNSS current-vs-KF-GINS runs."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory such as output/data5_kf_sparse15s_cmp_r2",
    )
    parser.add_argument(
        "--min-plot-dt",
        type=float,
        default=0.1,
        help="Minimum time spacing in seconds kept for plotting only.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(root: Path, raw: str) -> Path:
    return root / Path(raw.replace("\\", "/"))


def wrap_deg(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def thin_for_plot(df: pd.DataFrame, min_dt: float) -> pd.DataFrame:
    if min_dt <= 0.0 or df.empty:
        return df
    t = df["timestamp"].to_numpy(dtype=np.float64)
    bins = np.floor((t - t[0]) / min_dt + 1.0e-12).astype(np.int64)
    keep = np.empty(len(df), dtype=bool)
    keep[0] = True
    keep[1:] = bins[1:] != bins[:-1]
    return df.loc[keep].reset_index(drop=True)


def read_current_states(path: Path) -> pd.DataFrame:
    dtype_map = {"timestamp": np.float64}
    for col in STATE_FIELDS + TRUTH_FIELDS:
        dtype_map[col] = np.float32
    return pd.read_csv(path, usecols=["timestamp", *STATE_FIELDS, *TRUTH_FIELDS], dtype=dtype_map)


def read_kf_states(path: Path) -> pd.DataFrame:
    dtype_map = {"timestamp": np.float64}
    for col in STATE_FIELDS:
        dtype_map[col] = np.float32
    return pd.read_csv(path, usecols=["timestamp", *STATE_FIELDS], dtype=dtype_map)


def merge_states(current_df: pd.DataFrame, kf_df: pd.DataFrame) -> pd.DataFrame:
    try:
        merged = current_df.merge(
            kf_df,
            on="timestamp",
            how="inner",
            suffixes=("_current", "_kf"),
            validate="one_to_one",
        )
        if len(merged) >= 0.95 * min(len(current_df), len(kf_df)):
            return merged
    except Exception:
        pass

    merged = pd.merge_asof(
        current_df.sort_values("timestamp"),
        kf_df.sort_values("timestamp"),
        on="timestamp",
        suffixes=("_current", "_kf"),
        tolerance=1.0e-6,
        direction="nearest",
    )
    merged = merged.dropna().reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("Failed to align current and KF-GINS state series.")
    return merged


def aligned_state(merged: pd.DataFrame, field: str, source: str) -> np.ndarray:
    values = merged[f"{field}_{source}"].to_numpy(dtype=np.float64)
    if field not in ANGLE_FIELDS:
        return values
    truth = merged[f"truth_{field}"].to_numpy(dtype=np.float64)
    return truth + wrap_deg(values - truth)


def truth_state(merged: pd.DataFrame, field: str) -> np.ndarray:
    return merged[f"truth_{field}"].to_numpy(dtype=np.float64)


def state_error(merged: pd.DataFrame, field: str, source: str) -> np.ndarray:
    values = merged[f"{field}_{source}"].to_numpy(dtype=np.float64)
    truth = merged[f"truth_{field}"].to_numpy(dtype=np.float64)
    if field in ANGLE_FIELDS:
        return wrap_deg(values - truth)
    return values - truth


def load_key_fields(state_delta_path: Path) -> list[str]:
    df = pd.read_csv(state_delta_path)
    if "state_key" not in df.columns:
        return ["yaw_deg", "bg_z_degh", "ba_x_mgal", "sg_x_ppm", "sg_y_ppm", "sg_z_ppm", "sa_x_ppm", "sa_y_ppm", "p_n_m"]
    fields = [field for field in df["state_key"].tolist() if field in STATE_FIELDS]
    deduped: list[str] = []
    for field in fields:
        if field not in deduped:
            deduped.append(field)
    fallback = ["yaw_deg", "bg_z_degh", "ba_x_mgal", "ba_y_mgal", "sg_x_ppm", "sg_y_ppm", "sg_z_ppm", "sa_x_ppm", "sa_y_ppm"]
    for field in fallback:
        if field not in deduped:
            deduped.append(field)
    return deduped[:9]


def apply_axis_style(ax: plt.Axes, x: np.ndarray, field: str) -> None:
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    ax.set_ylabel(FIELD_LABELS[field])
    ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    ax.set_xlim(x[0], x[-1])


def style_with_label(style: dict, label: str) -> dict:
    styled = dict(style)
    styled["label"] = label
    return styled


def should_draw_truth(field: str, *, error_mode: bool) -> bool:
    return (not error_mode) and field not in NO_TRUTH_OVERLAY_FIELDS


def plot_group(
    merged: pd.DataFrame,
    x: np.ndarray,
    fields: Iterable[str],
    output_path: Path,
    title: str,
    *,
    error_mode: bool = False,
    current_label: str = "current",
    kf_label: str = "KF-GINS",
    truth_label: str = "truth",
) -> None:
    fields = list(fields)
    fig, axes = plt.subplots(len(fields), 1, figsize=(13.5, 3.2 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        if error_mode:
            ax.axhline(0.0, color="#111111", linestyle="--", linewidth=0.8, alpha=0.6, label="zero", zorder=1)
            ax.plot(x, state_error(merged, field, "current"), **style_with_label(CURRENT_STYLE, current_label))
            ax.plot(x, state_error(merged, field, "kf"), **style_with_label(KF_STYLE, kf_label))
        else:
            if should_draw_truth(field, error_mode=error_mode):
                ax.plot(x, truth_state(merged, field), **style_with_label(TRUTH_STYLE, truth_label))
            ax.plot(x, aligned_state(merged, field, "current"), **style_with_label(CURRENT_STYLE, current_label))
            ax.plot(x, aligned_state(merged, field, "kf"), **style_with_label(KF_STYLE, kf_label))
        apply_axis_style(ax, x, field)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    axes[-1].set_xlabel("time since run start (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_overview(
    merged: pd.DataFrame,
    x: np.ndarray,
    fields: Iterable[str],
    output_path: Path,
    title: str,
    *,
    current_label: str = "current",
    kf_label: str = "KF-GINS",
    truth_label: str = "truth",
) -> None:
    fields = list(fields)
    ncols = 3
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.5, 3.0 * nrows), sharex=True)
    axes_list = np.atleast_1d(axes).ravel().tolist()
    show_truth_in_legend = any(should_draw_truth(field, error_mode=False) for field in fields)

    for ax, field in zip(axes_list, fields):
        if should_draw_truth(field, error_mode=False):
            ax.plot(
                x,
                truth_state(merged, field),
                color=TRUTH_STYLE["color"],
                linestyle=TRUTH_STYLE["linestyle"],
                linewidth=TRUTH_STYLE["linewidth"],
                alpha=TRUTH_STYLE["alpha"],
                zorder=TRUTH_STYLE["zorder"],
            )
        ax.plot(
            x,
            aligned_state(merged, field, "current"),
            color=CURRENT_STYLE["color"],
            linestyle=CURRENT_STYLE["linestyle"],
            linewidth=CURRENT_STYLE["linewidth"],
            alpha=CURRENT_STYLE["alpha"],
            zorder=CURRENT_STYLE["zorder"],
        )
        ax.plot(
            x,
            aligned_state(merged, field, "kf"),
            color=KF_STYLE["color"],
            linestyle=KF_STYLE["linestyle"],
            linewidth=KF_STYLE["linewidth"],
            alpha=KF_STYLE["alpha"],
            zorder=KF_STYLE["zorder"],
        )
        ax.set_title(FIELD_LABELS[field], fontsize=9)
        apply_axis_style(ax, x, field)

    for ax in axes_list[len(fields):]:
        ax.axis("off")

    handles = []
    labels = []
    if show_truth_in_legend:
        handles.append(plt.Line2D([0], [0], **style_with_label(TRUTH_STYLE, truth_label)))
        labels.append(truth_label)
    handles.append(plt.Line2D([0], [0], **style_with_label(CURRENT_STYLE, current_label)))
    labels.append(current_label)
    handles.append(plt.Line2D([0], [0], **style_with_label(KF_STYLE, kf_label)))
    labels.append(kf_label)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    for ax in axes_list[max(0, len(fields) - ncols):len(fields)]:
        ax.set_xlabel("time since run start (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def update_manifest(manifest_path: Path, plots: dict[str, str]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["plots"] = plots
    manifest.setdefault("freshness", {})
    manifest["freshness"]["plots_generated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rewrite_summary(summary_path: Path, plots: dict[str, str], min_plot_dt: float) -> None:
    text = summary_path.read_text(encoding="utf-8")
    plot_lines = ["## Plot Outputs", *[f"- `{name}`: `{path.replace('\\\\', '/')}`" for name, path in plots.items()]]
    plot_section = "\n".join(plot_lines)

    marker = "## Plot Outputs"
    if marker in text:
        start = text.index(marker)
        next_idx = text.find("\n## ", start + len(marker))
        if next_idx == -1:
            text = text[:start].rstrip() + "\n\n" + plot_section + "\n"
        else:
            text = text[:start].rstrip() + "\n\n" + plot_section + "\n\n" + text[next_idx + 1 :]
    elif "## Notes" in text:
        text = text.replace("## Notes", plot_section + "\n\n## Notes", 1)
    else:
        text = text.rstrip() + "\n\n" + plot_section + "\n"

    old_note = "- Plot images were not regenerated in this rerun; the primary refreshed artifacts are raw outputs, `all_states`, metrics CSVs, manifest, and summary."
    new_note = (
        f"- Plot images were regenerated into `{summary_path.parent.as_posix()}/plots/`; "
        f"rendering uses a plotting-only decimation of {min_plot_dt:.3f}s and does not affect metrics."
    )
    if old_note in text:
        text = text.replace(old_note, new_note)
    elif new_note not in text and "## Notes" in text:
        text = text.rstrip() + "\n" if not text.endswith("\n") else text
        text += new_note + "\n"

    summary_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    run_dir = resolve_repo_path(root, args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_path = resolve_repo_path(root, manifest["current_case"]["all_states_path"])
    kf_path = resolve_repo_path(root, manifest["kf_case"]["all_states_path"])
    state_delta_path = resolve_repo_path(root, manifest["artifacts"]["state_delta_summary"])

    current_df = thin_for_plot(read_current_states(current_path), args.min_plot_dt)
    kf_df = thin_for_plot(read_kf_states(kf_path), args.min_plot_dt)
    merged = merge_states(current_df, kf_df)

    x = merged["timestamp"].to_numpy(dtype=np.float64)
    x = x - x[0]

    key_fields = load_key_fields(state_delta_path)

    plot_overview(
        merged,
        x,
        STATE_FIELDS,
        plots_dir / PLOT_FILENAMES["all_shared_states_overview"],
        "All Shared States Overview",
    )
    plot_overview(
        merged,
        x,
        key_fields,
        plots_dir / PLOT_FILENAMES["key_shared_states"],
        "Key Shared States",
    )
    plot_group(
        merged,
        x,
        GROUPS["position_state"],
        plots_dir / PLOT_FILENAMES["position_state"],
        "Position State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["velocity_state"],
        plots_dir / PLOT_FILENAMES["velocity_state"],
        "Velocity State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["attitude_state"],
        plots_dir / PLOT_FILENAMES["attitude_state"],
        "Attitude State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["ba_state"],
        plots_dir / PLOT_FILENAMES["ba_state"],
        "Accelerometer Bias State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["bg_state"],
        plots_dir / PLOT_FILENAMES["bg_state"],
        "Gyro Bias State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["sg_state"],
        plots_dir / PLOT_FILENAMES["sg_state"],
        "Gyro Scale State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["sa_state"],
        plots_dir / PLOT_FILENAMES["sa_state"],
        "Accelerometer Scale State Compare",
    )
    plot_group(
        merged,
        x,
        GROUPS["position_error"],
        plots_dir / PLOT_FILENAMES["position_error"],
        "Position Error Compare",
        error_mode=True,
    )
    plot_group(
        merged,
        x,
        GROUPS["velocity_error"],
        plots_dir / PLOT_FILENAMES["velocity_error"],
        "Velocity Error Compare",
        error_mode=True,
    )
    plot_group(
        merged,
        x,
        GROUPS["attitude_error"],
        plots_dir / PLOT_FILENAMES["attitude_error"],
        "Attitude Error Compare",
        error_mode=True,
    )

    plot_manifest = {
        name: str((run_dir / "plots" / filename).relative_to(root)).replace("/", "\\")
        for name, filename in PLOT_FILENAMES.items()
    }
    update_manifest(manifest_path, plot_manifest)
    rewrite_summary(summary_path, plot_manifest, args.min_plot_dt)


if __name__ == "__main__":
    main()
