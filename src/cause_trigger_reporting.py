from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap
from typing import Callable, Mapping, Sequence
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BACKEND_LABELS = {
    "hmml": "HMML (baseline)",
    "pcmci": "PCMCI",
    "pcmci_plus": "PCMCI+ with eligible exploratory τ=0 triggers",
}

PLOT_BACKEND_LABELS = {
    "hmml": "HMML (baseline)",
    "pcmci": "PCMCI",
    "pcmci_plus": "PCMCI+",
}

THESIS_COLOURS = {
    "series": "#0072B2",
    "event": "#D55E00",
    "split": "#7B3294",
    "eligible": "#009E73",
    "muted": "#BDBDBD",
    "cause": "#0072B2",
}

SPLIT_COLOURS = (
    "#8B26AC",  # S1
    "#01BB89",  # S2
    "#B59504",  # S3
)

TRIGGER_SOURCE_LABELS = {
    "lagged": "Lagged",
    "contemporaneous": "Exploratory same-hour PCMCI+ link (τ=0)",
    "lagged_and_contemporaneous": "Lagged and exploratory same-hour",
}


def set_reporting_style() -> None:
    """Apply the same publication style used by the data-figure modules."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 450,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.titlesize": 10.7,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.60,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.42,
        "lines.linewidth": 0.85,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _flag_is_true(value) -> bool:
    """Interpret nullable scalar flags without treating NaN or 'False' as true."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _wrapped_label(value, width: int = 27) -> str:
    return textwrap.fill(
        str(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _save_figure(
    figure,
    *,
    filename: str | None,
    save_dir: str | Path | None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
) -> list[Path]:
    if filename is None or save_dir is None:
        return []

    output_directory = Path(save_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_formats = (formats,) if isinstance(formats, str) else tuple(formats)
    stem = Path(filename).stem
    saved_paths: list[Path] = []

    for file_format in output_formats:
        extension = str(file_format).lstrip(".")
        path = output_directory / f"{stem}.{extension}"
        figure.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.055,
            dpi=dpi,
            facecolor="white",
        )
        saved_paths.append(path)

    return saved_paths


def plot_effect_with_splits(
    dataframe: pd.DataFrame,
    effect: str,
    split_summary: pd.DataFrame,
    *,
    event_time: pd.Timestamp | None = None,
    title: str | None = None,
    effect_label: str | None = None,
    event_label: str = "Contextual event time",
    filename: str | None = None,
    save_dir: str | Path | None = None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
):
    """Plot the effect and every distinct partition supported by the I2 grid."""
    required = {"split_id", "split_time", "supported_min_I2_lengths"}
    missing = sorted(required - set(split_summary.columns))
    if missing:
        raise ValueError(f"split_summary is missing required columns: {missing}")
    if effect not in dataframe.columns:
        raise ValueError(f"Effect variable {effect!r} is absent.")
    if dataframe.empty:
        raise ValueError("dataframe is empty.")

    set_reporting_style()
    display_effect = effect if effect_label is None else str(effect_label)
    figure, axis = plt.subplots(figsize=(8.4, 3.75))
    axis.plot(
        dataframe.index,
        dataframe[effect],
        label=display_effect,
        color=THESIS_COLOURS["series"],
        linewidth=0.90,
        zorder=2,
    )

    valid = split_summary.dropna(subset=["split_time"]).copy()
    valid["split_time"] = pd.to_datetime(valid["split_time"], utc=True)
    for split_number, (split_time, group) in enumerate(
        valid.groupby("split_time", sort=True)
    ):
        split_ids = [
            str(value)
            for value in group["split_id"].dropna().drop_duplicates().tolist()
        ]
        if "in_causal_grid" in group.columns:
            causal_mask = group["in_causal_grid"].map(_flag_is_true)
        else:
            causal_mask = pd.Series(True, index=group.index)
        causal_values = sorted({
            int(value)
            for value in group.loc[causal_mask, "min_I2_length"].dropna()
        })
        diagnostic_values = sorted({
            int(value)
            for value in group.loc[~causal_mask, "min_I2_length"].dropna()
        })
        label_parts = []
        if causal_values:
            label_parts.append(
                r"minimum $I_2$: " + ", ".join(map(str, causal_values)) + " h"
            )
        if diagnostic_values:
            label_parts.append(
                "split diagnostic: "
                + ", ".join(map(str, diagnostic_values))
                + " h"
            )
        suffix = f" ({'; '.join(label_parts)})" if label_parts else ""
        axis.axvline(
            split_time,
            color=SPLIT_COLOURS[split_number % len(SPLIT_COLOURS)],
            linestyle="--",
            linewidth=0.95,
            alpha=0.88,
            zorder=3,
            label=f"{'/'.join(split_ids)}: selected split{suffix}",
        )

    if event_time is not None:
        axis.axvline(
            pd.Timestamp(event_time),
            color=THESIS_COLOURS["event"],
            linestyle=":",
            linewidth=1.10,
            zorder=4,
            label=event_label,
        )

    time_index = pd.DatetimeIndex(pd.to_datetime(dataframe.index, utc=True))
    span_days = max(
        float((time_index.max() - time_index.min()) / pd.Timedelta("1D")),
        1.0,
    )
    tick_interval = max(1, int(np.ceil(span_days / 6.0)))
    locator = mdates.DayLocator(interval=tick_interval, tz=mdates.UTC)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
    axis.set_title(
        title or f"{display_effect} and data-selected partitions",
        loc="left",
        pad=7,
        fontweight="semibold",
    )
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel("Reference-standardised value")
    axis.margins(x=0.01)
    axis.grid(False, axis="x")
    axis.grid(True, axis="y", alpha=0.16, linewidth=0.45)

    values = pd.to_numeric(dataframe[effect], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).any() and np.nanmin(values) <= 0 <= np.nanmax(values):
        axis.axhline(0.0, color="0.25", linewidth=0.45, alpha=0.22, zorder=0)

    handles, legend_labels = axis.get_legend_handles_labels()
    ncol = 2 if len(handles) <= 4 else 3
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=ncol,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.105, right=0.99, top=0.90, bottom=0.25)

    saved_paths = _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.show()
    return figure, axis, saved_paths


def mean_shift_role_table(
    dataframe: pd.DataFrame,
    effect: str,
    split_summary: pd.DataFrame,
    *,
    variable_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return the two mean-based quantities used for role screening."""
    required = {"split_id", "split_index"}
    missing = sorted(required - set(split_summary.columns))
    if missing:
        raise ValueError(f"split_summary is missing required columns: {missing}")
    if effect not in dataframe.columns:
        raise ValueError(f"Effect variable {effect!r} is absent.")

    causal = split_summary
    if "in_causal_grid" in split_summary.columns:
        causal = split_summary.loc[
            split_summary["in_causal_grid"].map(_flag_is_true)
        ]
    causal = (
        causal.dropna(subset=["split_id", "split_index"])
        .sort_values(["split_id", "split_index"], kind="stable")
        .drop_duplicates("split_id")
    )

    labels = {} if variable_labels is None else variable_labels
    variables = [column for column in dataframe.columns if column != effect]
    rows = []
    for _, split in causal.iterrows():
        split_index = int(split["split_index"])
        if not 0 < split_index < len(dataframe):
            raise ValueError(
                f"Invalid split index {split_index} for {len(dataframe)} rows."
            )
        mean_i1 = dataframe.iloc[:split_index][variables].mean()
        mean_i2 = dataframe.iloc[split_index:][variables].mean()
        for variable in variables:
            before = float(mean_i1[variable])
            after = float(mean_i2[variable])
            trigger_score = abs(after) - abs(before)
            tolerance = (
                100.0
                * np.finfo(float).eps
                * max(1.0, abs(after), abs(before))
            )
            rows.append({
                "Partition": str(split["split_id"]),
                "variable": variable,
                "Variable": readable_name(variable, labels),
                "Mean I1": before,
                "Mean I2": after,
                "Trigger score": trigger_score,
                "Trigger eligible": bool(trigger_score > tolerance),
                "Cause-shift magnitude": abs(after - before),
            })

    return pd.DataFrame(rows)


def plot_mean_shift_roles(
    role_table: pd.DataFrame,
    *,
    title: str = "Mean-based role-selection diagnostics",
    filename: str | None = None,
    save_dir: str | Path | None = None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
):
    """Plot trigger eligibility and cause-ranking magnitude by partition."""
    required = {
        "Partition",
        "Variable",
        "Trigger score",
        "Trigger eligible",
        "Cause-shift magnitude",
    }
    missing = sorted(required - set(role_table.columns))
    if missing:
        raise ValueError(f"role_table is missing required columns: {missing}")
    if role_table.empty:
        raise ValueError("role_table is empty.")

    set_reporting_style()
    partitions = role_table["Partition"].drop_duplicates().tolist()
    variable_order = (
        role_table.groupby("Variable", sort=False)["Cause-shift magnitude"]
        .max()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    wrapped_variables = [_wrapped_label(value) for value in variable_order]

    trigger_values = pd.to_numeric(role_table["Trigger score"], errors="coerce")
    trigger_limit = float(np.nanmax(np.abs(trigger_values.to_numpy(dtype=float))))
    trigger_limit = (
        1.08 * trigger_limit
        if np.isfinite(trigger_limit) and trigger_limit > 0
        else 1.0
    )
    cause_values = pd.to_numeric(
        role_table["Cause-shift magnitude"], errors="coerce"
    )
    cause_limit = float(np.nanmax(cause_values.to_numpy(dtype=float)))
    cause_limit = (
        1.08 * cause_limit
        if np.isfinite(cause_limit) and cause_limit > 0
        else 1.0
    )

    row_height = max(2.3, 0.36 * len(variable_order) + 0.75)
    figure_height = row_height * len(partitions) + 0.85
    figure, axes = plt.subplots(
        len(partitions),
        2,
        figsize=(8.8, figure_height),
        squeeze=False,
        sharey=True,
    )

    for row_index, partition in enumerate(partitions):
        subset = (
            role_table.loc[role_table["Partition"].eq(partition)]
            .set_index("Variable")
            .reindex(variable_order)
        )
        positions = np.arange(len(subset))
        trigger_flags = subset["Trigger eligible"].map(_flag_is_true)
        trigger_colours = np.where(
            trigger_flags,
            THESIS_COLOURS["eligible"],
            THESIS_COLOURS["muted"],
        )

        trigger_axis, cause_axis = axes[row_index]
        trigger_axis.barh(
            positions,
            subset["Trigger score"],
            color=trigger_colours,
            edgecolor="white",
            linewidth=0.35,
            height=0.70,
        )
        trigger_axis.axvline(0.0, color="0.20", linewidth=0.70, zorder=3)
        trigger_axis.set_xlim(-trigger_limit, trigger_limit)
        trigger_axis.set_yticks(positions, wrapped_variables)
        trigger_axis.set_title(
            f"{partition} — trigger eligibility",
            loc="left",
            pad=5,
        )

        cause_axis.barh(
            positions,
            subset["Cause-shift magnitude"],
            color=THESIS_COLOURS["cause"],
            edgecolor="white",
            linewidth=0.35,
            height=0.70,
        )
        cause_axis.set_xlim(0.0, cause_limit)
        cause_axis.set_title(
            f"{partition} — cause ranking",
            loc="left",
            pad=5,
        )
        cause_axis.tick_params(axis="y", labelleft=False)

        for axis in (trigger_axis, cause_axis):
            axis.set_axisbelow(True)
            axis.grid(False, axis="y")
            axis.grid(True, axis="x", alpha=0.16, linewidth=0.45)

        if row_index == len(partitions) - 1:
            trigger_axis.set_xlabel(
                r"Trigger score: $|\bar{x}_{I_2}|-|\bar{x}_{I_1}|$"
            )
            cause_axis.set_xlabel(
                r"Cause-shift magnitude: $|\bar{x}_{I_2}-\bar{x}_{I_1}|$"
            )

    axes[0, 0].invert_yaxis()

    legend = [
        Patch(
            facecolor=THESIS_COLOURS["eligible"],
            edgecolor="none",
            label="Trigger-eligible (score > 0)",
        ),
        Patch(
            facecolor=THESIS_COLOURS["muted"],
            edgecolor="none",
            label="Not trigger-eligible",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
    )
    figure.suptitle(title, y=0.995, fontsize=11.0, fontweight="semibold")
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.965), h_pad=1.25, w_pad=1.15)
    saved_paths = _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.show()
    return figure, axes, saved_paths


def show_table(frame: pd.DataFrame, caption: str | None = None) -> None:
    from IPython.display import display

    styler = (
        frame.style
        .hide(axis="index")
        .set_properties(**{
            "text-align": "left",
            "vertical-align": "top",
            "white-space": "normal",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "600"),
                    ("border-bottom", "1px solid #999"),
                ],
            },
            {"selector": "td", "props": [("padding", "5px 9px")]},
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-weight", "600"),
                    ("padding-bottom", "6px"),
                ],
            },
        ])
    )
    if caption:
        styler = styler.set_caption(caption)
    display(styler)


def readable_name(value, variable_labels: Mapping[str, str] | None = None):
    if value is None:
        return value
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return value
    labels = {} if variable_labels is None else variable_labels
    return labels.get(str(value), str(value).removesuffix("_scaled"))


def split_display_table(split_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one concise row per tested minimum-I2 value."""
    columns = [
        "min_I2_length",
        "in_causal_grid",
        "split_id",
        "split_time",
        "I1_length",
        "I2_length",
        "split_score",
        "boundary_split",
        "distance_to_event",
    ]
    return (
        split_summary[columns]
        .rename(columns={
            "min_I2_length": "Minimum I2 (h)",
            "in_causal_grid": "Causal grid",
            "split_id": "Partition",
            "split_time": "Split time",
            "I1_length": "I1 (h)",
            "I2_length": "I2 (h)",
            "split_score": "Split score",
            "boundary_split": "Boundary split",
            "distance_to_event": "Offset from event",
        })
        .reset_index(drop=True)
    )


def _has_items(value) -> bool:
    return isinstance(value, (list, tuple, set, dict)) and len(value) > 0


def _longest_consecutive_run(values) -> int:
    values = sorted({int(value) for value in values})
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _warning_details(
    caught: Sequence[warnings.WarningMessage],
) -> tuple[int, list[str], list[str]]:
    messages = list(dict.fromkeys(str(item.message) for item in caught))
    categories = list(dict.fromkeys(item.category.__name__ for item in caught))
    return len(caught), messages, categories


def _attach_metadata_columns(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Attach scalar or sequence-valued run metadata to every row safely."""
    output = dataframe.copy()
    for column, value in metadata.items():
        if pd.api.types.is_scalar(value):
            output[column] = value
        else:
            output[column] = pd.Series(
                [value] * len(output),
                index=output.index,
                dtype=object,
            )
    return output


def add_casewide_bh_adjustment(
    diagnostics: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add separate case-wide BH audits for every moderation specification."""
    output = diagnostics.copy()

    specifications = []
    if "p_value" in output.columns:
        specifications.append(
            ("p_value", "bh_adjusted_p_value", "accepted_after_bh")
        )
    if "hierarchical_ols_p_value" in output.columns:
        specifications.append((
            "hierarchical_ols_p_value",
            "hierarchical_ols_bh_adjusted_p_value",
            "hierarchical_ols_accepted_after_bh",
        ))
    for column in output.columns:
        if (
            column.startswith("hac_")
            and column.endswith("_p_value")
            and "_bh_adjusted_" not in column
        ):
            prefix = column.removesuffix("_p_value")
            specifications.append((
                column,
                f"{prefix}_bh_adjusted_p_value",
                f"{prefix}_accepted_after_bh",
            ))

    for p_column, adjusted_column, decision_column in specifications:
        output[adjusted_column] = np.nan
        output[decision_column] = False
        if output.empty:
            continue

        p_values = pd.to_numeric(output[p_column], errors="coerce")
        valid_index = p_values.index[p_values.notna() & np.isfinite(p_values)]
        if len(valid_index) == 0:
            continue

        ordered = p_values.loc[valid_index].sort_values(kind="stable")
        m = len(ordered)
        raw_adjusted = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        output.loc[ordered.index, adjusted_column] = adjusted
        output.loc[ordered.index, decision_column] = adjusted <= float(alpha)

    return output


def run_unique_split_grid(
    df_model: pd.DataFrame,
    df_mean: pd.DataFrame,
    workflow_template,
    split_summary: pd.DataFrame,
    *,
    run_one: Callable,
    run_specs: Sequence[Mapping[str, object]],
    lags: Sequence[int],
    cond_ind_test: str,
    metadata: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run each unique split × backend × maximum-lag-order configuration once.

    Minimum-I2 values that induce the same I1/I2 partition are recorded as
    support metadata rather than treated as independent causal runs.
    """
    required = {
        "split_id",
        "representative_min_I2_length",
        "supported_min_I2_lengths",
        "n_supporting_min_I2_values",
        "split_time",
        "I1_length",
        "I2_length",
    }
    missing = sorted(required - set(split_summary.columns))
    if missing:
        raise ValueError(f"split_summary is missing required columns: {missing}")

    metadata = {} if metadata is None else dict(metadata)
    rows: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []
    causal_split_summary = split_summary.copy()
    if "in_causal_grid" in split_summary.columns:
        causal_split_summary = split_summary.loc[
            split_summary["in_causal_grid"].map(_flag_is_true)
        ].copy()
    unique_splits = (
        causal_split_summary
        .sort_values(["split_id", "min_I2_length"], kind="stable")
        .drop_duplicates("split_id")
    )

    def nullable_int(value):
        return None if pd.isna(value) else int(value)

    for _, split in unique_splits.iterrows():
        current_workflow = replace(
            workflow_template,
            min_I2_length=int(split["representative_min_I2_length"]),
        )

        split_metadata = {
            "split_id": split["split_id"],
            "representative_min_I2_length": int(
                split["representative_min_I2_length"]
            ),
            "supported_min_I2_lengths": list(split["supported_min_I2_lengths"]),
            "n_supporting_min_I2_values": int(split["n_supporting_min_I2_values"]),
            "split_timestamp_expected": split["split_time"],
            "I1_length_expected": nullable_int(split["I1_length"]),
            "I2_length_expected": nullable_int(split["I2_length"]),
        }

        for spec in run_specs:
            run_name = str(spec["run"])
            backend = str(spec["backend"])
            for lag in lags:
                common = {
                    "experiment_type": "unique_split_backend_max_lag_grid",
                    **split_metadata,
                    "run": run_name,
                    "backend": backend,
                    "lag": int(lag),
                    **metadata,
                }
                caught: list[warnings.WarningMessage] = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result, diagnostics = run_one(
                            df_model,
                            df_mean,
                            current_workflow,
                            run_name=run_name,
                            backend=backend,
                            lag=int(lag),
                            cond_ind_test=spec.get("cond_ind_test", cond_ind_test),
                            use_contemporaneous_triggers=spec.get(
                                "use_contemporaneous_triggers"
                            ),
                        )

                    warning_count, warning_messages, warning_categories = _warning_details(caught)
                    actual_split = result.get("split_timestamp")
                    expected_split = split["split_time"]
                    split_mismatch = (
                        pd.isna(expected_split) and actual_split is not None
                    ) or (
                        pd.notna(expected_split)
                        and (
                            actual_split is None
                            or pd.Timestamp(actual_split) != pd.Timestamp(expected_split)
                        )
                    )
                    if split_mismatch:
                        raise RuntimeError(
                            "Representative minimum-I2 value reproduced a different split: "
                            f"expected {expected_split}, got {actual_split}."
                        )

                    success_row = {
                        **common,
                        "split_timestamp": actual_split,
                        "split_score": result.get("split_score"),
                        "boundary_split": result.get("boundary_split"),
                        "I1_length": result.get("I1_length"),
                        "I2_length": result.get("I2_length"),
                        "B_1": result.get("B_1"),
                        "B_2": result.get("B_2"),
                        "trigger_candidates": result.get("T_candidates"),
                        "T_candidates_lagged": result.get("T_candidates_lagged"),
                        "T_candidates_contemporaneous": result.get(
                            "T_candidates_contemporaneous"
                        ),
                        "accepted_triggers": result.get("T"),
                        "causes": result.get("C"),
                        "pairs": result.get("pairs"),
                        "n_contemporaneous_links": len(
                            result.get("contemporaneous_links", {})
                        ),
                        "n_diagnostics": len(diagnostics),
                        "stop_reason": result.get("stop_reason"),
                        "warning_count": warning_count,
                        "warning_scope": (
                            "complete backend run, including parent search"
                            if warning_count
                            else None
                        ),
                        "warning_categories": warning_categories,
                        "warning_messages": warning_messages,
                        "error": None,
                    }

                    diagnostic_frame = None
                    if not diagnostics.empty:
                        diagnostic_frame = diagnostics.assign(
                            experiment_type=common["experiment_type"],
                            split_id=split["split_id"],
                            representative_min_I2_length=int(
                                split["representative_min_I2_length"]
                            ),
                            supported_min_I2_lengths=[
                                list(split["supported_min_I2_lengths"])
                            ] * len(diagnostics),
                            n_supporting_min_I2_values=int(
                                split["n_supporting_min_I2_values"]
                            ),
                            split_timestamp=actual_split,
                        )
                        diagnostic_frame = _attach_metadata_columns(
                            diagnostic_frame,
                            metadata,
                        )

                    # Commit a successful configuration only after its
                    # diagnostics have also been prepared. This guarantees
                    # exactly one grid row even if post-processing fails.
                    rows.append(success_row)
                    if diagnostic_frame is not None:
                        diagnostic_frames.append(diagnostic_frame)
                except Exception as exc:
                    warning_count, warning_messages, warning_categories = _warning_details(caught)
                    rows.append({
                        **common,
                        "warning_count": warning_count,
                        "warning_scope": (
                            "complete backend run, including parent search"
                            if warning_count
                            else None
                        ),
                        "warning_categories": warning_categories,
                        "warning_messages": warning_messages,
                        "error": str(exc),
                    })

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, pd.DataFrame()

    key_columns = ["split_id", "run", "backend", "lag"]
    duplicate_mask = grid.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            grid.loc[duplicate_mask, key_columns]
            .drop_duplicates()
            .to_dict("records")
        )
        raise RuntimeError(
            "Duplicate unique-grid configurations were produced: "
            f"{duplicate_keys[:5]}"
        )

    expected_rows = len(unique_splits) * len(run_specs) * len(lags)
    if len(grid) != expected_rows:
        raise RuntimeError(
            "Incomplete unique-partition grid inside run_unique_split_grid: "
            f"expected {expected_rows} rows, received {len(grid)}."
        )

    grid["effective_I2_rows"] = (
        pd.to_numeric(grid.get("I2_length"), errors="coerce")
        - pd.to_numeric(grid["lag"], errors="coerce")
    )
    backend_order = {
        str(spec["backend"]): position for position, spec in enumerate(run_specs)
    }
    grid["_backend_order"] = grid["backend"].map(backend_order)
    grid = (
        grid
        .sort_values(["split_id", "_backend_order", "lag"], kind="stable")
        .drop(columns="_backend_order")
        .reset_index(drop=True)
    )

    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame()
    )
    if not diagnostics.empty:
        diagnostics = add_casewide_bh_adjustment(
            diagnostics,
            alpha=float(workflow_template.alpha),
        )
        diagnostics["_backend_order"] = diagnostics["backend"].map(backend_order)
        diagnostics = (
            diagnostics
            .sort_values(
                ["split_id", "_backend_order", "lag", "accepted", "cause", "trigger"],
                ascending=[True, True, True, False, True, True],
                kind="stable",
            )
            .drop(columns="_backend_order")
            .reset_index(drop=True)
        )

    return grid, diagnostics


def backend_split_summary(
    grid: pd.DataFrame,
    *,
    lag_count: int,
    diagnostics: pd.DataFrame | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarise each backend separately within every unique partition."""
    columns = [
        "Partition",
        "Minimum-I2 support",
        "Backend",
        "Successful runs",
        "Post-split parent lags",
        "Trigger-candidate lags",
        "Fitted-moderation lags",
        "Nominally supported pair lags",
        "τ=0-link lags",
        "Warning runs",
        "Result",
    ]
    if grid.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (split_id, backend), group in grid.groupby(
        ["split_id", "backend"], sort=False, dropna=False
    ):
        successful = group.loc[group["error"].isna()].copy()
        empty = pd.Series([[] for _ in range(len(successful))], index=successful.index)
        zero = pd.Series(0, index=successful.index, dtype=float)
        n_b2 = int(successful.get("B_2", empty).apply(_has_items).sum())
        n_candidates = int(
            successful.get("trigger_candidates", empty).apply(_has_items).sum()
        )
        n_pair_lags = int(successful.get("pairs", empty).apply(_has_items).sum())
        n_tau0 = int(
            successful.get("n_contemporaneous_links", zero).fillna(0).gt(0).sum()
        )
        n_warning_runs = int(successful.get("warning_count", zero).fillna(0).gt(0).sum())

        n_fitted_lags = 0
        if diagnostics is not None and not diagnostics.empty:
            matched = diagnostics.loc[
                diagnostics.get("split_id", pd.Series(dtype=object)).eq(split_id)
                & diagnostics.get("backend", pd.Series(dtype=object)).eq(backend)
            ].copy()
            if not matched.empty:
                fitted = pd.to_numeric(matched.get("p_value"), errors="coerce").notna()
                n_fitted_lags = int(
                    pd.to_numeric(matched.loc[fitted, "lag"], errors="coerce").nunique()
                )
                raw_accepted = matched.get(
                    "accepted", pd.Series(False, index=matched.index)
                ).map(_flag_is_true)
                n_pair_lags = int(
                    pd.to_numeric(
                        matched.loc[raw_accepted, "lag"], errors="coerce"
                    ).nunique()
                )

        if n_pair_lags:
            result = (
                "Nominally supported pair(s) at "
                f"{n_pair_lags} maximum-lag order(s)"
            )
        elif n_fitted_lags:
            result = (
                "Moderation fitted at "
                f"{n_fitted_lags} maximum-lag order(s); none accepted"
            )
        elif n_candidates:
            result = "Candidates found; no evaluable moderation model"
        elif n_b2:
            result = "Post-split parents found; no trigger candidates"
        else:
            result = "No post-split parent structure"

        support = group["supported_min_I2_lengths"].iloc[0]
        rows.append({
            "Partition": split_id,
            "Minimum-I2 support": str(support),
            "Backend": backend_labels.get(str(backend), str(backend)),
            "Successful runs": f"{len(successful)}/{lag_count}",
            "Post-split parent lags": f"{n_b2}/{lag_count}",
            "Trigger-candidate lags": f"{n_candidates}/{lag_count}",
            "Fitted-moderation lags": f"{n_fitted_lags}/{lag_count}",
            "Nominally supported pair lags": f"{n_pair_lags}/{lag_count}",
            "τ=0-link lags": f"{n_tau0}/{lag_count}",
            "Warning runs": f"{n_warning_runs}/{lag_count}",
            "Result": result,
        })

    return pd.DataFrame(rows).reindex(columns=columns)


def pair_split_lag_summary(
    diagnostics: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    lags: Sequence[int],
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarise nominal and BH-supported pairs without duplicate splits."""
    columns = [
        "Backend",
        "Trigger source",
        "Cause",
        "Trigger",
        "Nominally supported cells",
        "BH-supported cells",
        "Partitions",
        "Partition support",
        "Minimum-I2 values",
        "Maximum lag orders",
        "Lag-order support",
        "Longest adjacent lag run",
    ]
    if diagnostics.empty or grid.empty:
        return pd.DataFrame(columns=columns)

    accepted = diagnostics.loc[
        diagnostics["accepted"].map(_flag_is_true)
    ].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns)

    key = ["split_id", "backend", "lag", "cause", "trigger", "trigger_source"]
    accepted = accepted.drop_duplicates(key)
    summary = (
        accepted
        .groupby(["backend", "cause", "trigger", "trigger_source"], as_index=False)
        .agg(
            accepted_cells=("lag", "size"),
            split_ids=("split_id", lambda values: sorted(set(values))),
            lag_orders=("lag", lambda values: sorted({int(value) for value in values})),
            bh_cells=(
                "accepted_after_bh",
                lambda values: int(pd.Series(values).map(_flag_is_true).sum()),
            ),
        )
    )
    support_by_split = (
        grid[["split_id", "supported_min_I2_lengths"]]
        .drop_duplicates("split_id")
        .set_index("split_id")["supported_min_I2_lengths"]
        .to_dict()
    )
    summary["min_i2_values"] = summary["split_ids"].map(
        lambda split_ids: sorted({
            int(value)
            for split_id in split_ids
            for value in support_by_split.get(split_id, [])
        })
    )
    summary["n_splits"] = summary["split_ids"].map(len)
    summary["n_lags"] = summary["lag_orders"].map(len)
    summary["longest"] = summary["lag_orders"].map(_longest_consecutive_run)
    summary = summary.sort_values(
        ["accepted_cells", "n_splits", "n_lags", "longest", "backend", "cause", "trigger"],
        ascending=[False, False, False, False, True, True, True],
        kind="stable",
    )

    labels = {} if variable_labels is None else variable_labels
    n_splits_total = int(grid["split_id"].nunique())
    total_cells = n_splits_total * len(tuple(lags))

    return pd.DataFrame({
        "Backend": summary["backend"].map(
            lambda value: backend_labels.get(str(value), str(value))
        ),
        "Trigger source": summary["trigger_source"].map(
            lambda value: TRIGGER_SOURCE_LABELS.get(str(value), str(value))
        ),
        "Cause": summary["cause"].map(lambda value: readable_name(value, labels)),
        "Trigger": summary["trigger"].map(lambda value: readable_name(value, labels)),
        "Nominally supported cells": (
            summary["accepted_cells"].astype(int).astype(str) + f"/{total_cells}"
        ),
        "BH-supported cells": summary["bh_cells"].astype(int).astype(str) + f"/{total_cells}",
        "Partitions": summary["split_ids"].astype(str),
        "Partition support": summary["n_splits"].astype(int).astype(str) + f"/{n_splits_total}",
        "Minimum-I2 values": summary["min_i2_values"].astype(str),
        "Maximum lag orders": summary["lag_orders"].astype(str),
        "Lag-order support": summary["n_lags"].astype(int).astype(str) + f"/{len(tuple(lags))}",
        "Longest adjacent lag run": summary["longest"].astype(int),
    }).reset_index(drop=True)


def plot_pair_stability_heatmap(
    diagnostics: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    lags: Sequence[int],
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
    title: str = "Ordered-pair stability across maximum-lag orders",
    filename: str | None = None,
    save_dir: str | Path | None = None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
):
    """Plot raw and BH-supported ordered-pair decisions by split and backend."""
    required_diagnostics = {
        "split_id",
        "backend",
        "lag",
        "cause",
        "trigger",
        "trigger_source",
        "accepted",
        "accepted_after_bh",
        "p_value",
        "gamma_2",
    }
    missing = sorted(required_diagnostics - set(diagnostics.columns))
    if missing:
        raise ValueError(f"diagnostics is missing required columns: {missing}")
    if grid.empty:
        raise ValueError("grid is empty.")

    set_reporting_style()
    valid = diagnostics.dropna(subset=["cause", "trigger"]).copy()
    accepted = valid.loc[valid["accepted"].map(_flag_is_true)].copy()
    if accepted.empty:
        raise ValueError("No accepted ordered pair is available to plot.")

    pair_columns = ["cause", "trigger", "trigger_source"]
    pairs = (
        accepted.groupby(pair_columns, dropna=False)
        .agg(
            accepted_cells=("accepted", "size"),
            bh_cells=(
                "accepted_after_bh",
                lambda values: sum(map(_flag_is_true, values)),
            ),
        )
        .reset_index()
    )
    labels = {} if variable_labels is None else variable_labels

    def pair_label(row) -> str:
        cause = _wrapped_label(readable_name(row["cause"], labels), width=24)
        trigger = _wrapped_label(readable_name(row["trigger"], labels), width=24)
        source = str(row["trigger_source"])
        if source == "contemporaneous":
            source_note = "\nsource: exploratory τ=0"
        elif source == "lagged_and_contemporaneous":
            source_note = "\nsource: lagged + exploratory τ=0"
        else:
            source_note = ""
        return f"C: {cause}\nT: {trigger}{source_note}"

    pairs["pair_label"] = pairs.apply(pair_label, axis=1)
    pairs = pairs.sort_values(
        ["bh_cells", "accepted_cells", "pair_label"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    lag_values = [int(value) for value in lags]
    if not lag_values:
        raise ValueError("lags is empty.")
    lag_positions = {lag: position for position, lag in enumerate(lag_values)}
    partitions = grid["split_id"].dropna().drop_duplicates().tolist()
    observed_backends = grid["backend"].dropna().astype(str).drop_duplicates().tolist()
    backends = [value for value in backend_labels if value in observed_backends]
    backends.extend(value for value in observed_backends if value not in backends)

    status_labels = {
        0: "Not selected",
        1: "Selected; not evaluable",
        2: "Fitted; H₀ not rejected",
        3: "Nominal F-test only",
        4: "BH-supported",
    }
    colours = ["#F7F7F7", "#C7C7C7", "#A6CEE3", "#E69F00", "#009E73"]
    colour_map = ListedColormap(colours)
    normaliser = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), colour_map.N)

    panel_height = max(2.75, 0.40 * len(pairs) + 0.55)
    footer_height = 0.82
    title_height = 0.28
    figure_height = title_height + panel_height * len(partitions) + footer_height
    figure_width = max(8.8, 2.65 * len(backends) + 1.85)
    figure, axes = plt.subplots(
        len(partitions),
        len(backends),
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_index, partition in enumerate(partitions):
        for column_index, backend in enumerate(backends):
            axis = axes[row_index, column_index]
            matrix = np.zeros((len(pairs), len(lag_values)), dtype=int)
            signs = np.full(matrix.shape, "", dtype=object)
            facet = valid.loc[
                valid["split_id"].eq(partition)
                & valid["backend"].astype(str).eq(backend)
            ]

            for pair_index, pair in pairs.iterrows():
                pair_rows = facet.loc[
                    facet["cause"].eq(pair["cause"])
                    & facet["trigger"].eq(pair["trigger"])
                    & facet["trigger_source"].eq(pair["trigger_source"])
                ]
                for _, diagnostic in pair_rows.iterrows():
                    lag = int(diagnostic["lag"])
                    if lag not in lag_positions:
                        continue
                    lag_index = lag_positions[lag]
                    p_value = pd.to_numeric(
                        pd.Series([diagnostic.get("p_value")]), errors="coerce"
                    ).iloc[0]
                    if not np.isfinite(p_value):
                        status = 1
                    elif _flag_is_true(diagnostic.get("accepted_after_bh", False)):
                        status = 4
                    elif _flag_is_true(diagnostic.get("accepted", False)):
                        status = 3
                    else:
                        status = 2
                    current_status = matrix[pair_index, lag_index]
                    gamma = pd.to_numeric(
                        pd.Series([diagnostic.get("gamma_2")]), errors="coerce"
                    ).iloc[0]
                    if status >= current_status:
                        matrix[pair_index, lag_index] = status
                    if status >= current_status and np.isfinite(gamma):
                        signs[pair_index, lag_index] = "+" if gamma >= 0 else "−"

            axis.imshow(
                matrix,
                cmap=colour_map,
                norm=normaliser,
                aspect="auto",
                interpolation="nearest",
            )
            axis.grid(False, which="major")
            axis.set_xticks(
                np.arange(-0.5, len(lag_values), 1.0),
                minor=True,
            )
            axis.set_yticks(
                np.arange(-0.5, len(pairs), 1.0),
                minor=True,
            )
            axis.grid(
                True,
                which="minor",
                color="white",
                linewidth=0.65,
            )
            axis.tick_params(which="minor", bottom=False, left=False)
            for pair_index in range(matrix.shape[0]):
                for lag_index in range(matrix.shape[1]):
                    if signs[pair_index, lag_index]:
                        status = matrix[pair_index, lag_index]
                        axis.text(
                            lag_index,
                            pair_index,
                            signs[pair_index, lag_index],
                            ha="center",
                            va="center",
                            fontsize=8.2,
                            fontweight="semibold",
                            color="white" if status == 4 else "0.12",
                        )

            facet_has_accepted = facet["accepted"].map(_flag_is_true).any()
            if not facet_has_accepted:
                axis.text(
                    0.5,
                    0.5,
                    "No accepted\nordered pair",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color="0.35",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.80",
                        "linewidth": 0.45,
                        "alpha": 0.92,
                        "pad": 3.0,
                    },
                )

            axis.set_title(
                f"{partition} · {PLOT_BACKEND_LABELS.get(backend, backend)}",
                pad=5,
                fontsize=10.0,
            )
            axis.set_xticks(np.arange(len(lag_values)), lag_values)
            axis.tick_params(
                axis="x",
                labelbottom=row_index == len(partitions) - 1,
                length=2.8,
                width=0.55,
                pad=3,
            )
            if row_index == len(partitions) - 1:
                axis.set_xlabel(r"Maximum lag order, $d$ (h)", labelpad=5)
            axis.set_yticks(np.arange(len(pairs)))
            if column_index == 0:
                axis.set_yticklabels(pairs["pair_label"])
                axis.tick_params(axis="y", labelleft=True, labelsize=8.6, pad=4)
            else:
                axis.tick_params(axis="y", labelleft=False)

    legend = [
        Patch(facecolor=colours[index], edgecolor="0.78", linewidth=0.55, label=label)
        for index, label in status_labels.items()
    ]
    legend_y = 0.20 / figure_height
    note_y = 0.045 / figure_height
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )
    figure.suptitle(
        title,
        y=1.0 - 0.05 / figure_height,
        fontsize=11.2,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        note_y,
        (
            "+/− gives the sign of γ₂."
        ),
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="0.28",
    )
    figure.tight_layout(
        rect=(
            0.0,
            footer_height / figure_height,
            1.0,
            1.0 - title_height / figure_height,
        ),
        h_pad=1.15,
        w_pad=0.65,
    )
    saved_paths = _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.show()
    return figure, axes, saved_paths


def _moderation_specifications(
    diagnostics: pd.DataFrame,
    hac_lags: Sequence[int],
) -> list[dict[str, str]]:
    specifications = [
        {
            "label": "Primary model",
            "p_value": "p_value",
            "accepted": "accepted",
            "accepted_after_bh": "accepted_after_bh",
            "gamma_2": "gamma_2",
        },
        {
            "label": "Hierarchical OLS",
            "p_value": "hierarchical_ols_p_value",
            "accepted": "hierarchical_ols_accepted",
            "accepted_after_bh": "hierarchical_ols_accepted_after_bh",
            "gamma_2": "hierarchical_ols_gamma_2",
        },
    ]
    specifications.extend({
        "label": f"HAC-{int(lag)} h",
        "p_value": f"hac_{int(lag)}_p_value",
        "accepted": f"hac_{int(lag)}_accepted",
        "accepted_after_bh": f"hac_{int(lag)}_accepted_after_bh",
        "gamma_2": "hierarchical_ols_gamma_2",
    } for lag in hac_lags)

    return [
        specification
        for specification in specifications
        if specification["p_value"] in diagnostics.columns
    ]


def moderation_summary(
    diagnostics: pd.DataFrame,
    *,
    hac_lags: Sequence[int] = (6, 12, 24),
) -> pd.DataFrame:
    """Summarise primary, hierarchical, and HAC moderation decisions."""
    columns = [
        "Specification",
        "Fitted tests",
        "Nominally supported",
        "BH-supported",
        "Retained among original model support",
        "Same interaction sign as the original model",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)

    specifications = _moderation_specifications(diagnostics, hac_lags)
    if not specifications:
        return pd.DataFrame(columns=columns)

    primary_p = pd.to_numeric(
        diagnostics.get("p_value"), errors="coerce"
    )
    primary_supported = (
        primary_p.notna()
        & np.isfinite(primary_p)
        & diagnostics.get(
            "accepted", pd.Series(False, index=diagnostics.index)
        ).map(_flag_is_true)
    )
    primary_gamma = pd.to_numeric(
        diagnostics.get("gamma_2"), errors="coerce"
    )
    primary_count = int(primary_supported.sum())

    rows = []
    for specification in specifications:
        p_values = pd.to_numeric(
            diagnostics[specification["p_value"]], errors="coerce"
        )
        fitted = p_values.notna() & np.isfinite(p_values)
        nominal = fitted & diagnostics.get(
            specification["accepted"],
            pd.Series(False, index=diagnostics.index),
        ).map(_flag_is_true)
        adjusted = fitted & diagnostics.get(
            specification["accepted_after_bh"],
            pd.Series(False, index=diagnostics.index),
        ).map(_flag_is_true)

        retained_count = int((primary_supported & nominal).sum())
        gamma = pd.to_numeric(
            diagnostics.get(specification["gamma_2"]), errors="coerce"
        )
        comparable_sign = (
            primary_supported
            & primary_gamma.notna()
            & np.isfinite(primary_gamma)
            & gamma.notna()
            & np.isfinite(gamma)
        )
        same_sign = (
            np.sign(primary_gamma.loc[comparable_sign])
            == np.sign(gamma.loc[comparable_sign])
        )

        rows.append({
            "Specification": specification["label"],
            "Fitted tests": int(fitted.sum()),
            "Nominally supported": int(nominal.sum()),
            "BH-supported": int(adjusted.sum()),
            "Retained among original model support": (
                f"{retained_count}/{primary_count}"
            ),
            "Same interaction sign as the original model": (
                f"{int(same_sign.sum())}/{int(comparable_sign.sum())}"
            ),
        })

    return pd.DataFrame(rows).reindex(columns=columns)


def plot_moderation_heatmap(
    diagnostics: pd.DataFrame,
    *,
    hac_lags: Sequence[int] = (6, 12, 24),
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = PLOT_BACKEND_LABELS,
    primary_supported_only: bool = True,
    title: str = "Hierarchical--HAC of moderation support",
    filename: str | None = None,
    save_dir: str | Path | None = None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
):
    """Plot specification sensitivity for each evaluated classification."""
    required = {
        "split_id",
        "backend",
        "lag",
        "cause",
        "trigger",
        "p_value",
        "accepted",
        "gamma_2",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"diagnostics is missing required columns: {missing}")

    specifications = _moderation_specifications(diagnostics, hac_lags)
    if len(specifications) < 2:
        raise ValueError("Hierarchical--HAC diagnostics are not available.")

    rows = diagnostics.dropna(subset=["cause", "trigger"]).copy()
    rows = rows.loc[
        pd.to_numeric(rows["p_value"], errors="coerce").notna()
    ]
    if primary_supported_only:
        rows = rows.loc[rows["accepted"].map(_flag_is_true)]
    if rows.empty:
        raise ValueError("No moderation rows are available for the plot.")

    identity = ["split_id", "backend", "lag", "cause", "trigger"]
    rows = rows.drop_duplicates(identity, keep="first")
    backend_order = {name: index for index, name in enumerate(backend_labels)}
    rows["_backend_order"] = rows["backend"].map(backend_order).fillna(
        len(backend_order)
    )
    rows = rows.sort_values(
        ["cause", "trigger", "split_id", "_backend_order", "lag"],
        kind="stable",
    ).drop(columns="_backend_order").reset_index(drop=True)

    matrix = np.zeros((len(rows), len(specifications)), dtype=int)
    signs = np.full(matrix.shape, "", dtype=object)
    for row_index, (_, row) in enumerate(rows.iterrows()):
        for column_index, specification in enumerate(specifications):
            p_value = pd.to_numeric(
                pd.Series([row.get(specification["p_value"])]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(p_value):
                continue

            status = 1
            if _flag_is_true(row.get(specification["accepted"], False)):
                status = 2
            if _flag_is_true(
                row.get(specification["accepted_after_bh"], False)
            ):
                status = 3
            matrix[row_index, column_index] = status

            gamma = pd.to_numeric(
                pd.Series([row.get(specification["gamma_2"])]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(gamma):
                signs[row_index, column_index] = "+" if gamma >= 0 else "−"

    labels = {} if variable_labels is None else variable_labels
    row_labels = []
    for _, row in rows.iterrows():
        backend = backend_labels.get(str(row["backend"]), str(row["backend"]))
        cause = _wrapped_label(readable_name(row["cause"], labels), width=25)
        trigger = _wrapped_label(readable_name(row["trigger"], labels), width=25)
        row_labels.append(
            f"{row['split_id']} · {backend} · d={int(row['lag'])}\n"
            f"C: {cause}  |  T: {trigger}"
        )

    set_reporting_style()
    colours = ["#D9D9D9", "#A6CEE3", "#E69F00", "#009E73"]
    labels_by_status = {
        0: "Not evaluable",
        1: "Fitted; unsupported",
        2: "Nominal support only",
        3: "BH-supported",
    }
    colour_map = ListedColormap(colours)
    normaliser = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), colour_map.N)
    figure_height = max(4.0, 0.42 * len(rows) + 1.55)
    figure_width = max(8.2, 1.35 * len(specifications) + 4.7)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.imshow(
        matrix,
        cmap=colour_map,
        norm=normaliser,
        aspect="auto",
        interpolation="nearest",
    )
    axis.grid(False, which="major")
    axis.set_xticks(np.arange(len(specifications)))
    axis.set_xticklabels(
        [specification["label"] for specification in specifications],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(row_labels, fontsize=8.2)
    axis.set_xticks(np.arange(-0.5, len(specifications), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1.0), minor=True)
    axis.grid(True, which="minor", color="white", linewidth=0.70)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            if not signs[row_index, column_index]:
                continue
            axis.text(
                column_index,
                row_index,
                signs[row_index, column_index],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="semibold",
                color="white" if matrix[row_index, column_index] == 3 else "0.12",
            )

    legend = [
        Patch(
            facecolor=colours[index],
            edgecolor="0.78",
            linewidth=0.55,
            label=label,
        )
        for index, label in labels_by_status.items()
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        columnspacing=1.25,
        handlelength=1.6,
    )
    axis.set_title(title, loc="left", pad=8, fontweight="semibold")
    figure.text(
        0.5,
        0.003,
        "+/− gives the fitted interaction sign; HAC uses the hierarchical model.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="0.28",
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    saved_paths = _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.show()
    return figure, axis, saved_paths


def reference_criteria_by_lag(lag_references: pd.DataFrame) -> dict[int, str]:
    return (
        lag_references
        .groupby("selected_lag")["criterion"]
        .apply(lambda values: "/".join(sorted(values)))
        .to_dict()
    )


def _format_pairs(
    diagnostics: pd.DataFrame,
    variable_labels: Mapping[str, str] | None,
) -> str:
    if diagnostics.empty:
        return ""
    labels = {} if variable_labels is None else variable_labels
    values = []
    for _, row in diagnostics.iterrows():
        source = TRIGGER_SOURCE_LABELS.get(
            str(row.get("trigger_source")), str(row.get("trigger_source"))
        )
        values.append(
            f"Cause: {readable_name(row.get('cause'), labels)}; "
            f"trigger: {readable_name(row.get('trigger'), labels)} [{source}]"
        )
    return "; ".join(dict.fromkeys(values))


def automatic_lag_outcome_table(
    grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    lag_references: pd.DataFrame,
    *,
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Show what every backend returns at the automatically selected VAR orders."""
    columns = [
        "Criterion",
        "Selected d (h)",
        "Search boundary",
        "Partition",
        "Minimum-I2 support",
        "Backend",
        "Outcome",
    ]
    if grid.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, reference in lag_references.iterrows():
        lag = int(reference["selected_lag"])
        subset = grid.loc[grid["lag"].eq(lag)]
        for _, run in subset.iterrows():
            matched = diagnostics.loc[
                diagnostics.get("split_id", pd.Series(dtype=object)).eq(run["split_id"])
                & diagnostics.get("backend", pd.Series(dtype=object)).eq(run["backend"])
                & pd.to_numeric(diagnostics.get("lag"), errors="coerce").eq(lag)
            ] if not diagnostics.empty else pd.DataFrame()
            accepted = (
                matched.loc[
                    matched.get(
                        "accepted", pd.Series(False, index=matched.index)
                    ).map(_flag_is_true)
                ]
                if not matched.empty
                else pd.DataFrame()
            )

            if not accepted.empty:
                outcome = (
                    "Nominally supported: "
                    + _format_pairs(accepted, variable_labels)
                )
            elif not matched.empty and pd.to_numeric(matched.get("p_value"), errors="coerce").notna().any():
                minimum_p = pd.to_numeric(matched["p_value"], errors="coerce").min()
                outcome = f"Moderation tested; none accepted (minimum p={minimum_p:.3g})"
            elif not matched.empty:
                outcome = "Trigger candidates found; no evaluable moderation model"
            elif pd.notna(run.get("stop_reason")):
                outcome = str(run.get("stop_reason"))
            elif _has_items(run.get("trigger_candidates")):
                outcome = "Trigger candidates found; no evaluable moderation model"
            elif _has_items(run.get("B_2")):
                outcome = "Post-split parents found; no trigger candidates"
            else:
                outcome = "No usable post-split parent structure"

            rows.append({
                "Criterion": reference["criterion"],
                "Selected d (h)": lag,
                "Search boundary": _flag_is_true(
                    reference.get("search_boundary", False)
                ),
                "Partition": run["split_id"],
                "Minimum-I2 support": str(run["supported_min_I2_lengths"]),
                "Backend": backend_labels.get(str(run["backend"]), str(run["backend"])),
                "Outcome": outcome,
            })

    return pd.DataFrame(rows).reindex(columns=columns)


def warning_summary(grid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Partition",
        "Backend",
        "Runs with warnings",
        "Warning count",
        "Scope",
        "Messages",
    ]
    if grid.empty or "warning_count" not in grid.columns:
        return pd.DataFrame(columns=columns)
    warned = grid.loc[pd.to_numeric(grid["warning_count"], errors="coerce").fillna(0).gt(0)]
    if warned.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (split_id, backend), group in warned.groupby(["split_id", "backend"], sort=False):
        messages = []
        for value in group["warning_messages"]:
            if isinstance(value, list):
                messages.extend(value)
        scopes = [
            str(value)
            for value in group.get(
                "warning_scope", pd.Series(dtype=object)
            ).dropna()
        ]
        rows.append({
            "Partition": split_id,
            "Backend": BACKEND_LABELS.get(str(backend), str(backend)),
            "Runs with warnings": int(len(group)),
            "Warning count": int(pd.to_numeric(group["warning_count"], errors="coerce").sum()),
            "Scope": "; ".join(dict.fromkeys(scopes)),
            "Messages": "; ".join(dict.fromkeys(messages)),
        })
    return pd.DataFrame(rows).reindex(columns=columns)


def warning_pattern_summary(
    grid: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Locate warning patterns in lag space and relate them to fitted tests."""
    columns = [
        "Partition",
        "Backend",
        "Maximum lag orders",
        "Runs",
        "Warning count",
        "Scope",
        "Fitted moderation tests",
        "Nominal F-test support",
        "BH-supported",
        "Execution errors",
        "Warning pattern",
    ]
    if grid.empty or "warning_count" not in grid.columns:
        return pd.DataFrame(columns=columns)

    warned = grid.loc[
        pd.to_numeric(grid["warning_count"], errors="coerce").fillna(0).gt(0)
    ].copy()
    if warned.empty:
        return pd.DataFrame(columns=columns)

    def normalise_messages(value) -> tuple[str, ...]:
        if isinstance(value, (list, tuple, set)):
            return tuple(str(item) for item in value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return tuple(str(item) for item in parsed)
            return (value,)
        return (str(value),)

    warned["_warning_pattern"] = warned["warning_messages"].map(
        normalise_messages
    )
    rows = []
    for (split_id, backend, pattern), group in warned.groupby(
        ["split_id", "backend", "_warning_pattern"],
        sort=False,
        dropna=False,
    ):
        lag_values = sorted({int(value) for value in group["lag"]})
        matched = pd.DataFrame()
        if diagnostics is not None and not diagnostics.empty:
            matched = diagnostics.loc[
                diagnostics.get("split_id", pd.Series(dtype=object)).eq(split_id)
                & diagnostics.get("backend", pd.Series(dtype=object)).eq(backend)
                & pd.to_numeric(diagnostics.get("lag"), errors="coerce").isin(lag_values)
            ].copy()

        fitted = (
            pd.to_numeric(matched.get("p_value"), errors="coerce").notna()
            if not matched.empty
            else pd.Series(dtype=bool)
        )
        raw = (
            matched.get("accepted", pd.Series(False, index=matched.index))
            .map(_flag_is_true)
            if not matched.empty
            else pd.Series(dtype=bool)
        )
        adjusted = (
            matched.get(
                "accepted_after_bh",
                pd.Series(False, index=matched.index),
            )
            .map(_flag_is_true)
            if not matched.empty
            else pd.Series(dtype=bool)
        )
        rows.append({
            "Partition": split_id,
            "Backend": BACKEND_LABELS.get(str(backend), str(backend)),
            "Maximum lag orders": str(lag_values),
            "Runs": int(len(group)),
            "Warning count": int(
                pd.to_numeric(group["warning_count"], errors="coerce").sum()
            ),
            "Scope": "; ".join(dict.fromkeys(
                str(value)
                for value in group.get(
                    "warning_scope", pd.Series(dtype=object)
                ).dropna()
            )),
            "Fitted moderation tests": int(fitted.sum()),
            "Nominal F-test support": int(raw.sum()),
            "BH-supported": int(adjusted.sum()),
            "Execution errors": int(group["error"].notna().sum()),
            "Warning pattern": "; ".join(pattern),
        })

    return pd.DataFrame(rows).reindex(columns=columns)



def multiplicity_audit_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame([{
            "Fitted moderation tests": 0,
            "Nominal F-test support": 0,
            "BH-supported": 0,
        }])

    fitted = pd.to_numeric(
        diagnostics.get("p_value"), errors="coerce"
    ).notna()
    raw = diagnostics.get(
        "accepted", pd.Series(False, index=diagnostics.index)
    ).map(_flag_is_true)
    adjusted = diagnostics.get(
        "accepted_after_bh",
        pd.Series(False, index=diagnostics.index),
    ).map(_flag_is_true)
    return pd.DataFrame([{
        "Fitted moderation tests": int(fitted.sum()),
        "Nominal F-test support": int(raw.sum()),
        "BH-supported": int(adjusted.sum()),
    }])

def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype != object:
            continue
        output[column] = output[column].map(
            lambda value: (
                json.dumps(_json_ready(value), ensure_ascii=False, default=str)
                if isinstance(value, (list, tuple, set, dict))
                else value.isoformat() if isinstance(value, pd.Timestamp) else value
            )
        )
    return output


def build_summary_export(
    *,
    design: pd.DataFrame,
    lag_references: pd.DataFrame,
    automatic_lag_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    warning_table: pd.DataFrame,
    diagnostics: pd.DataFrame,
    errors: pd.DataFrame,
    role_shift_table: pd.DataFrame | None = None,
    lag_sensitivity_table: pd.DataFrame | None = None,
    sensitivity_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in design.iterrows():
        rows.append({"section": "design", "metric": row["Item"], "value": row["Value"]})

    for _, row in lag_references.iterrows():
        rows.append({
            "section": "automatic_lag_reference",
            "lag": int(row["selected_lag"]),
            "metric": str(row["criterion"]),
            "value": f"{int(row['selected_lag'])} h",
            "details": {
                "maximum_lag_tested": int(row["max_lags"]),
                "search_boundary": _flag_is_true(
                    row.get("search_boundary", False)
                ),
            },
        })

    for _, row in automatic_lag_summary.iterrows():
        rows.append({
            "section": "automatic_lag_outcome",
            "backend": row["Backend"],
            "partition": row["Partition"],
            "lag": int(row["Selected d (h)"]),
            "metric": row["Criterion"],
            "value": row["Outcome"],
            "details": {
                "minimum_I2_support": row["Minimum-I2 support"],
                "search_boundary": _flag_is_true(row["Search boundary"]),
            },
        })

    for _, row in split_summary.iterrows():
        rows.append({
            "section": "split_grid",
            "partition": row["split_id"],
            "metric": f"min_I2={int(row['min_I2_length'])} h",
            "value": row.get("split_time"),
            "details": {
                "I1_length": row.get("I1_length"),
                "I2_length": row.get("I2_length"),
                "split_score": row.get("split_score"),
                "boundary_split": row.get("boundary_split"),
                "distance_to_event": row.get("distance_to_event"),
            },
        })

    for _, row in backend_summary.iterrows():
        rows.append({
            "section": "backend_by_unique_split",
            "backend": row["Backend"],
            "partition": row["Partition"],
            "metric": "result",
            "value": row["Result"],
            "details": {
                key: row[key]
                for key in backend_summary.columns
                if key not in {"Backend", "Partition", "Result"}
            },
        })

    for _, row in pair_summary.iterrows():
        rows.append({
            "section": "pair_stability",
            "backend": row["Backend"],
            "cause": row["Cause"],
            "trigger": row["Trigger"],
            "trigger_source": row["Trigger source"],
            "metric": "nominal_split_lag_support",
            "value": row["Nominally supported cells"],
            "details": {
                key: row[key]
                for key in pair_summary.columns
                if key not in {
                    "Backend",
                    "Cause",
                    "Trigger",
                    "Trigger source",
                    "Nominally supported cells",
                }
            },
        })

    for _, row in warning_table.iterrows():
        rows.append({
            "section": "warnings",
            "backend": row["Backend"],
            "partition": row["Partition"],
            "metric": "warning_count",
            "value": int(row["Warning count"]),
            "details": {
                "runs_with_warnings": int(row["Runs with warnings"]),
                "scope": row.get("Scope"),
                "messages": row["Messages"],
            },
        })

    if not diagnostics.empty and "p_value" in diagnostics.columns:
        fitted = pd.to_numeric(diagnostics["p_value"], errors="coerce").notna()
        rows.extend([
            {
                "section": "multiplicity_audit",
                "metric": "fitted_moderation_tests",
                "value": int(fitted.sum()),
            },
            {
                "section": "multiplicity_audit",
                "metric": "nominal_F_supported_tests",
                "value": int(
                    diagnostics.get(
                        "accepted",
                        pd.Series(False, index=diagnostics.index),
                    )
                    .map(_flag_is_true)
                    .sum()
                ),
            },
            {
                "section": "multiplicity_audit",
                "metric": "BH_supported_tests",
                "value": int(
                    diagnostics.get(
                        "accepted_after_bh",
                        pd.Series(False, index=diagnostics.index),
                    )
                    .map(_flag_is_true)
                    .sum()
                ),
            },
        ])

    if role_shift_table is not None and not role_shift_table.empty:
        for _, row in role_shift_table.iterrows():
            rows.append({
                "section": "mean_shift_role_diagnostic",
                "partition": row["Partition"],
                "metric": row["variable"],
                "value": row["Trigger score"],
                "details": {
                    "variable_label": row["Variable"],
                    "mean_I1": row["Mean I1"],
                    "mean_I2": row["Mean I2"],
                    "trigger_eligible": _flag_is_true(row["Trigger eligible"]),
                    "cause_shift_magnitude": row["Cause-shift magnitude"],
                },
            })

    if lag_sensitivity_table is not None and not lag_sensitivity_table.empty:
        for _, row in lag_sensitivity_table.iterrows():
            rows.append({
                "section": "lag_order_sensitivity",
                "lag": int(row["selected_lag"]),
                "metric": row["criterion"],
                "value": f"{int(row['selected_lag'])} h",
                "details": {
                    "maximum_lag_tested": int(row["max_lags"]),
                    "search_boundary": _flag_is_true(row["search_boundary"]),
                    "warning": row.get("warning"),
                },
            })

    if sensitivity_summary is not None and not sensitivity_summary.empty:
        for _, row in sensitivity_summary.iterrows():
            rows.append({
                "section": "hierarchical_HAC_sensitivity",
                "metric": row["Specification"],
                "value": int(row["Nominally supported"]),
                "details": {
                    key: row[key]
                    for key in sensitivity_summary.columns
                    if key not in {"Specification", "Nominally supported"}
                },
            })

    rows.append({"section": "execution", "metric": "error_count", "value": int(len(errors))})

    return pd.DataFrame(rows).reindex(columns=[
        "section",
        "backend",
        "partition",
        "cause",
        "trigger",
        "trigger_source",
        "lag",
        "metric",
        "value",
        "details",
    ])


def export_audit_csvs(
    *,
    results_dir: str | Path,
    case_prefix: str,
    experiment_grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    design: pd.DataFrame,
    lag_references: pd.DataFrame,
    automatic_lag_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    warning_table: pd.DataFrame,
    variable_labels: Mapping[str, str] | None = None,
    role_shift_table: pd.DataFrame | None = None,
    lag_sensitivity_table: pd.DataFrame | None = None,
    sensitivity_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    criteria_by_lag = reference_criteria_by_lag(lag_references)
    labels = {} if variable_labels is None else variable_labels

    runs_export = experiment_grid.copy()
    runs_export["backend_label"] = runs_export["backend"].map(
        lambda value: BACKEND_LABELS.get(str(value), str(value))
    )
    runs_export["reference_criterion"] = runs_export["lag"].map(criteria_by_lag).fillna("")
    runs_export = runs_export.sort_values(["split_id", "backend", "lag"], kind="stable")

    moderation_export = diagnostics.copy()
    if not moderation_export.empty:
        moderation_export["backend_label"] = moderation_export["backend"].map(
            lambda value: BACKEND_LABELS.get(str(value), str(value))
        )
        moderation_export["trigger_source_label"] = moderation_export["trigger_source"].map(
            lambda value: TRIGGER_SOURCE_LABELS.get(str(value), str(value))
        )
        moderation_export["reference_criterion"] = (
            moderation_export["lag"].map(criteria_by_lag).fillna("")
        )
        moderation_export["cause_label"] = moderation_export["cause"].map(
            lambda value: readable_name(value, labels)
        )
        moderation_export["trigger_label"] = moderation_export["trigger"].map(
            lambda value: readable_name(value, labels)
        )
        moderation_export = moderation_export.sort_values(
            ["split_id", "backend", "lag", "accepted", "cause", "trigger"],
            ascending=[True, True, True, False, True, True],
            kind="stable",
        )

    errors = experiment_grid.loc[experiment_grid["error"].notna()].copy()
    summary_export = build_summary_export(
        design=design,
        lag_references=lag_references,
        automatic_lag_summary=automatic_lag_summary,
        split_summary=split_summary,
        backend_summary=backend_summary,
        pair_summary=pair_summary,
        warning_table=warning_table,
        diagnostics=diagnostics,
        errors=errors,
        role_shift_table=role_shift_table,
        lag_sensitivity_table=lag_sensitivity_table,
        sensitivity_summary=sensitivity_summary,
    )

    paths = {
        "runs": results_dir / f"{case_prefix}_cause_trigger_all_runs.csv",
        "moderation": results_dir / f"{case_prefix}_cause_trigger_moderation_diagnostics.csv",
        "summary": results_dir / f"{case_prefix}_cause_trigger_summary.csv",
    }
    csv_ready(runs_export).to_csv(paths["runs"], index=False)
    csv_ready(moderation_export).to_csv(paths["moderation"], index=False)
    csv_ready(summary_export).to_csv(paths["summary"], index=False)

    summary_description = (
        "Design, split stability, automatic AIC/BIC outcomes, backend behaviour, "
        "pair stability, mean-shift role diagnostics, warning audit, descriptive "
        "multiplicity audit, hierarchical--HAC sensitivity, and any lag-order "
        "sensitivity check."
    )
    
    return pd.DataFrame([
        {
            "File": paths["runs"].name,
            "Contents": (
                "Every unique split × backend × maximum-lag-order run, including "
                "supporting minimum-I2 values, candidate sets, pairs, warnings, stop reasons, and errors."
            ),
            "Rows": len(runs_export),
        },
        {
            "File": paths["moderation"].name,
            "Contents": (
                "Every evaluated cause-trigger combination, nominal paper-compatible decision, "
                "trigger source, hierarchical--HAC sensitivities, complete test "
                "statistics, and specification-wise case-wide BH audits."
            ),
            "Rows": len(moderation_export),
        },
        {
            "File": paths["summary"].name,
            "Contents": summary_description,
            "Rows": len(summary_export),
        },
    ])