from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKEND_LABELS = {
    "hmml": "HMML (baseline)",
    "pcmci": "PCMCI",
    "pcmci_plus": "PCMCI+ with eligible τ=0 triggers",
}

TRIGGER_SOURCE_LABELS = {
    "lagged": "Lagged",
    "contemporaneous": "Same-hour PCMCI+ link (τ=0)",
}


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

    figure, axis = plt.subplots(figsize=(14, 4))
    axis.plot(dataframe.index, dataframe[effect], label=effect, linewidth=0.8)

    valid = split_summary.dropna(subset=["split_time"]).copy()
    valid["split_time"] = pd.to_datetime(valid["split_time"], utc=True)
    for split_time, group in valid.groupby("split_time", sort=True):
        causal_values = sorted({
            int(value)
            for value in group.loc[group.get("in_causal_grid", True), "min_I2_length"]
        })
        diagnostic_values = sorted({
            int(value)
            for value in group.loc[~group.get("in_causal_grid", True), "min_I2_length"]
        }) if "in_causal_grid" in group.columns else []
        label_parts = []
        if causal_values:
            label_parts.append("causal minimum I2: " + ", ".join(map(str, causal_values)) + " h")
        if diagnostic_values:
            label_parts.append("split-only: " + ", ".join(map(str, diagnostic_values)) + " h")
        axis.axvline(
            split_time,
            linestyle="--",
            linewidth=1.0,
            label="Detected split (" + "; ".join(label_parts) + ")",
        )

    if event_time is not None:
        axis.axvline(
            pd.Timestamp(event_time),
            linestyle=":",
            linewidth=1.0,
            label=event_label,
        )

    axis.set(
        title=title or f"{effect}: data-selected split partitions",
        xlabel="Time",
        ylabel="Reference-standardised value",
    )
    axis.legend()
    figure.tight_layout()

    saved_paths = _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.show()
    return figure, axis, saved_paths


def show_table(frame: pd.DataFrame, caption: str | None = None) -> None:
    """Display a compact notebook table without triggering duplicate rendering."""
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


def _warning_details(caught: Sequence[warnings.WarningMessage]) -> tuple[int, list[str], list[str]]:
    messages = list(dict.fromkeys(str(item.message) for item in caught))
    categories = list(dict.fromkeys(item.category.__name__ for item in caught))
    return len(caught), messages, categories


def add_casewide_bh_adjustment(
    diagnostics: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add a descriptive case-wide Benjamini–Hochberg moderation audit."""
    output = diagnostics.copy()
    output["bh_adjusted_p_value"] = np.nan
    output["accepted_after_bh"] = False
    if output.empty or "p_value" not in output.columns:
        return output

    p_values = pd.to_numeric(output["p_value"], errors="coerce")
    valid_index = p_values.index[p_values.notna() & np.isfinite(p_values)]
    if len(valid_index) == 0:
        return output

    ordered = p_values.loc[valid_index].sort_values(kind="stable")
    m = len(ordered)
    raw_adjusted = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output.loc[ordered.index, "bh_adjusted_p_value"] = adjusted
    output.loc[ordered.index, "accepted_after_bh"] = adjusted <= float(alpha)
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
    causal_split_summary = (
        split_summary.loc[split_summary["in_causal_grid"]].copy()
        if "in_causal_grid" in split_summary.columns
        else split_summary.copy()
    )
    unique_splits = (
        causal_split_summary
        .sort_values(["split_id", "min_I2_length"], kind="stable")
        .drop_duplicates("split_id")
    )

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
            "I1_length_expected": int(split["I1_length"]),
            "I2_length_expected": int(split["I2_length"]),
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
                    if actual_split is not None and pd.Timestamp(actual_split) != pd.Timestamp(
                        split["split_time"]
                    ):
                        raise RuntimeError(
                            "Representative minimum-I2 value reproduced a different split: "
                            f"expected {split['split_time']}, got {actual_split}."
                        )

                    rows.append({
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
                        "warning_categories": warning_categories,
                        "warning_messages": warning_messages,
                        "error": None,
                    })

                    if not diagnostics.empty:
                        diagnostic_frames.append(
                            diagnostics.assign(
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
                                **metadata,
                            )
                        )
                except Exception as exc:
                    warning_count, warning_messages, warning_categories = _warning_details(caught)
                    rows.append({
                        **common,
                        "warning_count": warning_count,
                        "warning_categories": warning_categories,
                        "warning_messages": warning_messages,
                        "error": str(exc),
                    })

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, pd.DataFrame()

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
        "Accepted-pair lags",
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
        n_pairs = int(successful.get("pairs", empty).apply(_has_items).sum())
        n_tau0 = int(
            successful.get("n_contemporaneous_links", zero).fillna(0).gt(0).sum()
        )
        n_warning_runs = int(successful.get("warning_count", zero).fillna(0).gt(0).sum())

        if n_pairs:
            result = f"Accepted pair(s) at {n_pairs} maximum-lag order(s)"
        elif n_candidates:
            result = "Candidates found; none passed moderation"
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
            "Accepted-pair lags": f"{n_pairs}/{lag_count}",
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
    """Summarise raw and BH-supported pairs without counting duplicate splits."""
    columns = [
        "Backend",
        "Trigger source",
        "Cause",
        "Trigger",
        "Accepted cells",
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

    accepted = diagnostics.loc[diagnostics["accepted"].eq(True)].copy()
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
                lambda values: int(pd.Series(values).fillna(False).astype(bool).sum()),
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
        "Accepted cells": summary["accepted_cells"].astype(int).astype(str) + f"/{total_cells}",
        "BH-supported cells": summary["bh_cells"].astype(int).astype(str) + f"/{total_cells}",
        "Partitions": summary["split_ids"].astype(str),
        "Partition support": summary["n_splits"].astype(int).astype(str) + f"/{n_splits_total}",
        "Minimum-I2 values": summary["min_i2_values"].astype(str),
        "Maximum lag orders": summary["lag_orders"].astype(str),
        "Lag-order support": summary["n_lags"].astype(int).astype(str) + f"/{len(tuple(lags))}",
        "Longest adjacent lag run": summary["longest"].astype(int),
    }).reset_index(drop=True)


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
            f"{readable_name(row.get('cause'), labels)} → "
            f"{readable_name(row.get('trigger'), labels)} [{source}]"
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
            accepted = matched.loc[matched.get("accepted", False).eq(True)] if not matched.empty else pd.DataFrame()

            if not accepted.empty:
                outcome = "Accepted: " + _format_pairs(accepted, variable_labels)
            elif not matched.empty and pd.to_numeric(matched.get("p_value"), errors="coerce").notna().any():
                minimum_p = pd.to_numeric(matched["p_value"], errors="coerce").min()
                outcome = f"Moderation tested; none accepted (minimum p={minimum_p:.3g})"
            elif pd.notna(run.get("stop_reason")):
                outcome = str(run.get("stop_reason"))
            elif _has_items(run.get("trigger_candidates")):
                outcome = "Trigger candidates found; no accepted moderation result"
            elif _has_items(run.get("B_2")):
                outcome = "Post-split parents found; no trigger candidates"
            else:
                outcome = "No usable post-split parent structure"

            rows.append({
                "Criterion": reference["criterion"],
                "Selected d (h)": lag,
                "Search boundary": bool(reference.get("search_boundary", False)),
                "Partition": run["split_id"],
                "Minimum-I2 support": str(run["supported_min_I2_lengths"]),
                "Backend": backend_labels.get(str(run["backend"]), str(run["backend"])),
                "Outcome": outcome,
            })

    return pd.DataFrame(rows).reindex(columns=columns)


def warning_summary(grid: pd.DataFrame) -> pd.DataFrame:
    columns = ["Partition", "Backend", "Runs with warnings", "Warning count", "Messages"]
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
        rows.append({
            "Partition": split_id,
            "Backend": BACKEND_LABELS.get(str(backend), str(backend)),
            "Runs with warnings": int(len(group)),
            "Warning count": int(pd.to_numeric(group["warning_count"], errors="coerce").sum()),
            "Messages": "; ".join(dict.fromkeys(messages)),
        })
    return pd.DataFrame(rows).reindex(columns=columns)



def multiplicity_audit_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row raw-versus-BH moderation audit."""
    if diagnostics.empty:
        return pd.DataFrame([{
            "Fitted moderation tests": 0,
            "F-test accepted": 0,
            "BH-supported": 0,
        }])

    fitted = pd.to_numeric(
        diagnostics.get("p_value"), errors="coerce"
    ).notna()
    raw = diagnostics.get(
        "accepted", pd.Series(False, index=diagnostics.index)
    ).fillna(False).astype(bool)
    adjusted = diagnostics.get(
        "accepted_after_bh",
        pd.Series(False, index=diagnostics.index),
    ).fillna(False).astype(bool)
    return pd.DataFrame([{
        "Fitted moderation tests": int(fitted.sum()),
        "F-test accepted": int(raw.sum()),
        "BH-supported": int(adjusted.sum()),
    }])

def delayed_lag_correlation(
    frame: pd.DataFrame,
    *,
    effect: str,
    predictor: str,
    max_lag: int,
) -> pd.DataFrame:
    rows = []
    for lag in range(int(max_lag) + 1):
        aligned = pd.concat(
            [
                frame[effect].rename("effect"),
                frame[predictor].shift(lag).rename("predictor_lagged"),
            ],
            axis=1,
        ).dropna()
        rows.append({
            "lag_hours": lag,
            "correlation": aligned["effect"].corr(aligned["predictor_lagged"]),
            "n_aligned": len(aligned),
        })
    return pd.DataFrame(rows)



def top_delayed_lags(scan: pd.DataFrame, *, n: int = 8) -> pd.DataFrame:
    """Return the strongest absolute correlations from a delayed-lag scan."""
    if scan.empty:
        return scan.copy()
    return (
        scan.assign(_absolute=scan["correlation"].abs())
        .nlargest(int(n), "_absolute")
        .drop(columns="_absolute")
        .sort_values("lag_hours", kind="stable")
        .reset_index(drop=True)
    )

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
    delayed_scan: pd.DataFrame | None = None,
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
                "search_boundary": bool(row.get("search_boundary", False)),
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
                "search_boundary": bool(row["Search boundary"]),
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
            "metric": "accepted_split_lag_support",
            "value": row["Accepted cells"],
            "details": {
                key: row[key]
                for key in pair_summary.columns
                if key not in {"Backend", "Cause", "Trigger", "Trigger source", "Accepted cells"}
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
                "metric": "f_accepted_tests",
                "value": int(diagnostics.get("accepted", False).fillna(False).astype(bool).sum()),
            },
            {
                "section": "multiplicity_audit",
                "metric": "BH_supported_tests",
                "value": int(
                    diagnostics.get("accepted_after_bh", False)
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            },
        ])

    if delayed_scan is not None and not delayed_scan.empty:
        for _, row in delayed_scan.iterrows():
            rows.append({
                "section": "delayed_lag_scan",
                "lag": int(row["lag_hours"]),
                "metric": "correlation",
                "value": float(row["correlation"]),
                "details": {"aligned_rows": int(row["n_aligned"]), "descriptive_only": True},
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
    delayed_scan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write exactly three auditable CSV outputs."""
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
        delayed_scan=delayed_scan,
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
        "pair stability, warning audit, and descriptive multiplicity audit."
    )
    if delayed_scan is not None:
        summary_description = summary_description[:-1] + ", plus the delayed-lag scan."

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
                "Every evaluated cause-trigger combination, raw paper-compatible decision, "
                "trigger source, complete test statistics, and case-wide BH audit."
            ),
            "Rows": len(moderation_export),
        },
        {
            "File": paths["summary"].name,
            "Contents": summary_description,
            "Rows": len(summary_export),
        },
    ])