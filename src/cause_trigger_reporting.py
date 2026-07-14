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
    "pcmci_plus": "PCMCI+ τ=0",
}


def _save_figure(
    figure,
    *,
    filename: str | None,
    save_dir: str | Path | None,
    formats: str | Sequence[str] = ("pdf", "png"),
    dpi: int = 450,
) -> list[Path]:
    """Save a figure in one or more formats and return the written paths."""
    if filename is None or save_dir is None:
        return []

    output_directory = Path(save_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_formats = (
        (formats,)
        if isinstance(formats, str)
        else tuple(formats)
    )
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
    """Plot the effect with all distinct splits induced by the MIN_I2 grid."""
    required = {"min_I2_length", "split_time"}
    missing = sorted(required - set(split_summary.columns))
    if missing:
        raise ValueError(
            f"split_summary is missing required columns: {missing}"
        )

    figure, axis = plt.subplots(figsize=(16, 4))
    axis.plot(
        dataframe.index,
        dataframe[effect],
        label=effect,
        linewidth=0.8,
    )

    valid_splits = split_summary.dropna(subset=["split_time"]).copy()
    if not valid_splits.empty:
        valid_splits["split_time"] = pd.to_datetime(
            valid_splits["split_time"],
            utc=True,
        )

        grouped = (
            valid_splits
            .groupby("split_time", sort=True)["min_I2_length"]
            .apply(lambda values: sorted({int(value) for value in values}))
        )

        for split_time, min_i2_values in grouped.items():
            values_text = ", ".join(str(value) for value in min_i2_values)
            axis.axvline(
                split_time,
                linestyle="--",
                linewidth=1.0,
                label=f"Detected split (min I2: {values_text} h)",
            )

    if event_time is not None:
        axis.axvline(
            pd.Timestamp(event_time),
            linestyle=":",
            linewidth=1.0,
            label=event_label,
        )

    axis.set(
        title=(
            title
            or f"{effect}: splits across the minimum-I2 grid"
        ),
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


def show_table(
    frame: pd.DataFrame,
    caption: str | None = None,
):
    """Display a compact dataframe table in a notebook."""
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
            {
                "selector": "td",
                "props": [("padding", "5px 9px")],
            },
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
    return styler


def _has_items(value) -> bool:
    return (
        isinstance(value, (list, tuple, set, dict))
        and len(value) > 0
    )


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


def readable_name(
    value,
    variable_labels: Mapping[str, str] | None = None,
):
    if value is None:
        return value
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return value

    labels = {} if variable_labels is None else variable_labels
    return labels.get(
        str(value),
        str(value).removesuffix("_scaled"),
    )


def run_parameter_grid(
    df_model: pd.DataFrame,
    df_mean: pd.DataFrame,
    workflow_template,
    *,
    run_one: Callable,
    run_specs: Sequence[Mapping[str, object]],
    min_i2_values: Sequence[int],
    lags: Sequence[int],
    cond_ind_test: str,
    metadata: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete MIN_I2 × backend × maximum-lag-order experiment.

    Every requested configuration is retained, including failed or empty runs.
    ``workflow_template.min_I2_length`` is overwritten for every grid value.
    """
    rows: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []
    metadata = {} if metadata is None else dict(metadata)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._optimize",
        )

        for min_i2_length in min_i2_values:
            current_workflow = replace(
                workflow_template,
                min_I2_length=int(min_i2_length),
            )

            for spec in run_specs:
                run_name = str(spec["run"])
                backend = str(spec["backend"])

                for lag in lags:
                    common = {
                        "experiment_type": (
                            "min_I2_backend_max_lag_grid"
                        ),
                        "min_I2_length": int(min_i2_length),
                        "run": run_name,
                        "backend": backend,
                        "lag": int(lag),
                        **metadata,
                    }

                    try:
                        result, diagnostics = run_one(
                            df_model,
                            df_mean,
                            current_workflow,
                            run_name=run_name,
                            backend=backend,
                            lag=int(lag),
                            cond_ind_test=spec.get(
                                "cond_ind_test",
                                cond_ind_test,
                            ),
                            use_contemporaneous_triggers=spec.get(
                                "use_contemporaneous_triggers"
                            ),
                        )

                        rows.append({
                            **common,
                            "split_timestamp": result.get(
                                "split_timestamp"
                            ),
                            "split_score": result.get("split_score"),
                            "boundary_split": result.get(
                                "boundary_split"
                            ),
                            "I1_length": result.get("I1_length"),
                            "I2_length": result.get("I2_length"),
                            "B_1": result.get("B_1"),
                            "B_2": result.get("B_2"),
                            "trigger_candidates": result.get(
                                "T_candidates"
                            ),
                            "T_candidates_lagged": result.get(
                                "T_candidates_lagged"
                            ),
                            "T_candidates_contemporaneous": result.get(
                                "T_candidates_contemporaneous"
                            ),
                            "accepted_triggers": result.get("T"),
                            "causes": result.get("C"),
                            "pairs": result.get("pairs"),
                            "n_contemporaneous_links": len(
                                result.get(
                                    "contemporaneous_links",
                                    {},
                                )
                            ),
                            "n_diagnostics": len(diagnostics),
                            "stop_reason": result.get("stop_reason"),
                            "error": None,
                        })

                        if not diagnostics.empty:
                            diagnostic_frames.append(
                                diagnostics.assign(
                                    experiment_type=common[
                                        "experiment_type"
                                    ],
                                    min_I2_length=int(
                                        min_i2_length
                                    ),
                                    **metadata,
                                )
                            )

                    except Exception as exc:
                        rows.append({
                            **common,
                            "error": str(exc),
                        })

    grid_columns = [
        "experiment_type",
        "min_I2_length",
        "run",
        "backend",
        "lag",
        *metadata.keys(),
        "split_timestamp",
        "split_score",
        "boundary_split",
        "I1_length",
        "I2_length",
        "B_1",
        "B_2",
        "trigger_candidates",
        "T_candidates_lagged",
        "T_candidates_contemporaneous",
        "accepted_triggers",
        "causes",
        "pairs",
        "n_contemporaneous_links",
        "n_diagnostics",
        "stop_reason",
        "error",
    ]
    grid = pd.DataFrame(rows).reindex(columns=grid_columns)
    if grid.empty:
        return grid, pd.DataFrame()

    grid["effective_I2_rows"] = (
        pd.to_numeric(
            grid.get("I2_length"),
            errors="coerce",
        )
        - pd.to_numeric(grid["lag"], errors="coerce")
    )

    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame(columns=[
            "run",
            "backend",
            "lag",
            "trigger",
            "trigger_source",
            "cause",
            "accepted",
            "p_value",
            "f_stat",
            "critical_f",
            "rss_reduction_ratio",
            "gamma_2",
            "reason",
            "experiment_type",
            "min_I2_length",
            *metadata.keys(),
        ])
    )

    backend_order = {
        str(spec["backend"]): position
        for position, spec in enumerate(run_specs)
    }
    grid["_backend_order"] = grid["backend"].map(backend_order)
    grid = (
        grid
        .sort_values(
            ["min_I2_length", "_backend_order", "lag"],
            kind="stable",
        )
        .drop(columns="_backend_order")
        .reset_index(drop=True)
    )

    if not diagnostics.empty:
        diagnostics["_backend_order"] = diagnostics["backend"].map(
            backend_order
        )
        diagnostics = (
            diagnostics
            .sort_values(
                [
                    "min_I2_length",
                    "_backend_order",
                    "lag",
                    "accepted",
                    "cause",
                    "trigger",
                ],
                ascending=[True, True, True, False, True, True],
                kind="stable",
            )
            .drop(columns="_backend_order")
            .reset_index(drop=True)
        )

    return grid, diagnostics


def backend_min_i2_summary(
    grid: pd.DataFrame,
    *,
    lag_count: int,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarise each backend separately at every minimum-I2 setting."""
    columns = [
        "Backend",
        "Minimum I2 (h)",
        "Successful runs",
        "Post-split parents",
        "Trigger candidates",
        "Accepted-pair lag orders",
        "τ=0-link lag orders",
        "Result",
    ]
    required = {"run", "backend", "lag", "min_I2_length", "error"}
    if grid.empty or not required.issubset(grid.columns):
        return pd.DataFrame(columns=columns)

    rows = []
    for (_, backend, min_i2), group in grid.groupby(
        ["run", "backend", "min_I2_length"],
        dropna=False,
        sort=False,
    ):
        successful = group.loc[group["error"].isna()].copy()
        empty = pd.Series(
            [[] for _ in range(len(successful))],
            index=successful.index,
            dtype=object,
        )
        zero = pd.Series(
            0,
            index=successful.index,
            dtype=float,
        )

        n_b2 = int(
            successful.get("B_2", empty).apply(_has_items).sum()
        )
        n_candidates = int(
            successful.get(
                "trigger_candidates",
                empty,
            ).apply(_has_items).sum()
        )
        n_pairs = int(
            successful.get("pairs", empty).apply(_has_items).sum()
        )
        n_tau0 = int(
            successful.get(
                "n_contemporaneous_links",
                zero,
            )
            .fillna(0)
            .gt(0)
            .sum()
        )

        if n_pairs:
            result = f"Accepted pair(s) at {n_pairs} lag order(s)"
        elif n_candidates:
            result = "Candidates found; none passed moderation"
        elif n_b2:
            result = "Post-split parents found; no trigger candidates"
        else:
            result = "No post-split parent structure"

        rows.append({
            "Backend": backend_labels.get(
                str(backend),
                str(backend),
            ),
            "Minimum I2 (h)": int(min_i2),
            "Successful runs": f"{len(successful)}/{lag_count}",
            "Post-split parents": f"{n_b2}/{lag_count}",
            "Trigger candidates": f"{n_candidates}/{lag_count}",
            "Accepted-pair lag orders": f"{n_pairs}/{lag_count}",
            "τ=0-link lag orders": f"{n_tau0}/{lag_count}",
            "Result": result,
        })

    return pd.DataFrame(rows).reindex(columns=columns)


def pair_grid_stability_summary(
    grid: pd.DataFrame,
    *,
    min_i2_values: Sequence[int],
    lags: Sequence[int],
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarise accepted-pair recurrence over the full two-parameter grid."""
    columns = [
        "Backend",
        "Cause",
        "Trigger",
        "Grid support",
        "Minimum-I2 lengths",
        "Minimum-I2 support",
        "Maximum lag orders",
        "Lag-order support",
        "Longest adjacent lag run",
    ]
    required = {
        "run",
        "backend",
        "lag",
        "min_I2_length",
        "pairs",
        "error",
    }
    if grid.empty or not required.issubset(grid.columns):
        return pd.DataFrame(columns=columns)

    records = []
    successful = grid.loc[grid["error"].isna()]
    for _, row in successful.iterrows():
        pairs = row.get("pairs")
        if not isinstance(pairs, list):
            continue

        for pair in pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
            ):
                continue
            records.append({
                "run": row["run"],
                "backend": row["backend"],
                "lag": int(row["lag"]),
                "min_I2_length": int(row["min_I2_length"]),
                "cause": pair[0],
                "trigger": pair[1],
            })

    if not records:
        return pd.DataFrame(columns=columns)

    accepted = pd.DataFrame(records).drop_duplicates()
    summary = (
        accepted
        .groupby(
            ["run", "backend", "cause", "trigger"],
            as_index=False,
            sort=False,
        )
        .agg(
            accepted_cells=(
                "lag",
                "size",
            ),
            min_i2_lengths=(
                "min_I2_length",
                lambda values: sorted(
                    {int(value) for value in values}
                ),
            ),
            lag_orders=(
                "lag",
                lambda values: sorted(
                    {int(value) for value in values}
                ),
            ),
        )
    )
    summary["n_min_i2"] = summary["min_i2_lengths"].map(len)
    summary["n_lags"] = summary["lag_orders"].map(len)
    summary["longest"] = summary["lag_orders"].map(
        _longest_consecutive_run
    )
    summary = summary.sort_values(
        [
            "accepted_cells",
            "n_min_i2",
            "n_lags",
            "longest",
            "backend",
            "cause",
            "trigger",
        ],
        ascending=[False, False, False, False, True, True, True],
    )

    labels = {} if variable_labels is None else variable_labels
    grid_size = len(tuple(min_i2_values)) * len(tuple(lags))

    return pd.DataFrame({
        "Backend": summary["backend"].map(
            lambda value: backend_labels.get(
                str(value),
                str(value),
            )
        ),
        "Cause": summary["cause"].map(
            lambda value: readable_name(value, labels)
        ),
        "Trigger": summary["trigger"].map(
            lambda value: readable_name(value, labels)
        ),
        "Grid support": (
            summary["accepted_cells"].astype(int).astype(str)
            + f"/{grid_size}"
        ),
        "Minimum-I2 lengths": summary["min_i2_lengths"].astype(str),
        "Minimum-I2 support": (
            summary["n_min_i2"].astype(int).astype(str)
            + f"/{len(tuple(min_i2_values))}"
        ),
        "Maximum lag orders": summary["lag_orders"].astype(str),
        "Lag-order support": (
            summary["n_lags"].astype(int).astype(str)
            + f"/{len(tuple(lags))}"
        ),
        "Longest adjacent lag run": summary["longest"].astype(int),
    }).reset_index(drop=True)


def reference_criteria_by_lag(
    lag_references: pd.DataFrame,
) -> dict[int, str]:
    """Map each conventional reference lag to its information criterion."""
    if lag_references.empty:
        return {}

    return (
        lag_references
        .groupby("selected_lag")["criterion"]
        .apply(lambda values: "/".join(sorted(set(values))))
        .to_dict()
    )


def delayed_lag_correlation(
    frame: pd.DataFrame,
    *,
    effect: str,
    predictor: str,
    max_lag: int,
) -> pd.DataFrame:
    """Descriptive predictor(t-lag)-effect(t) correlation scan."""
    rows = []

    for lag in range(int(max_lag) + 1):
        aligned = pd.concat(
            [
                frame[effect].rename("effect"),
                frame[predictor].shift(lag).rename(
                    "predictor_lagged"
                ),
            ],
            axis=1,
        ).dropna()

        rows.append({
            "lag_hours": lag,
            "correlation": aligned["effect"].corr(
                aligned["predictor_lagged"]
            ),
            "n_aligned": len(aligned),
        })

    return pd.DataFrame(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialise nested structures consistently before CSV export."""
    output = frame.copy()

    for column in output.columns:
        if output[column].dtype != object:
            continue

        output[column] = output[column].map(
            lambda value: (
                json.dumps(
                    _json_ready(value),
                    ensure_ascii=False,
                    default=str,
                )
                if isinstance(
                    value,
                    (list, tuple, set, dict),
                )
                else (
                    value.isoformat()
                    if isinstance(value, pd.Timestamp)
                    else value
                )
            )
        )

    return output


def build_summary_export(
    *,
    design: pd.DataFrame,
    lag_references: pd.DataFrame,
    split_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    errors: pd.DataFrame,
    delayed_scan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one tidy summary of the complete parameter-grid experiment."""
    rows = []

    for _, row in design.iterrows():
        rows.append({
            "section": "design",
            "metric": row["Item"],
            "value": row["Value"],
        })

    for _, row in lag_references.iterrows():
        rows.append({
            "section": "lag_reference",
            "lag": int(row["selected_lag"]),
            "metric": str(row["criterion"]),
            "value": f"{int(row['selected_lag'])} h",
            "details": {
                "distribution": row.get("distribution"),
                "maximum_lag_tested": row.get("max_lags"),
                "diagnostic_only": True,
            },
        })

    for _, row in split_summary.iterrows():
        rows.append({
            "section": "split_grid",
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
            "section": "backend_by_min_i2",
            "backend": row["Backend"],
            "metric": f"min_I2={int(row['Minimum I2 (h)'])} h",
            "value": row["Result"],
            "details": {
                key: row[key]
                for key in backend_summary.columns
                if key not in {
                    "Backend",
                    "Minimum I2 (h)",
                    "Result",
                }
            },
        })

    for _, row in pair_summary.iterrows():
        rows.append({
            "section": "pair_grid_stability",
            "backend": row["Backend"],
            "cause": row["Cause"],
            "trigger": row["Trigger"],
            "metric": "accepted_configuration_support",
            "value": row["Grid support"],
            "details": {
                "minimum_I2_lengths": row[
                    "Minimum-I2 lengths"
                ],
                "minimum_I2_support": row[
                    "Minimum-I2 support"
                ],
                "maximum_lag_orders": row[
                    "Maximum lag orders"
                ],
                "lag_order_support": row[
                    "Lag-order support"
                ],
                "longest_adjacent_lag_run": int(
                    row["Longest adjacent lag run"]
                ),
            },
        })

    if delayed_scan is not None and not delayed_scan.empty:
        for _, row in delayed_scan.iterrows():
            rows.append({
                "section": "delayed_lag_scan",
                "lag": int(row["lag_hours"]),
                "metric": "correlation",
                "value": float(row["correlation"]),
                "details": {
                    "aligned_rows": int(row["n_aligned"]),
                    "descriptive_only": True,
                },
            })

    rows.append({
        "section": "execution",
        "metric": "error_count",
        "value": int(len(errors)),
    })

    return pd.DataFrame(rows).reindex(columns=[
        "section",
        "backend",
        "cause",
        "trigger",
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
    split_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    variable_labels: Mapping[str, str] | None = None,
    delayed_scan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write exactly three audit CSVs for the complete grid experiment."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    criteria_by_lag = reference_criteria_by_lag(
        lag_references
    )
    labels = {} if variable_labels is None else variable_labels

    runs_export = experiment_grid.copy()
    runs_export["backend_label"] = runs_export["backend"].map(
        lambda value: BACKEND_LABELS.get(
            str(value),
            str(value),
        )
    )
    runs_export["reference_criterion"] = (
        runs_export["lag"]
        .map(criteria_by_lag)
        .fillna("")
    )
    runs_export = runs_export.sort_values(
        ["min_I2_length", "backend", "lag"],
        kind="stable",
    )

    moderation_export = diagnostics.copy()
    if not moderation_export.empty:
        moderation_export["backend_label"] = (
            moderation_export["backend"].map(
                lambda value: BACKEND_LABELS.get(
                    str(value),
                    str(value),
                )
            )
        )
        moderation_export["reference_criterion"] = (
            moderation_export["lag"]
            .map(criteria_by_lag)
            .fillna("")
        )
        moderation_export["cause_label"] = (
            moderation_export["cause"].map(
                lambda value: readable_name(value, labels)
            )
        )
        moderation_export["trigger_label"] = (
            moderation_export["trigger"].map(
                lambda value: readable_name(value, labels)
            )
        )
        moderation_export = moderation_export.sort_values(
            [
                "min_I2_length",
                "backend",
                "lag",
                "accepted",
                "cause",
                "trigger",
            ],
            ascending=[True, True, True, False, True, True],
            kind="stable",
        )

    errors = experiment_grid.loc[
        experiment_grid["error"].notna()
    ].copy()
    summary_export = build_summary_export(
        design=design,
        lag_references=lag_references,
        split_summary=split_summary,
        backend_summary=backend_summary,
        pair_summary=pair_summary,
        errors=errors,
        delayed_scan=delayed_scan,
    )

    paths = {
        "runs": (
            results_dir
            / f"{case_prefix}_cause_trigger_all_runs.csv"
        ),
        "moderation": (
            results_dir
            / (
                f"{case_prefix}_cause_trigger_"
                "moderation_diagnostics.csv"
            )
        ),
        "summary": (
            results_dir
            / f"{case_prefix}_cause_trigger_summary.csv"
        ),
    }

    csv_ready(runs_export).to_csv(
        paths["runs"],
        index=False,
    )
    csv_ready(moderation_export).to_csv(
        paths["moderation"],
        index=False,
    )
    csv_ready(summary_export).to_csv(
        paths["summary"],
        index=False,
    )

    summary_description = (
        "Design, AIC/BIC references, split results across the "
        "minimum-I2 grid, backend-by-minimum-I2 stability, and "
        "accepted-pair support across the full parameter grid."
    )
    if delayed_scan is not None:
        summary_description = (
            summary_description[:-1]
            + ", plus the complete descriptive delayed-lag scan."
        )

    return pd.DataFrame([
        {
            "File": paths["runs"].name,
            "Contents": (
                "Every minimum-I2 × backend × maximum-lag-order "
                "configuration, including empty results, candidate "
                "sets, accepted pairs, effective sample size, stop "
                "reasons, and errors."
            ),
            "Rows": len(runs_export),
        },
        {
            "File": paths["moderation"].name,
            "Contents": (
                "Every evaluated cause-trigger combination, "
                "accepted or rejected, with complete test "
                "statistics, reasons, and grid metadata."
            ),
            "Rows": len(moderation_export),
        },
        {
            "File": paths["summary"].name,
            "Contents": summary_description,
            "Rows": len(summary_export),
        },
    ])