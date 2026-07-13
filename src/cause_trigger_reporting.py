from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd


BACKEND_LABELS = {
    "hmml": "HMML",
    "pcmci": "PCMCI",
    "pcmci_plus": "PCMCI+ τ=0",
}


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
    """Return a readable variable label while preserving missing values."""
    if value is None:
        return value
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return value

    labels = {} if variable_labels is None else variable_labels
    return labels.get(
        str(value),
        str(value).removesuffix("_scaled"),
    )


def run_complete_grid(
    df_model: pd.DataFrame,
    df_mean: pd.DataFrame,
    workflow,
    *,
    run_one: Callable,
    run_specs: Sequence[Mapping[str, object]],
    lags: Sequence[int],
    cond_ind_test: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every backend–lag combination once and collect all diagnostics.

    Failed combinations are retained as rows so the audit output is complete.
    """
    rows: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._optimize",
        )

        for spec in run_specs:
            run_name = str(spec["run"])
            backend = str(spec["backend"])

            for lag in lags:
                try:
                    output = run_one(
                        df_model,
                        df_mean,
                        workflow,
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
                    result, diagnostics = output[:2]

                    rows.append({
                        "run": run_name,
                        "backend": backend,
                        "lag": int(lag),
                        "split_timestamp": result.get(
                            "split_timestamp"
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
                        diagnostic_frames.append(diagnostics)

                except Exception as exc:
                    rows.append({
                        "run": run_name,
                        "backend": backend,
                        "lag": int(lag),
                        "error": str(exc),
                    })

    grid = pd.DataFrame(rows)
    grid["effective_I2_rows"] = (
        pd.to_numeric(grid.get("I2_length"), errors="coerce")
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
        ])
    )
    return grid, diagnostics


def backend_stability_summary(
    grid: pd.DataFrame,
    *,
    lag_count: int,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarize signal progression for each causal-discovery backend."""
    rows = []

    for (_, backend), group in grid.groupby(
        ["run", "backend"],
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
            result = f"Accepted pair(s) at {n_pairs} lag(s)"
        elif n_candidates:
            result = "Candidates found; none passed moderation"
        else:
            result = "No post-split candidate structure"

        rows.append({
            "Backend": backend_labels.get(
                str(backend),
                str(backend),
            ),
            "Successful runs": (
                f"{len(successful)}/{lag_count}"
            ),
            "Post-split parents": f"{n_b2}/{lag_count}",
            "Trigger candidates": (
                f"{n_candidates}/{lag_count}"
            ),
            "Accepted-pair lags": f"{n_pairs}/{lag_count}",
            "τ=0-link lags": f"{n_tau0}/{lag_count}",
            "Result": result,
        })

    return pd.DataFrame(rows)


def pair_stability_summary(
    grid: pd.DataFrame,
    *,
    lag_count: int,
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Summarize accepted pair recurrence and adjacent-lag stability."""
    records = []

    for _, row in grid.loc[grid["error"].isna()].iterrows():
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
                "cause": pair[0],
                "trigger": pair[1],
            })

    columns = [
        "Backend",
        "Cause",
        "Trigger",
        "Accepted lags",
        "Lag support",
        "Longest adjacent run",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    summary = (
        pd.DataFrame(records)
        .drop_duplicates()
        .groupby(
            ["run", "backend", "cause", "trigger"],
            as_index=False,
            sort=False,
        )
        .agg(
            n_lags=("lag", "nunique"),
            lags=(
                "lag",
                lambda x: sorted(
                    {int(value) for value in x}
                ),
            ),
        )
    )
    summary["longest"] = summary["lags"].apply(
        _longest_consecutive_run
    )
    summary = summary.sort_values(
        [
            "n_lags",
            "longest",
            "backend",
            "cause",
            "trigger",
        ],
        ascending=[False, False, True, True, True],
    )

    labels = {} if variable_labels is None else variable_labels

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
        "Accepted lags": summary["lags"].astype(str),
        "Lag support": (
            summary["n_lags"].astype(int).astype(str)
            + f"/{lag_count}"
        ),
        "Longest adjacent run": summary["longest"].astype(int),
    }).reset_index(drop=True)


def reference_criteria_by_lag(
    lag_references: pd.DataFrame,
) -> dict[int, str]:
    """Map each reference lag to its joined information criteria."""
    return (
        lag_references
        .groupby("selected_lag")["criterion"]
        .apply(lambda values: "/".join(sorted(values)))
        .to_dict()
    )


def reference_moderation_table(
    diagnostics: pd.DataFrame,
    lag_references: pd.DataFrame,
    *,
    variable_labels: Mapping[str, str] | None = None,
    backend_labels: Mapping[str, str] = BACKEND_LABELS,
) -> pd.DataFrame:
    """Return accepted moderation tests at the AIC/BIC reference lags."""
    columns = [
        "Criterion",
        "Backend",
        "Lag",
        "Cause",
        "Trigger",
        "p-value",
        "RSS reduction",
        "Interaction coefficient",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)

    criteria_by_lag = reference_criteria_by_lag(
        lag_references
    )
    reference_lags = set(criteria_by_lag)

    accepted = diagnostics.loc[
        diagnostics["lag"].isin(reference_lags)
        & diagnostics["accepted"].eq(True)
    ].copy()

    if accepted.empty:
        return pd.DataFrame(columns=columns)

    labels = {} if variable_labels is None else variable_labels
    accepted["Criterion"] = accepted["lag"].map(
        criteria_by_lag
    )
    accepted["Backend"] = accepted["backend"].map(
        lambda value: backend_labels.get(
            str(value),
            str(value),
        )
    )
    accepted["Cause"] = accepted["cause"].map(
        lambda value: readable_name(value, labels)
    )
    accepted["Trigger"] = accepted["trigger"].map(
        lambda value: readable_name(value, labels)
    )

    return accepted[
        [
            "Criterion",
            "Backend",
            "lag",
            "Cause",
            "Trigger",
            "p_value",
            "rss_reduction_ratio",
            "gamma_2",
        ]
    ].rename(columns={
        "lag": "Lag",
        "p_value": "p-value",
        "rss_reduction_ratio": "RSS reduction",
        "gamma_2": "Interaction coefficient",
    })


def delayed_lag_correlation(
    frame: pd.DataFrame,
    *,
    effect: str,
    predictor: str,
    max_lag: int,
) -> pd.DataFrame:
    """Descriptive predictor(t-lag)–effect(t) correlation scan."""
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
    """Serialize nested structures consistently before CSV export."""
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
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    conclusion: str,
    errors: pd.DataFrame,
    delayed_scan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one tidy summary of design, stability, and interpretation."""
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
            },
        })

    for _, row in backend_summary.iterrows():
        rows.append({
            "section": "backend_stability",
            "backend": row["Backend"],
            "metric": "result",
            "value": row["Result"],
            "details": {
                key: row[key]
                for key in backend_summary.columns
                if key not in {"Backend", "Result"}
            },
        })

    for _, row in pair_summary.iterrows():
        rows.append({
            "section": "pair_stability",
            "backend": row["Backend"],
            "cause": row["Cause"],
            "trigger": row["Trigger"],
            "metric": "accepted_lags",
            "value": row["Accepted lags"],
            "details": {
                "lag_support": row["Lag support"],
                "longest_adjacent_run": int(
                    row["Longest adjacent run"]
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

    rows.extend([
        {
            "section": "execution",
            "metric": "error_count",
            "value": int(len(errors)),
        },
        {
            "section": "interpretation",
            "metric": "conclusion",
            "value": conclusion,
        },
    ])

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
    lag_grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    design: pd.DataFrame,
    lag_references: pd.DataFrame,
    backend_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    conclusion: str,
    variable_labels: Mapping[str, str] | None = None,
    delayed_scan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write exactly three audit CSVs and return a concise file manifest."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    criteria_by_lag = reference_criteria_by_lag(
        lag_references
    )
    labels = {} if variable_labels is None else variable_labels

    runs_export = lag_grid.copy()
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
                "backend",
                "lag",
                "accepted",
                "cause",
                "trigger",
            ],
            ascending=[True, True, False, True, True],
        )

    errors = lag_grid.loc[
        lag_grid["error"].notna()
    ].copy()
    summary_export = build_summary_export(
        design=design,
        lag_references=lag_references,
        backend_summary=backend_summary,
        pair_summary=pair_summary,
        conclusion=conclusion,
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
        "Design, split, lag references, backend stability, "
        "pair stability, and conclusion."
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
                "Every backend × lag run, including empty "
                "results, candidate sets, accepted pairs, "
                "effective sample size, stop reasons, and errors."
            ),
            "Rows": len(runs_export),
        },
        {
            "File": paths["moderation"].name,
            "Contents": (
                "Every evaluated cause–trigger combination, "
                "accepted or rejected, with complete test "
                "statistics and reasons."
            ),
            "Rows": len(moderation_export),
        },
        {
            "File": paths["summary"].name,
            "Contents": summary_description,
            "Rows": len(summary_export),
        },
    ])
