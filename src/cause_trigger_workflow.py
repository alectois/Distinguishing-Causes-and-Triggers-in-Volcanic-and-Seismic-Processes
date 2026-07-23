from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cause_trigger import (
    CauseTriggerConfig,
    diagnostics_to_dataframe,
    find_effect_split,
    run_cause_trigger,
    standard_scale_from_reference,
    validate_regular_time_index,
)
from cause_trigger_cases import CaseSpec
from parameter_extraction import select_var_lag


# HMML is the main parent-discovery backend. PCMCI and PCMCI+ provide
# comparisons; PCMCI+ also screens eligible directed tau=0 links as
# exploratory same-hour trigger candidates.
COMPACT_RUN_SPECS = (
    {"run": "hmml", "backend": "hmml"},
    {"run": "pcmci", "backend": "pcmci"},
    {
        "run": "pcmci_plus_tau0",
        "backend": "pcmci_plus",
        "use_contemporaneous_triggers": True,
    },
)


@dataclass(frozen=True)
class WorkflowConfig:
    """Shared Cause–Trigger settings for one case-study analysis."""

    effect: str
    alpha: float = 0.05
    selected_lag: int = 1
    min_I1_length: int = 48
    min_I2_length: int = 48
    distribution: str = "gaussian"

    # Ridge refit after PCMCI/PCMCI+ parent selection.
    refit_alpha: float = 1.0
    refit_cv: bool = True
    refit_cv_folds: int = 3

    # PCMCI / PCMCI+.
    # pc_alpha controls method-specific preliminary or graph selection.
    # pcmci_alpha_level is the final threshold applied to lagged p/q values
    # for both PCMCI and PCMCI+. PCMCI+ tau=0 links remain exploratory.
    pcmci_pc_alpha: float = 0.20
    pcmci_plus_pc_alpha: float = 0.01
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: Optional[str] = "fdr_bh"
    pcmci_cond_ind_test: str = "robust_parcorr"
    pcmci_verbosity: int = 0

    # PCMCI+ contemporaneous-link settings.
    pcmci_contemp_collider_rule: str = "majority"
    pcmci_conflict_resolution: bool = True
    pcmci_plus_use_contemporaneous_triggers: bool = False

    # Hierarchical-model Newey--West sensitivity tests.
    newey_west_lags: tuple[int, ...] = (6, 12, 24)


@dataclass
class PreparedCaseAnalysis:
    X_full_raw: pd.DataFrame
    X_case_raw: pd.DataFrame
    reference_raw: pd.DataFrame
    X_model: pd.DataFrame
    X_decision: pd.DataFrame
    scaling_report: pd.DataFrame
    lag_references: pd.DataFrame
    split_summary: pd.DataFrame
    unique_splits: pd.DataFrame
    design_summary: pd.DataFrame
    workflow_template: WorkflowConfig


def _resolve_time_column(
    dataframe: pd.DataFrame,
    requested_time_col: str,
) -> str:
    if requested_time_col in dataframe.columns:
        return requested_time_col

    for candidate in ("time", "timestamp", "datetime", "date"):
        if candidate in dataframe.columns:
            return candidate

    unnamed = [
        column
        for column in dataframe.columns
        if str(column).startswith("Unnamed:")
    ]
    if len(unnamed) == 1:
        return unnamed[0]

    raise ValueError(
        "Could not find a time column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def load_model_frame(
    csv_path: str | Path,
    *,
    time_col: str = "time",
    index_name: str = "time",
    drop_columns: Sequence[str] = ("station",),
    include_columns: Optional[Sequence[str]] = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Load the numeric unscaled model frame indexed by UTC time."""
    csv_path = Path(csv_path)
    dataframe = pd.read_csv(csv_path)
    resolved_time_col = _resolve_time_column(dataframe, time_col)

    dataframe[resolved_time_col] = pd.to_datetime(
        dataframe[resolved_time_col],
        utc=True,
        errors="coerce",
    )
    if dataframe[resolved_time_col].isna().any():
        raise ValueError(f"Invalid timestamps in {csv_path}.")

    if include_columns is not None:
        required = [resolved_time_col, *include_columns]
        missing = sorted(set(required) - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing requested columns: {missing}")
        dataframe = dataframe[required]

    model = (
        dataframe
        .drop(columns=list(drop_columns), errors="ignore")
        .set_index(resolved_time_col)
        .sort_index()
        .select_dtypes(include=[np.number])
    )
    model.index.name = index_name

    if model.index.has_duplicates:
        raise ValueError("Model dataframe has duplicate timestamps.")
    if require_complete and model.isna().any().any():
        missing = model.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(f"Model dataframe contains NaNs:\n{missing}")

    return model


def pre_case_reference_interval(
    df_full: pd.DataFrame,
    case_start: pd.Timestamp,
    *,
    reference_days: int = 14,
    min_coverage: float = 0.90,
    case_name: str = "Case",
) -> pd.DataFrame:
    """Return the adjacent pre-case reference interval."""
    case_start = pd.Timestamp(case_start)
    reference_start = case_start - pd.Timedelta(days=int(reference_days))
    reference = df_full.loc[
        (df_full.index >= reference_start)
        & (df_full.index < case_start)
    ].copy()

    expected_rows = int(
        (case_start - reference_start) / pd.Timedelta("1h")
    )
    coverage = len(reference) / expected_rows if expected_rows else 0.0

    if coverage < min_coverage:
        raise ValueError(
            f"Insufficient {case_name} reference coverage: "
            f"{len(reference)}/{expected_rows} ({coverage:.1%})."
        )
    if reference.isna().any().any():
        missing = reference.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(f"Reference interval contains NaNs:\n{missing}")

    reference.attrs.update(
        reference_start=reference_start,
        reference_end=case_start,
        expected_rows=expected_rows,
        coverage=coverage,
    )
    return reference


def prepare_case_frames(
    raw_case: pd.DataFrame,
    reference_raw: pd.DataFrame,
    *,
    case: CaseSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct aligned modelling and decision frames.

    X_model is standardised within the case interval for model fitting.
    X_decision is standardised against the pre-case reference interval for
    split selection, trigger screening, and cause ranking.
    """
    validate_regular_time_index(raw_case, expected_step="1h")

    if raw_case.isna().any().any():
        raise ValueError(f"{case.name} case interval contains missing values.")
    if reference_raw.empty:
        raise ValueError(f"{case.name} reference interval is empty.")
    if reference_raw.isna().any().any():
        raise ValueError(f"{case.name} reference interval contains missing values.")
    if list(reference_raw.columns) != list(raw_case.columns):
        raise ValueError(
            "Reference and case intervals must have identical column order."
        )
    if reference_raw.index.max() >= raw_case.index.min():
        raise ValueError("Reference interval must end before the case interval.")

    parameters = case.fit_transform_parameters(reference_raw)
    transformed_reference = case.transform_for_scaling(
        reference_raw.copy(),
        transform_parameters=parameters,
    )
    transformed_case = case.transform_for_scaling(
        raw_case.copy(),
        transform_parameters=parameters,
    )

    X_model = case.standard_scale_dataframe(transformed_case)
    X_decision, report = standard_scale_from_reference(
        transformed_reference,
        transformed_case,
    )

    scaled_columns = [f"{column}_scaled" for column in transformed_case.columns]
    X_model.columns = scaled_columns
    X_decision.columns = scaled_columns
    X_model.index = raw_case.index
    X_decision.index = raw_case.index

    report["model_case_mean"] = X_model.mean().to_numpy()
    report["model_case_std"] = X_model.std(ddof=0).to_numpy()
    report["reference_start"] = reference_raw.index.min()
    report["reference_end"] = reference_raw.index.max()
    report["case_start"] = raw_case.index.min()
    report["case_end"] = raw_case.index.max()

    return X_model, X_decision, report


def case_study_interval(
    df_full: pd.DataFrame,
    event_time: pd.Timestamp,
    *,
    pre_days: int,
    post_days: int | None = None,
) -> pd.DataFrame:
    """Return a complete hourly case grid around the contextual event."""
    if (post_days is None):
        raise ValueError("Specify the number of days after the event.")

    event_hour = pd.Timestamp(event_time).floor("h")
    start = event_hour - pd.Timedelta(days=int(pre_days))
    end = event_hour + pd.Timedelta(days=int(post_days))
    index = pd.date_range(start, end, freq="1h", inclusive="left")
    case = df_full.reindex(index)
    case.index.name = df_full.index.name
    return case


def reference_parameter_table(
    dataframe: pd.DataFrame,
    effect: str,
    *,
    max_lags: int = 12,
    criteria: Sequence[str] = ("aic", "bic"),
    fallback_lag: int = 1,
) -> pd.DataFrame:
    """Return VAR AIC/BIC lag-order references."""
    if effect not in dataframe.columns:
        raise ValueError(f"Effect variable {effect!r} is absent.")

    rows = []
    for criterion in criteria:
        lag = select_var_lag(
            dataframe,
            max_lags=max_lags,
            criterion=criterion,
            fallback_lag=fallback_lag,
        )
        rows.append({
            "criterion": criterion.upper(),
            "selected_lag": int(lag),
            "distribution": "gaussian",
            "max_lags": int(max_lags),
            "search_boundary": int(lag) == int(max_lags),
        })
    return pd.DataFrame(rows)


def split_diagnostics(
    dataframe: pd.DataFrame,
    effect: str,
    *,
    event_time: Optional[pd.Timestamp] = None,
    min_I1_length: int = 48,
    min_I2_length: int = 48,
) -> dict:
    split = find_effect_split(
        dataframe[effect],
        min_I1_length=min_I1_length,
        min_I2_length=min_I2_length,
        return_info=True,
    )
    split_index = split["split_index"]
    if split_index is None:
        return {
            "effect": effect,
            "split_index": None,
            "split_time": None,
            "I1_length": None,
            "I2_length": None,
            "abs_mean_I1": None,
            "abs_mean_I2": None,
            "split_score": None,
            "boundary_split": None,
            "distance_to_event": None,
        }

    split_time = dataframe.index[split_index]
    mean_1 = float(dataframe[effect].iloc[:split_index].mean())
    mean_2 = float(dataframe[effect].iloc[split_index:].mean())
    return {
        "effect": effect,
        "split_index": int(split_index),
        "split_time": split_time,
        "I1_length": int(split_index),
        "I2_length": int(len(dataframe) - split_index),
        "abs_mean_I1": abs(mean_1),
        "abs_mean_I2": abs(mean_2),
        "split_score": split["score"],
        "boundary_split": split["boundary_split"],
        "distance_to_event": (
            split_time - pd.Timestamp(event_time)
            if event_time is not None
            else None
        ),
    }


def identify_unique_splits(split_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign stable split IDs and return one row per unique partition."""
    required = {
        "min_I2_length",
        "split_time",
        "split_index",
        "I1_length",
        "I2_length",
    }
    missing = sorted(required - set(split_summary.columns))
    if missing:
        raise ValueError(f"split_summary is missing required columns: {missing}")

    annotated = split_summary.copy()
    key_columns = ["split_time", "split_index", "I1_length", "I2_length"]

    key_order = (
        annotated[key_columns]
        .drop_duplicates()
        .sort_values(key_columns, kind="stable", na_position="last")
        .reset_index(drop=True)
    )
    key_order["split_id"] = [f"S{index + 1}" for index in range(len(key_order))]
    annotated = annotated.merge(key_order, on=key_columns, how="left", validate="many_to_one")

    split_metadata = (
        annotated
        .groupby("split_id", as_index=False, sort=False)
        .agg(
            min_I2_values=(
                "min_I2_length",
                lambda values: sorted({int(value) for value in values}),
            ),
            fit_min_I2_length=("min_I2_length", "min"),
            any_boundary_split=("boundary_split", "any"),
            all_boundary_splits=("boundary_split", "all"),
        )
    )
    split_metadata["n_min_I2_values"] = split_metadata[
        "min_I2_values"
    ].map(len)
    annotated = annotated.merge(
        split_metadata,
        on="split_id",
        how="left",
        validate="many_to_one",
    )

    unique = (
        annotated
        .sort_values(["split_id", "min_I2_length"], kind="stable")
        .drop_duplicates("split_id", keep="first")
        .reset_index(drop=True)
    )
    return annotated, unique


def make_cause_trigger_config(
    workflow: WorkflowConfig,
    *,
    backend: str,
    lag: Optional[int] = None,
    cond_ind_test: Optional[str] = None,
    use_contemporaneous_triggers: Optional[bool] = None,
) -> CauseTriggerConfig:
    """Create one core configuration from the workflow settings."""
    if use_contemporaneous_triggers is None:
        use_contemporaneous_triggers = (
            workflow.pcmci_plus_use_contemporaneous_triggers
        )

    return CauseTriggerConfig(
        y_t=workflow.effect,
        lags=workflow.selected_lag if lag is None else int(lag),
        distribution=workflow.distribution,
        alpha=workflow.alpha,
        min_I1_length=workflow.min_I1_length,
        min_I2_length=workflow.min_I2_length,
        causal_backend=backend,
        refit_alpha=workflow.refit_alpha,
        refit_cv=workflow.refit_cv,
        refit_cv_folds=workflow.refit_cv_folds,
        pcmci_pc_alpha=workflow.pcmci_pc_alpha,
        pcmci_plus_pc_alpha=workflow.pcmci_plus_pc_alpha,
        pcmci_alpha_level=workflow.pcmci_alpha_level,
        pcmci_fdr_method=workflow.pcmci_fdr_method,
        pcmci_cond_ind_test=(
            workflow.pcmci_cond_ind_test
            if cond_ind_test is None
            else cond_ind_test
        ),
        pcmci_verbosity=workflow.pcmci_verbosity,
        pcmci_contemp_collider_rule=workflow.pcmci_contemp_collider_rule,
        pcmci_conflict_resolution=workflow.pcmci_conflict_resolution,
        pcmci_plus_use_contemporaneous_triggers=use_contemporaneous_triggers,
        newey_west_lags=tuple(
            int(value) for value in workflow.newey_west_lags
        ),
    )


def run_one(
    df_model: pd.DataFrame,
    df_decision: pd.DataFrame,
    workflow: WorkflowConfig,
    *,
    run_name: str,
    backend: str,
    lag: Optional[int] = None,
    cond_ind_test: Optional[str] = None,
    use_contemporaneous_triggers: Optional[bool] = None,
) -> tuple[dict, pd.DataFrame]:
    """Run one backend/lag combination and return result diagnostics."""
    config = make_cause_trigger_config(
        workflow,
        backend=backend,
        lag=lag,
        cond_ind_test=cond_ind_test,
        use_contemporaneous_triggers=use_contemporaneous_triggers,
    )
    result = run_cause_trigger(
        df_model,
        config,
        X_decision=df_decision,
    )
    diagnostics = diagnostics_to_dataframe(result)
    if not diagnostics.empty:
        diagnostics.insert(0, "run", run_name)
        diagnostics.insert(1, "backend", backend)
        diagnostics.insert(2, "lag", config.lags)

    return result, diagnostics


def prepare_case_analysis(
    *,
    data_path: str | Path,
    case: CaseSpec,
    event_time: pd.Timestamp,
    event_label: str,
    case_pre_days: int,
    case_post_days: int,
    reference_days: int,
    reference_min_coverage: float,
    min_I1_length: int,
    min_I2_values: Sequence[int],
    lag_grid: Sequence[int],
    alpha: float = 0.05,
    pcmci_pc_alpha: float = 0.2,
    pcmci_plus_pc_alpha: float = 0.01,
    pcmci_alpha_level: float = 0.05,
    pcmci_fdr_method: str | None = "fdr_bh",
    cond_ind_test: str = "robust_parcorr",
    newey_west_lags: Sequence[int] = (6, 12, 24),
    run_specs: Sequence[Mapping[str, object]] = COMPACT_RUN_SPECS,
) -> PreparedCaseAnalysis:
    """Prepare data, splits, conventional lag references, and design tables."""
    lag_grid = tuple(int(value) for value in lag_grid)
    min_I2_values = tuple(int(value) for value in min_I2_values)
    newey_west_lags = tuple(int(value) for value in newey_west_lags)
    if not lag_grid:
        raise ValueError("lag_grid cannot be empty.")
    if not min_I2_values:
        raise ValueError("min_I2_values cannot be empty.")
    if not newey_west_lags or any(value < 1 for value in newey_west_lags):
        raise ValueError("newey_west_lags must contain positive integers.")
    if len(set(newey_west_lags)) != len(newey_west_lags):
        raise ValueError("newey_west_lags must not contain duplicates.")

    X_full_raw = load_model_frame(
        data_path,
        include_columns=case.model_columns,
        require_complete=False,
        index_name=case.index_name,
    )
    X_case_raw = case_study_interval(
        X_full_raw,
        event_time,
        pre_days=case_pre_days,
        post_days=case_post_days,
    )
    reference_raw = pre_case_reference_interval(
        X_full_raw,
        case_start=X_case_raw.index.min(),
        reference_days=reference_days,
        min_coverage=reference_min_coverage,
        case_name=case.name,
    )
    X_model, X_decision, scaling_report = prepare_case_frames(
        X_case_raw,
        reference_raw,
        case=case,
    )

    max_lags = max(lag_grid)
    lag_references = reference_parameter_table(
        X_model,
        case.effect,
        max_lags=max_lags,
        fallback_lag=min(lag_grid),
    )

    workflow_template = WorkflowConfig(
        effect=case.effect,
        alpha=alpha,
        selected_lag=min(lag_grid),
        min_I1_length=int(min_I1_length),
        min_I2_length=min(min_I2_values),
        distribution="gaussian",
        refit_alpha=1.0,
        refit_cv=True,
        refit_cv_folds=3,
        pcmci_pc_alpha=pcmci_pc_alpha,
        pcmci_plus_pc_alpha=pcmci_plus_pc_alpha,
        pcmci_alpha_level=pcmci_alpha_level,
        pcmci_fdr_method=pcmci_fdr_method,
        pcmci_cond_ind_test=cond_ind_test,
        pcmci_plus_use_contemporaneous_triggers=False,
        newey_west_lags=newey_west_lags,
    )

    split_raw = pd.DataFrame([
        {
            "min_I2_length": min_i2,
            **split_diagnostics(
                X_decision,
                case.effect,
                event_time=event_time,
                min_I1_length=min_I1_length,
                min_I2_length=min_i2,
            ),
        }
        for min_i2 in min_I2_values
    ])
    split_raw["in_causal_grid"] = True
    split_raw["_display_order"] = range(len(split_raw))

    causal_rows, unique_splits = identify_unique_splits(
        split_raw.copy()
    )
    split_summary = (
        causal_rows
        .sort_values("_display_order", kind="stable")
        .drop(columns="_display_order")
        .reset_index(drop=True)
    )

    case_end_exclusive = X_case_raw.index.max() + pd.Timedelta("1h")
    effect_peak = X_case_raw[case.raw_effect].idxmax()
    requested_runs = len(min_I2_values) * len(lag_grid) * len(run_specs)
    unique_runs = len(unique_splits) * len(lag_grid) * len(run_specs)

    design_summary = pd.DataFrame([
        {
            "Item": "Dataset coverage",
            "Value": (
                f"{X_full_raw.index.min()} to {X_full_raw.index.max()} "
                f"({len(X_full_raw)} retained hourly rows)"
            ),
        },
        {
            "Item": "Causal-analysis interval",
            "Value": (
                f"[{X_case_raw.index.min()}, {case_end_exclusive}) "
                f"({len(X_case_raw)} h; {case_pre_days} d before and "
                f"{case_post_days} d after the event hour)"
            ),
        },
        {
            "Item": "Reference interval",
            "Value": (
                f"[{reference_raw.attrs['reference_start']}, "
                f"{reference_raw.attrs['reference_end']}); "
                f"{len(reference_raw)}/{reference_raw.attrs['expected_rows']} h "
                f"({reference_raw.attrs['coverage']:.1%} coverage)"
            ),
        },
        {"Item": event_label, "Value": str(pd.Timestamp(event_time))},
        {
            "Item": "Effect maximum",
            "Value": f"{effect_peak} ({effect_peak - pd.Timestamp(event_time)} from event)",
        },
        {
            "Item": "Minimum interval design",
            "Value": f"I1 >= {min_I1_length} h; minimum-I2 grid = {min_I2_values} h",
        },
        {
            "Item": "Maximum-lag-order design",
            "Value": f"d = {min(lag_grid)}-{max(lag_grid)} h for HMML, PCMCI, and PCMCI+",
        },
        {
            "Item": "PCMCI extension settings",
            "Value": (
                f"PCMCI pc_alpha={pcmci_pc_alpha}; "
                f"PCMCI+ graph pc_alpha={pcmci_plus_pc_alpha}; "
                f"lagged-link alpha={pcmci_alpha_level}; "
                f"lagged-link FDR={pcmci_fdr_method or 'none'}; "
                f"PCMCI+ tau=0 links exploratory without BH adjustment; "
                f"conditional-independence test={cond_ind_test}"
            ),
        },
        {
            "Item": "Interaction sensitivity",
            "Value": (
                "Classification F-test; hierarchical F-test and Newey--West "
                f"tests at truncation lags {newey_west_lags} h"
            ),
        },
        {
            "Item": "Unique response partitions",
            "Value": (
                f"{len(unique_splits)} unique partition(s) from "
                f"{len(min_I2_values)} minimum-I2 settings"
            ),
        },
        {
            "Item": "Causal run count",
            "Value": (
                f"{unique_runs} unique runs after split deduplication "
                f"({requested_runs} rows would be produced without deduplication)"
            ),
        },
    ])

    return PreparedCaseAnalysis(
        X_full_raw=X_full_raw,
        X_case_raw=X_case_raw,
        reference_raw=reference_raw,
        X_model=X_model,
        X_decision=X_decision,
        scaling_report=scaling_report,
        lag_references=lag_references,
        split_summary=split_summary,
        unique_splits=unique_splits,
        design_summary=design_summary,
        workflow_template=workflow_template,
    )
