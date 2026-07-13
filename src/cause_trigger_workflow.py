from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

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


# PCMCI+ extension that also tests eligible directed tau=0 links.
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
    """Shared Cause–Trigger experiment settings for either case study."""

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
    pcmci_pc_alpha: float = 0.05
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: Optional[str] = "fdr_bh"
    pcmci_cond_ind_test: str = "parcorr"
    pcmci_verbosity: int = 0

    # PCMCI+ contemporaneous-link settings.
    pcmci_contemp_collider_rule: str = "majority"
    pcmci_conflict_resolution: bool = True
    pcmci_plus_use_contemporaneous_triggers: bool = False


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
    Construct aligned case- and reference-standardized frames.

    ``X_model`` is case-standardized for causal discovery and moderation.
    ``X_mean`` uses the pre-case reference distribution for mean comparisons.
    """
    validate_regular_time_index(raw_case, expected_step="1h")

    if raw_case.isna().any().any():
        raise ValueError(
            f"{case.name} case interval contains missing values."
        )
    if reference_raw.empty:
        raise ValueError(f"{case.name} reference interval is empty.")
    if reference_raw.isna().any().any():
        raise ValueError(
            f"{case.name} reference interval contains missing values."
        )
    if list(reference_raw.columns) != list(raw_case.columns):
        raise ValueError(
            "Reference and case intervals must have identical column order."
        )
    if reference_raw.index.max() >= raw_case.index.min():
        raise ValueError(
            "Reference interval must end before the case interval."
        )

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
    X_mean, report = standard_scale_from_reference(
        transformed_reference,
        transformed_case,
    )

    scaled_columns = [
        f"{column}_scaled"
        for column in transformed_case.columns
    ]
    X_model.columns = scaled_columns
    X_mean.columns = scaled_columns
    X_model.index = raw_case.index
    X_mean.index = raw_case.index

    report["model_case_mean"] = X_model.mean().to_numpy()
    report["model_case_std"] = X_model.std(ddof=0).to_numpy()
    report["reference_start"] = reference_raw.index.min()
    report["reference_end"] = reference_raw.index.max()
    report["case_start"] = raw_case.index.min()
    report["case_end"] = raw_case.index.max()

    return X_model, X_mean, report


def case_study_interval(
    df_full: pd.DataFrame,
    event_time: pd.Timestamp,
    *,
    pre_days: int,
    post_hours: int,
) -> pd.DataFrame:
    """Return a complete hourly case grid around the contextual event."""
    event_hour = pd.Timestamp(event_time).floor("h")
    start = event_hour - pd.Timedelta(days=int(pre_days))
    end = event_hour + pd.Timedelta(hours=int(post_hours))
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
    """Return AIC/BIC VAR lag references; HMML remains Gaussian."""
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
    """Return one concise description of the algorithmic split."""
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
        "split_index": split_index,
        "split_time": split_time,
        "I1_length": split_index,
        "I2_length": len(dataframe) - split_index,
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
        pcmci_plus_use_contemporaneous_triggers=(
            use_contemporaneous_triggers
        ),
    )


def run_one(
    df_model: pd.DataFrame,
    df_mean: pd.DataFrame,
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
    result = run_cause_trigger(df_model, config, X_mean=df_mean)
    diagnostics = diagnostics_to_dataframe(result)
    if not diagnostics.empty:
        diagnostics.insert(0, "run", run_name)
        diagnostics.insert(1, "backend", backend)
        diagnostics.insert(2, "lag", config.lags)

    return result, diagnostics
