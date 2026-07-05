from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cause_trigger import (
    CauseTriggerConfig,
    diagnostics_to_dataframe,
    find_increase_split,
    run_cause_trigger,
)
from parameter_extraction import find_parameters


EFFECT = "effect_tremor_5_15_scaled"

DEFAULT_RUN_SPECS = (
    {"run": "hmml_backend_beta", "backend": "hmml", "refit_method": None},
    {"run": "pcmci_ridge", "backend": "pcmci", "refit_method": "ridge"},
    {"run": "pcmci_adaptive_lasso", "backend": "pcmci", "refit_method": "adaptive_lasso"},
    {"run": "pcmci_plus_ridge", "backend": "pcmci_plus", "refit_method": "ridge"},
    {"run": "pcmci_plus_adaptive_lasso", "backend": "pcmci_plus", "refit_method": "adaptive_lasso"},
)


@dataclass(frozen=True)
class WhakaariWorkflowConfig:
    """Shared experiment settings for one Whakaari Cause--Trigger analysis.

    The main notebook should keep the method generalized:
        - selected_lag is normally obtained by VAR/AIC before creating this config;
        - distribution is normally obtained by find_parameters(), with Gaussian expected
          for standardized data containing negative values;
        - the split is selected inside run_cause_trigger() from the effect series.

    event_time is kept only for plotting/context. It is not used by the algorithm.
    """

    effect: str = EFFECT
    event_time: Optional[pd.Timestamp] = None
    alpha: float = 0.05
    selected_lag: int = 1
    max_lags: int = 6
    min_I1_length: int = 48
    min_I2_length: int = 48
    distribution: str = "gaussian"
    parameter_source: str = "automatic"

    # Used only when PCMCI/PCMCI+ select parents and beta_star is refitted.
    refit_alpha: float = 1.0
    refit_cv: bool = True
    refit_cv_folds: int = 5
    refit_lasso_max_iter: int = 50_000
    adaptive_gamma: float = 1.0
    adaptive_epsilon: float = 1e-6

    # PCMCI / PCMCI+ settings.
    pcmci_pc_alpha: float = 0.05
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: Optional[str] = "fdr_bh"
    pcmci_cond_ind_test: str = "parcorr"
    pcmci_verbosity: int = 0

    # PCMCI+ settings.
    pcmci_contemp_collider_rule: str = "majority"
    pcmci_conflict_resolution: bool = True
    pcmci_keep_raw_results: bool = False
    pcmci_plus_use_contemporaneous_triggers: bool = False
    
    run_specs: Sequence[Mapping[str, object]] = field(default_factory=lambda: DEFAULT_RUN_SPECS)


# ---------------------------------------------------------------------------
# Data preparation and diagnostics
# ---------------------------------------------------------------------------


def _resolve_time_column(df: pd.DataFrame, requested_time_col: str) -> str:
    """Resolve common timestamp-column variants from saved Whakaari CSV files."""
    if requested_time_col in df.columns:
        return requested_time_col

    for candidate in ("timestamp", "time", "datetime", "date"):
        if candidate in df.columns:
            return candidate

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if len(unnamed_cols) == 1:
        return unnamed_cols[0]

    raise ValueError(
        f"Could not find a timestamp column. Requested {requested_time_col!r}; "
        f"available columns: {list(df.columns)}"
    )


def load_model_frame(
    csv_path: str | Path,
    *,
    time_col: str = "timestamp",
    drop_columns: Sequence[str] = (),
    include_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a final Whakaari CSV as a numeric, time-indexed modelling dataframe.

    Expected input is the standardized Whakaari dataframe written by
    save_whakaari_datasets(), i.e. timestamp index plus *_scaled variables.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    resolved_time_col = _resolve_time_column(df, time_col)

    df[resolved_time_col] = pd.to_datetime(df[resolved_time_col], utc=True, errors="coerce")
    if df[resolved_time_col].isna().any():
        bad = int(df[resolved_time_col].isna().sum())
        raise ValueError(f"Found {bad} invalid timestamps in {csv_path}")

    if include_columns is not None:
        keep = [resolved_time_col, *include_columns]
        missing = sorted(set(keep) - set(df.columns))
        if missing:
            raise ValueError(f"Missing requested columns in {csv_path}: {missing}")
        df = df[keep]

    model = (
        df.drop(columns=list(drop_columns), errors="ignore")
        .set_index(resolved_time_col)
        .sort_index()
    )
    model.index.name = "timestamp"

    model = model.select_dtypes(include=[np.number])

    if not model.index.is_monotonic_increasing:
        raise ValueError("Model dataframe index is not sorted.")

    if model.index.has_duplicates:
        duplicates = model.index[model.index.duplicated()].unique()[:5]
        raise ValueError(f"Model dataframe has duplicate timestamps, e.g. {list(duplicates)}")

    if model.isna().any().any():
        missing = model.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        raise ValueError(f"Model dataframe contains NaNs:\n{missing}")

    return model


def model_overview(df: pd.DataFrame, effect: str) -> dict:
    """Return compact dataframe/target diagnostics for notebook display."""
    if effect not in df.columns:
        raise ValueError(f"Effect variable {effect!r} not found. Available: {list(df.columns)}")

    frequency = pd.infer_freq(df.index)
    target = df[effect]

    return {
        "n_rows": int(len(df)),
        "n_variables": int(df.shape[1]),
        "start": df.index.min(),
        "end": df.index.max(),
        "inferred_frequency": frequency,
        "effect": effect,
        "effect_min": float(target.min()),
        "effect_median": float(target.median()),
        "effect_mean": float(target.mean()),
        "effect_max": float(target.max()),
        "effect_abs_mean": float(target.abs().mean()),
        "effect_iqr": float(target.quantile(0.75) - target.quantile(0.25)),
        "effect_n_unique": int(target.nunique()),
    }


def target_extremes(df: pd.DataFrame, effect: str, n: int = 10) -> pd.DataFrame:
    """Return largest absolute effect values for optional debugging."""
    if effect not in df.columns:
        raise ValueError(f"Effect variable {effect!r} not found.")

    out = df[[effect]].copy()
    out["abs_effect"] = out[effect].abs()
    return out.sort_values("abs_effect", ascending=False).head(n)


def select_parameters(
    df: pd.DataFrame,
    effect: str,
    *,
    max_lags: int = 6,
    criterion: str = "aic",
    fallback_lag: int = 1,
    fallback_distribution: str = "gaussian",
) -> dict:
    """Select lag and distribution for the Whakaari Cause--Trigger run.

    Lag is selected using VAR order selection. Distribution is selected by
    find_parameters(); for standardized data with negative values, it will return
    Gaussian by design.
    """
    if effect not in df.columns:
        raise ValueError(f"Effect variable {effect!r} not found.")

    selected_distribution, selected_lag = find_parameters(
        X=df,
        target_series=df[effect],
        max_lags=max_lags,
        criterion=criterion,
        fallback_lag=fallback_lag,
        fallback_distribution=fallback_distribution,
    )

    return {
        "effect": effect,
        "selected_lag": int(selected_lag),
        "selected_distribution": selected_distribution,
        "lag_method": f"VAR-{criterion.upper()}",
        "max_lags": int(max_lags),
        "fallback_lag": int(fallback_lag),
        "fallback_distribution": fallback_distribution,
    }


def select_lag_by_var(
    df: pd.DataFrame,
    effect: str,
    *,
    max_lags: int = 6,
    criterion: str = "aic",
    fallback_lag: int = 1,
) -> int:
    """Select only the VAR lag; kept for parity with etna_cause_trigger.py."""
    params = select_parameters(
        df,
        effect,
        max_lags=max_lags,
        criterion=criterion,
        fallback_lag=fallback_lag,
        fallback_distribution="gaussian",
    )
    return int(params["selected_lag"])


def split_diagnostics(
    df: pd.DataFrame,
    effect: str,
    *,
    event_time: Optional[pd.Timestamp] = None,
    min_I1_length: int = 48,
    min_I2_length: int = 48,
) -> dict:
    """Inspect the automatic paper-style split used by find_increase_split().

    event_time is not used for the split. It is returned only as temporal context.
    """
    split_idx = find_increase_split(
        df[effect],
        min_I1_length=min_I1_length,
        min_I2_length=min_I2_length,
    )

    if split_idx is None:
        return {
            "effect": effect,
            "split_index": None,
            "split_time": None,
            "I1_length": None,
            "I2_length": None,
            "abs_mean_I1": None,
            "abs_mean_I2": None,
            "abs_mean_difference": None,
            "event_time": event_time,
            "distance_to_event": None,
            "boundary_split": None,
        }

    I1 = df.iloc[:split_idx]
    I2 = df.iloc[split_idx:]
    split_time = df.index[split_idx]
    boundary_split = split_idx == len(df) - min_I2_length

    distance_to_event = None
    if event_time is not None:
        distance_to_event = split_time - pd.Timestamp(event_time)

    return {
        "effect": effect,
        "split_index": int(split_idx),
        "split_time": split_time,
        "I1_length": int(len(I1)),
        "I2_length": int(len(I2)),
        "abs_mean_I1": float(abs(I1[effect].mean())),
        "abs_mean_I2": float(abs(I2[effect].mean())),
        "abs_mean_difference": float(abs(I2[effect].mean()) - abs(I1[effect].mean())),
        "event_time": event_time,
        "distance_to_event": distance_to_event,
        "boundary_split": bool(boundary_split),
    }


# ---------------------------------------------------------------------------
# Running Cause--Trigger configurations
# ---------------------------------------------------------------------------


def default_weighting_for_backend(backend: str, refit_method: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Resolve how beta_star is constructed for each causal-discovery backend."""
    if backend == "hmml":
        return "backend", None

    if backend in {"pcmci", "pcmci_plus"}:
        return "refit", refit_method or "ridge"

    raise ValueError(f"Unknown backend: {backend!r}")


def make_cause_trigger_config(
    workflow: WhakaariWorkflowConfig,
    *,
    backend: str,
    lag: Optional[int] = None,
    distribution: Optional[str] = None,
    parameter_source: Optional[str] = None,
    cond_ind_test: Optional[str] = None,
    refit_method: Optional[str] = None,
    use_contemporaneous_triggers: Optional[bool] = None,
) -> CauseTriggerConfig:
    """Create one CauseTriggerConfig without relying on notebook globals."""
    lag = workflow.selected_lag if lag is None else int(lag)
    distribution = workflow.distribution if distribution is None else distribution
    cond_ind_test = workflow.pcmci_cond_ind_test if cond_ind_test is None else cond_ind_test
    v_weighting, resolved_refit_method = default_weighting_for_backend(backend, refit_method)
    
    if use_contemporaneous_triggers is None:
        use_contemporaneous_triggers = workflow.pcmci_plus_use_contemporaneous_triggers
    
    if parameter_source is None:
        suffix = "backend_beta" if backend == "hmml" else f"{resolved_refit_method}_refit"
        parameter_source = f"{workflow.parameter_source}_{backend}_lag{lag}_{distribution}_{suffix}"

    return CauseTriggerConfig(
        y_t=workflow.effect,
        lags=lag,
        distribution=distribution,
        causal_backend=backend,
        alpha=workflow.alpha,
        min_I1_length=workflow.min_I1_length,
        min_I2_length=workflow.min_I2_length,
        parameter_source=parameter_source,
        selected_distribution=distribution,
        selected_lag=lag,
        v_weighting=v_weighting,
        refit_method=resolved_refit_method or "ridge",
        refit_alpha=workflow.refit_alpha,
        refit_cv=workflow.refit_cv,
        refit_cv_folds=workflow.refit_cv_folds,
        refit_lasso_max_iter=workflow.refit_lasso_max_iter,
        adaptive_gamma=workflow.adaptive_gamma,
        adaptive_epsilon=workflow.adaptive_epsilon,
        pcmci_pc_alpha=workflow.pcmci_pc_alpha,
        pcmci_alpha_level=workflow.pcmci_alpha_level,
        pcmci_fdr_method=workflow.pcmci_fdr_method,
        pcmci_cond_ind_test=cond_ind_test,
        pcmci_verbosity=workflow.pcmci_verbosity,
        pcmci_contemp_collider_rule=workflow.pcmci_contemp_collider_rule,
        pcmci_conflict_resolution=workflow.pcmci_conflict_resolution,
        pcmci_keep_raw_results=workflow.pcmci_keep_raw_results,
        pcmci_plus_use_contemporaneous_triggers=use_contemporaneous_triggers,
    )


def result_to_row(run_name: str, result: dict, effect: str) -> dict:
    """Convert one Cause--Trigger result dictionary into a compact comparison row."""
    refit_metadata = result.get("refit_metadata", {}) or {}
    return {
        "run": run_name,
        "effect": effect,
        "backend": result.get("backend"),
        "parameter_source": result.get("parameter_source"),
        "configured_lags": result.get("configured_lags"),
        "configured_distribution": result.get("configured_distribution"),
        "v_weighting": result.get("v_weighting"),
        "refit_method": result.get("refit_method"),
        "refit_nonzero_beta_count": refit_metadata.get("nonzero_beta_count"),
        "refit_alpha_used": refit_metadata.get("refit_alpha_used"),
        "split_index": result.get("split_index"),
        "split_timestamp": result.get("split_timestamp"),
        "I1_length": result.get("I1_length"),
        "I2_length": result.get("I2_length"),
        "target_abs_mean_I1": result.get("target_abs_mean_I1"),
        "target_abs_mean_I2": result.get("target_abs_mean_I2"),
        "target_abs_mean_difference": result.get("target_abs_mean_difference"),
        "B1": result.get("B_1"),
        "B2": result.get("B_2"),
        "autoregressive_effect_parent_in_B2": result.get("autoregressive_parent_in_B2"),
        "trigger_candidates": result.get("T_candidates"),
        "accepted_triggers": result.get("T"),
        "causes": result.get("C"),
        "pairs": result.get("pairs"),
        "n_contemporaneous_links": len(result.get("contemporaneous_links", {}) or {}),
        "stop_reason": result.get("stop_reason"),
    }


def label_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    run_name: str,
    effect: str,
    result: dict,
) -> pd.DataFrame:
    """Add run metadata to a diagnostics dataframe."""
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()

    out = diagnostics.copy()
    out["run"] = run_name
    out["effect"] = effect
    out["backend"] = result.get("backend")
    out["parameter_source"] = result.get("parameter_source")
    out["v_weighting"] = result.get("v_weighting")
    out["refit_method"] = result.get("refit_method")
    return out


def run_one(
    df: pd.DataFrame,
    workflow: WhakaariWorkflowConfig,
    *,
    run_name: str,
    backend: str,
    lag: Optional[int] = None,
    distribution: Optional[str] = None,
    cond_ind_test: Optional[str] = None,
    refit_method: Optional[str] = None,
    parameter_source: Optional[str] = None,
    use_contemporaneous_triggers: Optional[bool] = None,
) -> tuple[dict, pd.DataFrame, dict]:
    """Run one backend and return result, diagnostics, and comparison row."""
    config = make_cause_trigger_config(
        workflow,
        backend=backend,
        lag=lag,
        distribution=distribution,
        parameter_source=parameter_source,
        cond_ind_test=cond_ind_test,
        refit_method=refit_method,
        use_contemporaneous_triggers=use_contemporaneous_triggers,
    )
    result = run_cause_trigger(df, config)
    diagnostics = diagnostics_to_dataframe(result)
    row = result_to_row(run_name, result, workflow.effect)
    return result, diagnostics, row


def run_suite(
    df: pd.DataFrame,
    workflow: WhakaariWorkflowConfig,
    *,
    run_specs: Optional[Sequence[Mapping[str, object]]] = None,
    lag: Optional[int] = None,
    distribution: Optional[str] = None,
    cond_ind_test: Optional[str] = None,
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Run a compact backend suite and return results, comparison table, diagnostics table."""
    run_specs = workflow.run_specs if run_specs is None else run_specs
    results: dict[str, dict] = {}
    comparison_rows = []
    diagnostics_frames = []

    for spec in run_specs:
        run_name = str(spec["run"])
        backend = str(spec["backend"])
        refit_method = spec.get("refit_method")
        refit_method = None if refit_method is None else str(refit_method)

        result, diag, row = run_one(
            df,
            workflow,
            run_name=run_name,
            backend=backend,
            lag=lag,
            distribution=distribution,
            cond_ind_test=cond_ind_test,
            refit_method=refit_method,
        )
        results[run_name] = result
        comparison_rows.append(row)
        diagnostics_frames.append(
            label_diagnostics(diag, run_name=run_name, effect=workflow.effect, result=result)
        )

    comparison = pd.DataFrame(comparison_rows)
    diagnostics = (
        pd.concat(diagnostics_frames, ignore_index=True)
        if any(not frame.empty for frame in diagnostics_frames)
        else pd.DataFrame()
    )
    return results, comparison, diagnostics


def run_sensitivity_grid(
    df: pd.DataFrame,
    workflow: WhakaariWorkflowConfig,
    *,
    run_specs: Optional[Sequence[Mapping[str, object]]] = None,
    lags: Iterable[int] = (1, 2, 3),
    distributions: Iterable[str] = ("gaussian",),
    cond_ind_test: Optional[str] = None,
) -> pd.DataFrame:
    """Run backend/lag/distribution sensitivity checks with safe error reporting."""
    run_specs = workflow.run_specs if run_specs is None else run_specs
    cond_ind_test = workflow.pcmci_cond_ind_test if cond_ind_test is None else cond_ind_test
    rows = []

    for spec in run_specs:
        run_name = str(spec["run"])
        backend = str(spec["backend"])
        refit_method = spec.get("refit_method")
        refit_method = None if refit_method is None else str(refit_method)

        for lag in lags:
            for distribution in distributions:
                base_row = {
                    "effect": workflow.effect,
                    "backend": backend,
                    "lag": int(lag),
                    "distribution": distribution,
                    "cond_ind_test": cond_ind_test if backend in {"pcmci", "pcmci_plus"} else None,
                    "run": run_name,
                }
                try:
                    result, diagnostics, _ = run_one(
                        df,
                        workflow,
                        run_name=run_name,
                        backend=backend,
                        lag=int(lag),
                        distribution=distribution,
                        cond_ind_test=cond_ind_test,
                        refit_method=refit_method,
                        parameter_source=f"{run_name}_sensitivity_grid",
                    )
                    row = {
                        **base_row,
                        "v_weighting": result.get("v_weighting"),
                        "refit_method": result.get("refit_method"),
                        "refit_nonzero_beta_count": (result.get("refit_metadata", {}) or {}).get("nonzero_beta_count"),
                        "refit_alpha_used": (result.get("refit_metadata", {}) or {}).get("refit_alpha_used"),
                        "split_timestamp": result.get("split_timestamp"),
                        "I1_length": result.get("I1_length"),
                        "I2_length": result.get("I2_length"),
                        "target_abs_mean_difference": result.get("target_abs_mean_difference"),
                        "B_1": result.get("B_1"),
                        "B_2": result.get("B_2"),
                        "autoregressive_effect_parent_in_B2": result.get("autoregressive_parent_in_B2"),
                        "trigger_candidates": result.get("T_candidates"),
                        "accepted_triggers": result.get("T"),
                        "causes": result.get("C"),
                        "pairs": result.get("pairs"),
                        "n_contemporaneous_links": len(result.get("contemporaneous_links", {}) or {}),
                        "n_diagnostics": int(len(diagnostics)),
                        "min_p_value": diagnostics["p_value"].min()
                        if "p_value" in diagnostics.columns and not diagnostics.empty else None,
                        "max_rss_reduction_ratio": diagnostics["rss_reduction_ratio"].max()
                        if "rss_reduction_ratio" in diagnostics.columns and not diagnostics.empty else None,
                        "error": None,
                    }
                except Exception as exc:
                    row = {**base_row, "error": str(exc)}
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compact display helpers for concise notebooks
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS = [
    "run",
    "backend",
    "configured_lags",
    "configured_distribution",
    "split_timestamp",
    "I1_length",
    "I2_length",
    "B2",
    "trigger_candidates",
    "accepted_triggers",
    "causes",
    "pairs",
    "stop_reason",
]

DIAGNOSTIC_COLUMNS = [
    "run",
    "trigger",
    "cause",
    "accepted",
    "p_value",
    "f_stat",
    "critical_f",
    "rss_reduction_ratio",
    "gamma_2",
    "backend",
]

SENSITIVITY_COLUMNS = [
    "run",
    "backend",
    "lag",
    "distribution",
    "split_timestamp",
    "I1_length",
    "I2_length",
    "B_2",
    "trigger_candidates",
    "accepted_triggers",
    "causes",
    "pairs",
    "error",
]


def compact_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the main result columns for a concise notebook display."""
    cols = [c for c in SUMMARY_COLUMNS if c in comparison.columns]
    return comparison[cols].copy()


def compact_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return the most interpretable diagnostics columns."""
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()
    cols = [c for c in DIAGNOSTIC_COLUMNS if c in diagnostics.columns]
    return diagnostics[cols].copy()


def compact_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Return the main sensitivity columns for a concise notebook display."""
    if sensitivity is None or sensitivity.empty:
        return pd.DataFrame()
    cols = [c for c in SENSITIVITY_COLUMNS if c in sensitivity.columns]
    return sensitivity[cols].copy()


def accepted_sensitivity_rows(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Filter sensitivity rows with at least one accepted trigger."""
    if sensitivity is None or sensitivity.empty or "accepted_triggers" not in sensitivity.columns:
        return pd.DataFrame()

    mask = sensitivity["accepted_triggers"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    return compact_sensitivity(sensitivity.loc[mask])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_effect_with_split(
    df: pd.DataFrame,
    effect: str,
    result_or_split: Mapping[str, object],
    *,
    event_time: Optional[pd.Timestamp] = None,
    title: Optional[str] = None,
) -> None:
    """Plot an effect series with detected split and optional eruption time.

    The eruption/event line is contextual only; it is not used by the algorithmic split.
    """
    plt.figure(figsize=(16, 4))
    plt.plot(df.index, df[effect], label=effect, linewidth=0.8)

    split_timestamp = result_or_split.get("split_timestamp") or result_or_split.get("split_time")
    if split_timestamp is not None:
        plt.axvline(pd.Timestamp(split_timestamp), linestyle="--", label="Detected split")

    if event_time is not None:
        plt.axvline(pd.Timestamp(event_time), linestyle=":", label="Known eruption time")

    plt.title(title or f"{effect}: automatic split with eruption time shown for context")
    plt.xlabel("Time")
    plt.ylabel("Scaled value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_selected_variables(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    event_time: Optional[pd.Timestamp] = None,
    split_time: Optional[pd.Timestamp] = None,
    title: Optional[str] = None,
) -> None:
    """Plot selected scaled variables for quick qualitative inspection."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    plt.figure(figsize=(16, 5))
    for col in columns:
        plt.plot(df.index, df[col], linewidth=0.8, label=col)

    if split_time is not None:
        plt.axvline(pd.Timestamp(split_time), linestyle="--", label="Detected split")

    if event_time is not None:
        plt.axvline(pd.Timestamp(event_time), linestyle=":", label="Known eruption time")

    plt.title(title or "Selected Whakaari variables")
    plt.xlabel("Time")
    plt.ylabel("Scaled value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
