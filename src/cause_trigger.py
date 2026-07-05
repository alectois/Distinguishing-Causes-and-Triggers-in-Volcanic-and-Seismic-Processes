"""
Cause–Trigger algorithm implementation for thesis experiments.

This file is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Zenodo. DOI: 10.5281/zenodo.15109084

Original material: CC BY 4.0.
Modifications: refactoring, paper-compatible statistic update, volcanic-data adaptation,
diagnostic outputs, and optional causal-discovery backends.

Given a standardized time-series dataframe X and a target y_t,
return causes, trigger candidates, accepted triggers, cause-trigger pairs, and diagnostics.
""" 

import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from hmml_runner import HMMLRunner
from pcmci_runner import PCMCIBackend

@dataclass
class CauseTriggerConfig:
    y_t: str
    lags: int
    distribution: str = "gaussian"
    alpha: float = 0.05

    # Original experiments constrained I2; thesis adaptation also constrains I1.
    min_I1_length: int = 12
    min_I2_length: int = 24

    # Split selection.
    # I1 = X.iloc[:split_index]
    # I2 = X.iloc[split_index:]
    # split_index maximizes abs(mean(y in I2)) - abs(mean(y in I1)),
    # subject to min_I1_length and min_I2_length.
    min_I1_length: int = 12
    min_I2_length: int = 24

    # How to construct beta_star for V = X_without-trigger @ beta_star
    # "backend" = use beta from backend; baseline for HMML
    # "refit"   = refit beta after parent selection; for PCMCI/PCMCI+
    v_weighting: str = "backend"

    # Used only when v_weighting="refit".
    # "ridge"          = stable coefficient refit under collinearity
    refit_alpha: float = 1.0
    refit_cv: bool = True
    refit_cv_folds: int = 5

    parameter_source: str = "manual"
    selected_distribution: str = None
    selected_lag: int = None

    # causal discovery backend
    # "hmml"        = paper-compatible HMML baseline
    # "pcmci"       = lagged PCMCI backend
    # "pcmci_plus"  = PCMCI+ backend; tau=0 links are stored as diagnostics,
    #                 but only tau>=1 links are used for B2 in the Cause–Trigger test.
    causal_backend: str = "hmml"

    # PCMCI / PCMCI+ settings
    pcmci_pc_alpha: float = 0.05
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: str = "fdr_bh"
    pcmci_cond_ind_test: str = "parcorr"
    pcmci_verbosity: int = 0

    # PCMCI+ settings
    pcmci_contemp_collider_rule: str = "majority"
    pcmci_conflict_resolution: bool = True
    pcmci_keep_raw_results: bool = False

    # If True, PCMCI+ tau=0 source-target links are allowed as trigger candidates.
    # They are not added to B2 and are not used to construct V.
    pcmci_plus_use_contemporaneous_triggers: bool = False

@dataclass
class BackendResult:
    parents: list
    adjacency: np.ndarray
    beta: pd.DataFrame
    lags: dict
    scores: dict
    raw_results: object = None


class HMMLBackend:
    def __init__(self, lags: int, distribution: str):
        self.runner = HMMLRunner(lags=lags, distribution=distribution)

    def discover(self, X: pd.DataFrame, y_t: str) -> BackendResult:
        beta, adjacency = self.runner.get_betas_and_adjacency(X, y_t=y_t)
        scores = {
            "_hmml_metadata": {
                "requested_distribution": self.runner.requested_distribution,
                "used_distribution": self.runner.used_distribution,
                "used_fallback": self.runner.used_fallback,
            }
        }
        parents = X.columns[np.where(adjacency == 1)[0]].to_list()

        return BackendResult(
            parents=parents,
            adjacency=adjacency,
            beta=beta,
            lags={},
            scores=scores,
            raw_results=None,
        )


def make_causal_backend(config: CauseTriggerConfig):
    if config.causal_backend == "hmml":
        return HMMLBackend(
            lags=config.lags,
            distribution=config.distribution,
        )

    if config.causal_backend in {"pcmci", "pcmci_plus"}:
        if config.v_weighting == "backend":
            warnings.warn(
                "PCMCI/PCMCI+ do not return HMML-style structural beta coefficients. "
                "Their beta-like matrix stores conditional-dependence/test-statistic strengths, "
                "not β* for V = X_without-trigger @ β*. "
                "For PCMCI/PCMCI+ thesis extensions, use v_weighting='refit'.",
                UserWarning,
            )

        return PCMCIBackend(
            tau_max=config.lags,
            pc_alpha=config.pcmci_pc_alpha,
            alpha_level=config.pcmci_alpha_level,
            fdr_method=config.pcmci_fdr_method,
            cond_ind_test=config.pcmci_cond_ind_test,
            verbosity=config.pcmci_verbosity,
            keep_raw_results=config.pcmci_keep_raw_results,
            method=config.causal_backend,
            contemp_collider_rule=config.pcmci_contemp_collider_rule,
            conflict_resolution=config.pcmci_conflict_resolution,
        )

    raise ValueError(
        f"Unknown causal_backend={config.causal_backend!r}. "
        "Use 'hmml', 'pcmci', or 'pcmci_plus'."
    )


def _to_position(index, value, side="left"):
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if getattr(index, "tz", None) is not None and ts.tzinfo is None:
        ts = ts.tz_localize(index.tz)
    elif getattr(index, "tz", None) is not None:
        ts = ts.tz_convert(index.tz)

    return int(index.searchsorted(ts, side=side))


def _mean_difference(interval_1: pd.Series, interval_2: pd.Series, direction: str):
    mean_1 = float(interval_1.mean())
    mean_2 = float(interval_2.mean())

    if direction == "increase":
        difference = mean_2 - mean_1
    elif direction == "absolute_mean":
        difference = abs(mean_2) - abs(mean_1)
    else:
        raise ValueError(
            f"Unknown split_direction={direction!r}. "
            "Use 'increase' or 'absolute_mean'."
        )

    return difference, mean_1, mean_2


def _welch_score(interval_1: pd.Series, interval_2: pd.Series, direction: str):
    difference, mean_1, mean_2 = _mean_difference(interval_1, interval_2, direction)

    n1 = len(interval_1)
    n2 = len(interval_2)

    var_1 = float(interval_1.var(ddof=1))
    var_2 = float(interval_2.var(ddof=1))

    standard_error = np.sqrt((var_1 / max(n1, 1)) + (var_2 / max(n2, 1)))
    standard_error = max(float(standard_error), 1e-12)

    score = difference / standard_error

    return score, difference, mean_1, mean_2


def find_effect_split(
    y: pd.Series,
    min_I1_length: int = 12,
    min_I2_length: int = 30,
    *,
    return_info: bool = False,
):
    """
    Paper-compatible Cause--Trigger split.

    Finds I1=(start, split_index) and I2=[split_index, end)
    such that abs(E[y]_I2) > abs(E[y]_I1), selecting the split
    that maximizes the absolute-mean difference.
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce")

    if y.isna().any():
        raise ValueError("find_effect_split received NaNs in the target series.")

    n = len(y)
    lower = int(min_I1_length)
    upper = n - int(min_I2_length)

    best = None
    best_score = float("-inf")

    for split_index in range(lower, upper + 1):
        interval_1 = y.iloc[:split_index]
        interval_2 = y.iloc[split_index:]

        mean_1 = float(interval_1.mean())
        mean_2 = float(interval_2.mean())

        abs_mean_1 = abs(mean_1)
        abs_mean_2 = abs(mean_2)

        score = abs_mean_2 - abs_mean_1

        if score <= 0:
            continue

        if score > best_score:
            best_score = score
            best = {
                "split_index": int(split_index),
                "split_end_index": int(n),
                "score": float(score),
                "target_mean_I1": mean_1,
                "target_mean_I2": mean_2,
                "target_abs_mean_I1": abs_mean_1,
                "target_abs_mean_I2": abs_mean_2,
                "target_abs_mean_difference": float(score),
                "I1_length": int(len(interval_1)),
                "I2_length": int(len(interval_2)),
                "boundary_split": bool(split_index == n - min_I2_length),
            }

    if best is None:
        out = {
            "split_index": None,
            "split_end_index": None,
            "score": None,
            "target_mean_I1": None,
            "target_mean_I2": None,
            "target_abs_mean_I1": None,
            "target_abs_mean_I2": None,
            "target_abs_mean_difference": None,
            "I1_length": None,
            "I2_length": None,
            "boundary_split": None,
        }
        return out if return_info else None

    return best if return_info else best["split_index"]

def find_increase_split(
    y: pd.Series,
    min_I1_length: int = 12,
    min_I2_length: int = 30,
):
    return find_effect_split(
        y,
        min_I1_length=min_I1_length,
        min_I2_length=min_I2_length,
        return_info=False,
    )

def residual_sum_of_squares(y_true, X_features, model):
    y_hat = model.predict(X_features)
    return np.sum((y_true - y_hat) ** 2)

def build_lagged_design_matrix_for_V(X_values, beta_values, lags, beta_is_ones=False):
    """
    May 16 paper-compatible construction of V.

    X_without-trigger has dimension:
        (n - d) x ((m - 1) * d)

    beta_star has dimension:
        ((m - 1) * d) x 1

    V = X_without-trigger @ beta_star
      has dimension:
        (n - d) x 1

    Column order used here is lag-major:
        [all variables at lag 1, all variables at lag 2, ..., all variables at lag d]

    Therefore beta_values must be shaped (d, p), with rows Lag_1...Lag_d
    and columns in the same order as X_values columns. The row-major flattening
    beta_values.reshape(-1, 1) then matches the design matrix exactly.
    """
    X_values = np.asarray(X_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)

    if X_values.ndim != 2:
        raise ValueError("X_values must be a 2D array")

    n, p = X_values.shape

    if n <= lags:
        raise ValueError(f"Need len(I2) > lags, got len={n}, lags={lags}")

    if p == 0:
        raise ValueError("Cannot build V with zero non-trigger variables")

    if beta_values.shape != (lags, p):
        raise ValueError(
            f"beta_values must have shape (lags, variables)=({lags}, {p}), "
            f"got {beta_values.shape}"
        )

    lagged_blocks = []
    for k in range(1, lags + 1):
        # aligned with y[d:], this block is X(t-k)
        X_lag_k = X_values[lags - k : n - k, :]
        lagged_blocks.append(X_lag_k)

    X_without_trigger_lagged = np.hstack(lagged_blocks)

    if beta_is_ones:
        beta_star = np.ones((X_without_trigger_lagged.shape[1], 1))
    else:
        beta_star = beta_values.reshape(-1, 1)

    V = X_without_trigger_lagged @ beta_star

    return V, X_without_trigger_lagged, beta_star

def build_lagged_matrix_for_refit(X_values, lags):
    """
    Build lagged matrix with the same column order used by
    build_lagged_design_matrix_for_V:

        [all variables at lag 1, all variables at lag 2, ..., all variables at lag d]

    Returns shape:
        (n - d) x (p * d)
    """
    X_values = np.asarray(X_values, dtype=float)

    if X_values.ndim != 2:
        raise ValueError("X_values must be a 2D array")

    n, p = X_values.shape

    if n <= lags:
        raise ValueError(f"Need len(X) > lags, got len={n}, lags={lags}")

    lagged_blocks = []
    for k in range(1, lags + 1):
        X_lag_k = X_values[lags - k : n - k, :]
        lagged_blocks.append(X_lag_k)

    return np.hstack(lagged_blocks)


def refit_beta_for_selected_parents(
    X: pd.DataFrame,
    y_t: str,
    selected_parents: list,
    lags: int,
    alpha: float = 1.0,
    cv: bool = True,
    cv_folds: int = 5,
):
    """
    Ridge refit for selected lagged parents.

    Used for PCMCI/PCMCI+ extensions because those backends select parents
    but do not provide HMML-style structural beta coefficients.
    """
    selected_parents = list(selected_parents)
    index = [f"Lag_{lag}" for lag in range(1, lags + 1)]

    if len(selected_parents) == 0:
        beta_df = pd.DataFrame(
            np.zeros((lags, 0)),
            index=index,
            columns=[],
        )
        return beta_df, {
            "refit_method": "ridge",
            "reason": "no selected parents",
            "nonzero_beta_count": 0,
        }

    X_values = X[selected_parents].to_numpy(dtype=float)
    y_values = X[y_t].iloc[lags:].to_numpy(dtype=float)
    X_lagged = build_lagged_matrix_for_refit(X_values, lags)

    if len(y_values) != X_lagged.shape[0]:
        raise ValueError(
            f"Refit length mismatch: len(y)={len(y_values)}, "
            f"X_lagged rows={X_lagged.shape[0]}"
        )

    n_splits = min(cv_folds, X_lagged.shape[0] - 1)
    use_cv = bool(cv and n_splits >= 2)
    cv_obj = TimeSeriesSplit(n_splits=n_splits) if use_cv else None

    ridge_alphas = np.logspace(-4, 4, 40)

    if use_cv:
        model = RidgeCV(alphas=ridge_alphas, cv=cv_obj)
    else:
        model = Ridge(alpha=alpha)

    model.fit(X_lagged, y_values)

    beta_flat = np.asarray(model.coef_, dtype=float).reshape(-1)

    if beta_flat.shape[0] != X_lagged.shape[1]:
        raise ValueError(
            f"Refit beta length mismatch: got {beta_flat.shape[0]}, "
            f"expected {X_lagged.shape[1]}"
        )

    beta_matrix = beta_flat.reshape(lags, len(selected_parents))

    beta_df = pd.DataFrame(
        beta_matrix,
        index=index,
        columns=selected_parents,
    )

    metadata = {
        "refit_method": "ridge",
        "refit_alpha_requested": float(alpha),
        "refit_alpha_used": (
            float(model.alpha_) if hasattr(model, "alpha_") else float(alpha)
        ),
        "refit_cv": bool(use_cv),
        "refit_cv_folds": int(n_splits) if use_cv else 0,
        "n_refit_rows": int(X_lagged.shape[0]),
        "n_refit_features": int(X_lagged.shape[1]),
        "selected_parents": selected_parents,
        "nonzero_beta_count": int(np.count_nonzero(beta_flat)),
        "beta_abs_max": float(np.max(np.abs(beta_flat))) if beta_flat.size else 0.0,
    }

    return beta_df, metadata

def f_statistic(rss_reduced, rss_full, n, d):
    """
    May 16 paper F-statistic for reduced Eq. (3) vs full Eq. (4).

    Eq. (3), reduced model:
        y_t = gamma_0 + gamma_1 V^t + eps_t

    Eq. (4), full model:
        y_t = gamma_0 + gamma_1 V^t + gamma_2 V^t x_s^t + eps_t

    May-16 notation:
        RSS1 = RSS_reduced from Eq. (3)
        RSS2 = RSS_full from Eq. (4)

        S = (RSS1 - RSS2) * (n - d - 3) / RSS2

    Critical value uses F_{1, n - d - 3}(1 - alpha).
    """
    denominator_df = n - d - 3

    if denominator_df <= 0:
        raise ValueError(
            f"Invalid May-16 F-test degrees of freedom: n - d - 3 = {denominator_df}. "
            f"Need len(I2) > lags + 3; got n={n}, d={d}."
        )

    if rss_full <= 0:
        raise ValueError(
            f"Full-model RSS must be positive for the F-statistic, got rss_full={rss_full}."
        )

    return ((rss_reduced - rss_full) * denominator_df) / rss_full


def test_moderation(y_response, V, x_s_values, n_I2, d, alpha=0.05):
    """
    May 16 paper-compatible moderation test.

    Reduced Eq. (3):
        y ~ V

    Full Eq. (4):
        y ~ V + V * x_s

    Here V = X_without-trigger @ beta_star has shape (n-d, 1), and
    x_s_values is contemporaneous x_s aligned with y[d:].
    """
    V = np.asarray(V, dtype=float)
    x_s_values = np.asarray(x_s_values, dtype=float)
    y_response = np.asarray(y_response, dtype=float).reshape(-1)

    if V.ndim != 2 or V.shape[1] != 1:
        raise ValueError(f"V must have shape (n-d, 1); got {V.shape}")

    if x_s_values.ndim != 2 or x_s_values.shape[1] != 1:
        raise ValueError(f"x_s_values must have shape (n-d, 1); got {x_s_values.shape}")

    if len(y_response) != V.shape[0] or len(y_response) != x_s_values.shape[0]:
        raise ValueError(
            "Mismatched response and feature lengths: "
            f"len(y)={len(y_response)}, V_rows={V.shape[0]}, x_rows={x_s_values.shape[0]}"
        )

    denominator_df = n_I2 - d - 3
    if denominator_df <= 0:
        raise ValueError(
            f"Invalid May-16 F-test degrees of freedom: n - d - 3 = {denominator_df}. "
            f"Need len(I2) > lags + 3; got n={n_I2}, d={d}."
        )

    if np.nanstd(V) == 0:
        raise ValueError("V is constant; moderation regression is not meaningful.")

    reduced_features = V
    interaction_feature = V * x_s_values
    full_features = np.hstack([V, interaction_feature])

    reduced_model = LinearRegression().fit(reduced_features, y_response)
    full_model = LinearRegression().fit(full_features, y_response)

    rss_reduced = residual_sum_of_squares(y_response, reduced_features, reduced_model)
    rss_full = residual_sum_of_squares(y_response, full_features, full_model)

    rss_reduction = rss_reduced - rss_full
    rss_reduction_ratio = rss_reduction / rss_reduced if rss_reduced != 0 else np.nan

    f_stat = f_statistic(
        rss_reduced=rss_reduced,
        rss_full=rss_full,
        n=n_I2,
        d=d,
    )

    critical_f = f.ppf(1 - alpha, dfn=1, dfd=denominator_df)
    p_value = 1 - f.cdf(f_stat, dfn=1, dfd=denominator_df)

    gamma_1 = full_model.coef_[0]
    gamma_2 = full_model.coef_[1]

    return {
        "accepted": bool(f_stat > critical_f),
        "f_stat": float(f_stat),
        "critical_f": float(critical_f),
        "p_value": float(p_value),
        "rss_reduced": float(rss_reduced),
        "rss_full": float(rss_full),
        "rss_reduction": float(rss_reduction),
        "rss_reduction_ratio": float(rss_reduction_ratio),
        "gamma_1": float(gamma_1),
        "gamma_2": float(gamma_2),
        "n_I2": int(n_I2),
        "d": int(d),
        "dfn": 1,
        "dfd": int(denominator_df),
        "n_regression_rows": int(len(y_response)),
    }

def run_cause_trigger(X: pd.DataFrame, config: CauseTriggerConfig):
    if config.y_t not in X.columns:
        raise ValueError(f"Target variable {config.y_t!r} is not in X.columns")
    
    if X.isna().any().any():
        raise ValueError(
            "run_cause_trigger received NaNs. Drop or impute missing values before running."
        )

    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        non_numeric = X.columns[
            [not np.issubdtype(dtype, np.number) for dtype in X.dtypes]
        ].to_list()
        raise ValueError(
            f"run_cause_trigger expects all columns to be numeric. "
            f"Non-numeric columns: {non_numeric}"
        )

    if config.lags < 1:
        raise ValueError(f"config.lags must be >= 1, got {config.lags}")

    valid_v_weighting = {"backend", "refit"}
    if config.v_weighting not in valid_v_weighting:
        raise ValueError(
            f"Unknown v_weighting={config.v_weighting!r}. "
            "Use 'backend' or 'refit'."
        )

    if config.causal_backend == "hmml" and config.v_weighting != "backend":
        warnings.warn(
            "For the May-16 paper-compatible HMML baseline, use v_weighting='backend'. "
            "Other V-weighting modes are thesis sensitivity variants.",
            UserWarning,
        )

    if config.causal_backend in {"pcmci", "pcmci_plus"} and config.v_weighting == "backend":
        warnings.warn(
            "PCMCI/PCMCI+ with v_weighting='backend' uses conditional-dependence "
            "test statistics as beta-like weights. This is not May-16 paper-equivalent. "
            "Use v_weighting='refit' for the main PCMCI/PCMCI+ thesis extension.",
            UserWarning,
        )

    min_required_I2 = config.lags + 4
    if config.min_I2_length < min_required_I2:
        warnings.warn(
            f"config.min_I2_length={config.min_I2_length} is smaller than "
            f"the May-16 F-test minimum {min_required_I2} for lags={config.lags}. "
            "Candidates with too-short I2 will be skipped.",
            UserWarning,
        )

    result = {
        "C": [],
        "T": [],
        "pairs": [],
        "B_2": [],
        "T_candidates": [],
        "split_index": None,
        "split_timestamp": None,
        "I1_length": None,
        "I2_length": None,
        "target_abs_mean_I1": None,
        "target_abs_mean_I2": None,
        "target_abs_mean_difference": None,
        "backend": config.causal_backend,
        "configured_lags": config.lags,
        "configured_distribution": config.distribution,
        "parameter_source": getattr(config, "parameter_source", "manual"),
        "v_weighting": config.v_weighting,
        "refit_method": "ridge" if config.v_weighting == "refit" else None,
        "refit_metadata": {},
        "diagnostics": [],
        "causal_lags": {},
        "causal_scores": {},
        "contemporaneous_links": {},
        "full_interval_contemporaneous_links": {},
        "selected_cause_shift_scores": {},
        "autoregressive_parent_in_B2": False,
        "autoregressive_parent_full_interval": False,
        "split_end_index": None,
        "split_end_timestamp": None,
        "split_boundary_candidate": None,
        "target_mean_I1": None,
        "target_mean_I2": None,
    }

    backend = make_causal_backend(config)

    split_info = find_effect_split(
        X[config.y_t],
        min_I1_length=config.min_I1_length,
        min_I2_length=config.min_I2_length,
        return_info=True,
    )

    split_index = split_info["split_index"]
    split_end_index = split_info["split_end_index"]

    result["split_index"] = split_index
    result["split_end_index"] = split_end_index
    result["split_score"] = split_info["score"]
    result["split_boundary_candidate"] = split_info["boundary_split"]
    result["target_mean_I1"] = split_info["target_mean_I1"]
    result["target_mean_I2"] = split_info["target_mean_I2"]
    result["split_lag_buffer"] = split_info.get("lag_buffer")
    result["split_score_start_index"] = split_info.get("split_score_start_index")
    result["split_score_end_index"] = split_info.get("split_score_end_index")
    result["split_score_I2_length"] = split_info.get("score_I2_length")

    if split_index is not None:
        result["split_timestamp"] = X.index[split_index]

    if split_end_index is not None:
        result["split_end_timestamp"] = X.index[split_end_index - 1]

    if split_index is None:
        discovery = backend.discover(X, y_t=config.y_t)
        causes = [p for p in discovery.parents if p != config.y_t] # Exclude target variable from causes

        result["autoregressive_parent_full_interval"] = config.y_t in discovery.parents
        result["C"] = causes
        result["backend"] = config.causal_backend
        result["causal_lags"] = discovery.lags
        result["causal_scores"] = discovery.scores
        result["full_interval_contemporaneous_links"] = discovery.scores.get(
            "_contemporaneous_links", {}
        )
        result["stop_reason"] = "No valid I1/I2 split found."
        return result

    I_1 = X.iloc[:split_index]
    I_2 = X.iloc[split_index:]

    result["I1_length"] = len(I_1)
    result["I2_length"] = len(I_2)

    target_abs_mean_I1 = abs(I_1[config.y_t].mean())
    target_abs_mean_I2 = abs(I_2[config.y_t].mean())

    result["target_abs_mean_I1"] = target_abs_mean_I1
    result["target_abs_mean_I2"] = target_abs_mean_I2
    result["target_abs_mean_difference"] = target_abs_mean_I2 - target_abs_mean_I1
    #Find B1, B2 for y_t on I1 and I2.
    discovery_1 = backend.discover(I_1, y_t=config.y_t)
    discovery_2 = backend.discover(I_2, y_t=config.y_t)

    B_2 = list(discovery_2.parents)

    # Raw backend parents may include the target itself as an autoregressive parent.
    B_1_raw = list(discovery_1.parents)
    B_2_raw = list(discovery_2.parents)

    # Cause--Trigger candidate sets must exclude the target.
    # The target autoregressive parent is diagnostically useful, but it is not a
    # valid cause or trigger candidate in this thesis interpretation.
    B_1 = [v for v in B_1_raw if v != config.y_t]
    B_2 = [v for v in B_2_raw if v != config.y_t]

    result["autoregressive_parent_in_B2"] = config.y_t in B_2_raw
    result["backend"] = config.causal_backend
    result["causal_lags"] = discovery_2.lags
    result["causal_scores"] = discovery_2.scores
    result["contemporaneous_links"] = discovery_2.scores.get(
        "_contemporaneous_links", {}
    )

    # Store both raw and CT-valid parent sets.
    result["B_1_raw"] = B_1_raw
    result["B_2_raw"] = B_2_raw
    result["B_1"] = B_1
    result["B_2"] = B_2

    # ------------------------------------------------------------------
    # Trigger candidates
    # ------------------------------------------------------------------
    # Paper-compatible lagged trigger candidates:
    # x_s is in lagged non-target B2 and increases from I1 to I2.
    T_candidates_lagged = []
    for col in B_2:
        if abs(I_2[col].mean()) > abs(I_1[col].mean()):
            T_candidates_lagged.append(col)

    # Thesis extension:
    # PCMCI+ tau=0 links may be used as same-bin / immediate trigger candidates.
    # They are NOT added to B2, and they are NOT used in V.
    contemporaneous_links = result.get("contemporaneous_links", {}) or {}

    T_candidates_contemporaneous = []
    if (
        config.causal_backend == "pcmci_plus"
        and config.pcmci_plus_use_contemporaneous_triggers
    ):
        for col in contemporaneous_links.keys():
            if col == config.y_t:
                continue
            if col not in X.columns:
                continue

            # Keep the original Cause--Trigger increase condition.
            if abs(I_2[col].mean()) > abs(I_1[col].mean()):
                T_candidates_contemporaneous.append(col)

    # Preserve order and remove duplicates.
    T_candidates = list(dict.fromkeys(
        T_candidates_lagged + T_candidates_contemporaneous
    ))

    result["T_candidates_lagged"] = T_candidates_lagged
    result["T_candidates_contemporaneous"] = T_candidates_contemporaneous
    result["T_candidates"] = T_candidates

    # We need at least one lagged non-target B2 parent to construct V.
    # A contemporaneous trigger can be outside B2, but the candidate cause part
    # must still come from lagged B2.
    if len(B_2) < 1:
        result["refit_metadata"] = {
            "reason": "no lagged non-target B2 variables for V",
            "raw_B2": B_2_raw,
            "non_target_B2": B_2,
            "contemporaneous_trigger_candidates": T_candidates_contemporaneous,
        }
        result["stop_reason"] = (
            "No lagged non-target B2 variables available to construct V."
        )
        return result

    if len(T_candidates) == 0:
        result["refit_metadata"] = {
            "reason": "no trigger candidates",
            "raw_B2": B_2_raw,
            "non_target_B2": B_2,
            "contemporaneous_links": contemporaneous_links,
        }
        result["stop_reason"] = (
            "No lagged or contemporaneous trigger candidates satisfy the I1/I2 increase condition."
        )
        return result

    # At least one trigger must leave at least one B2 variable available as
    # candidate cause after removing the trigger.
    valid_trigger_exists = any(
        len([v for v in B_2 if v != x_s]) > 0
        for x_s in T_candidates
    )

    if not valid_trigger_exists:
        result["refit_metadata"] = {
            "reason": "trigger candidates leave no B2 variable for cause",
            "raw_B2": B_2_raw,
            "non_target_B2": B_2,
            "T_candidates": T_candidates,
        }
        result["stop_reason"] = (
            "No trigger candidate leaves a lagged B2 variable available as cause."
        )
        return result
    
    if config.v_weighting == "backend":
        beta_2 = discovery_2.beta
        result_refit_metadata = {
            "v_weighting": "backend",
            "source": config.causal_backend,
        }

    elif config.v_weighting == "refit":
        # refit only on lagged non-target B2 parents.
        # Contemporaneous PCMCI+ trigger candidates are not added to B2
        # and are not used to construct V.
        beta_star, refit_metadata = refit_beta_for_selected_parents(
            X=I_2,
            y_t=config.y_t,
            selected_parents=non_trigger_vars,
            lags=config.lags,
            alpha=config.refit_alpha,
            cv=config.refit_cv,
            cv_folds=config.refit_cv_folds,
        )

    else:
        raise ValueError(
            f"Unknown v_weighting={config.v_weighting!r}. "
            "Use 'backend' or 'refit'."
        )

    result["refit_metadata"] = result_refit_metadata

    for x_s in T_candidates:
        B_2_without_trigger = [
            v for v in B_2
            if v != x_s
        ]

        if x_s in T_candidates_lagged and x_s in T_candidates_contemporaneous:
            trigger_source = "lagged_and_contemporaneous"
        elif x_s in T_candidates_contemporaneous:
            trigger_source = "contemporaneous_pcmci_plus_tau0"
        else:
            trigger_source = "lagged_B2"

        if len(B_2_without_trigger) == 0:
            result["diagnostics"].append({
                "trigger": x_s,
                "cause": None,
                "accepted": False,
                "reason": "No B2 variables left after removing candidate trigger and target.",
            })
            continue

        # cause selection:
        # x_u := arg max_k |E(x_k)_I1 - E(x_k)_I2|
        # where x_k is selected from the B2-based variable set.
        mu_before = I_1[B_2_without_trigger].mean()
        mu_after = I_2[B_2_without_trigger].mean()
        delta_mu = (mu_after - mu_before).abs()
        x_u = delta_mu.idxmax()

        result["selected_cause_shift_scores"][x_s] = delta_mu.to_dict()

        n_I2 = len(I_2)
        d = config.lags
        regression_rows = n_I2 - d
        denominator_df = n_I2 - d - 3

        if regression_rows <= 0 or denominator_df <= 0:
            result["diagnostics"].append({
                "trigger": x_s,
                "cause": x_u,
                "accepted": False,
                "reason": "Too few observations for May-16 moderation F-test.",
                "I2_length": n_I2,
                "lags": d,
                "n_regression_rows": regression_rows,
                "dfd": denominator_df,
                "required_min_I2_length": d + 4,
            })
            continue

        X_without_trigger = I_2[B_2_without_trigger]
        beta_without_trigger = beta_2[B_2_without_trigger]

        V, X_without_trigger_lagged, beta_star = build_lagged_design_matrix_for_V(
            X_without_trigger.to_numpy(),
            beta_without_trigger.to_numpy(),
            lags=d,
            beta_is_ones=False,
        )

        x_s_values = I_2[x_s].iloc[d:].to_numpy().reshape(-1, 1)
        y_response = I_2[config.y_t].iloc[d:].to_numpy()

        if np.nanstd(V) == 0:
            result["diagnostics"].append({
                "trigger": x_s,
                "cause": x_u,
                "accepted": False,
                "reason": "V is constant after X_without-trigger @ beta_star.",
                "I2_length": n_I2,
                "lags": d,
                "n_regression_rows": regression_rows,
                "V_std": float(np.nanstd(V)),
            })
            continue

        moderation = test_moderation(
            y_response=y_response,
            V=V,
            x_s_values=x_s_values,
            n_I2=n_I2,
            d=d,
            alpha=config.alpha,
        )

        moderation["V_mean"] = float(np.nanmean(V))
        moderation["V_std"] = float(np.nanstd(V))
        moderation["V_min"] = float(np.nanmin(V))
        moderation["V_max"] = float(np.nanmax(V))
        moderation["nonzero_beta_count"] = int(np.count_nonzero(beta_star))
        moderation["X_without_trigger_lagged_shape"] = tuple(X_without_trigger_lagged.shape)

        moderation["trigger"] = x_s
        moderation["cause"] = x_u
        moderation["trigger_source"] = trigger_source
        moderation["trigger_is_contemporaneous_pcmci_plus"] = (
            x_s in T_candidates_contemporaneous
        )
        moderation["trigger_tau0_info"] = contemporaneous_links.get(x_s)
        result["diagnostics"].append(moderation)

        if moderation["accepted"]:
            if x_s not in result["T"]:
                result["T"].append(x_s)

            if x_u not in result["C"]:
                result["C"].append(x_u)

            pair = (x_u, x_s)
            if pair not in result["pairs"]:
                result["pairs"].append(pair)

    return result

def diagnostics_to_dataframe(result: dict) -> pd.DataFrame:
    """
    Convert result['diagnostics'] into a dataframe for inspection,
    sorting, plotting, and tables.
    """
    diagnostics = result.get("diagnostics", [])

    if len(diagnostics) == 0:
        return pd.DataFrame()

    return pd.DataFrame(diagnostics)