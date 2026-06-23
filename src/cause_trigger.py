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
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
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

    # How to construct beta_star for V = X_without-trigger @ beta_star
    # "backend" = use beta from backend; baseline for HMML
    # "refit"   = refit beta after parent selection; for PCMCI/PCMCI+
    v_weighting: str = "backend"

    # Used only when v_weighting="refit".
    # "ridge"          = stable coefficient refit under collinearity
    # "adaptive_lasso" = sparse refit following Zou-style adaptive penalties
    refit_method: str = "ridge"
    refit_alpha: float = 1.0
    refit_cv: bool = True
    refit_cv_folds: int = 5
    refit_lasso_max_iter: int = 50000
    adaptive_gamma: float = 1.0
    adaptive_epsilon: float = 1e-6

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

def find_increase_split(
    y: pd.Series,
    min_I1_length: int = 12,
    min_I2_length: int = 30,
):
    # find I1, I2 such that |E(y_t)|_I2 > |E(y_t)|_I1
    best_split = None
    max_abs_mean_difference = float("-inf")

    for i in range(1, len(y) - 1):
        interval_1 = y.iloc[:i]
        interval_2 = y.iloc[i:]

        if len(interval_1) < min_I1_length:
            continue
        if len(interval_2) < min_I2_length:
            continue

        #|E(y_t)|_I2 > |E(y_t)|_I1
        abs_mean_difference = abs(interval_2.mean()) - abs(interval_1.mean())

        if abs_mean_difference > max_abs_mean_difference and abs_mean_difference > 0:
            max_abs_mean_difference = abs_mean_difference
            best_split = i

    return best_split


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
    method: str = "ridge",
    alpha: float = 1.0,
    cv: bool = True,
    cv_folds: int = 5,
    lasso_max_iter: int = 50000,
    adaptive_gamma: float = 1.0,
    adaptive_epsilon: float = 1e-6,
):
    """
    Estimate beta_star for selected lagged parents after causal discovery.

    This is intended for PCMCI/PCMCI+ thesis extensions:
        PCMCI/PCMCI+ selects B2.
        This function estimates beta_star by regression on selected lagged parents.

    Supported methods
    -----------------
    ridge:
        RidgeCV/Ridge refit. Main stable refit method.

    adaptive_lasso:
        Zou-style adaptive lasso refit. Uses an initial RidgeCV/Ridge estimate
        to construct adaptive penalty weights, then chooses the lasso penalty
        by TimeSeriesSplit cross-validation when cv=True.

    Returns
    -------
    beta_df : pd.DataFrame
        Shape (lags, len(selected_parents)).
        Rows: Lag_1 ... Lag_d.
        Columns: selected_parents.
    metadata : dict
        Information about method, selected alpha, nonzero count, etc.
    """
    selected_parents = list(selected_parents)
    method = method.lower()

    allowed_methods = {"ridge", "adaptive_lasso"}
    if method not in allowed_methods:
        raise ValueError(
            f"Unknown refit method {method!r}. "
            "Use only 'ridge' or 'adaptive_lasso'."
        )

    index = [f"Lag_{lag}" for lag in range(1, lags + 1)]

    if len(selected_parents) == 0:
        beta_df = pd.DataFrame(
            np.zeros((lags, 0)),
            index=index,
            columns=[],
        )
        return beta_df, {
            "refit_method": method,
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

    metadata = {
        "refit_method": method,
        "refit_alpha_requested": alpha,
        "refit_cv": use_cv,
        "refit_cv_folds": int(n_splits) if use_cv else 0,
        "n_refit_rows": int(X_lagged.shape[0]),
        "n_refit_features": int(X_lagged.shape[1]),
        "selected_parents": selected_parents,
    }

    ridge_alphas = np.logspace(-4, 4, 40)

    if method == "ridge":
        if use_cv:
            model = RidgeCV(alphas=ridge_alphas, cv=cv_obj)
        else:
            model = Ridge(alpha=alpha)

        model.fit(X_lagged, y_values)
        beta_flat = np.asarray(model.coef_, dtype=float).reshape(-1)

        metadata["refit_alpha_used"] = (
            float(model.alpha_) if hasattr(model, "alpha_") else float(alpha)
        )

    elif method == "adaptive_lasso":
        # Stage 1: stable initial estimator for adaptive weights.
        if use_cv:
            ridge_init = RidgeCV(alphas=ridge_alphas, cv=cv_obj)
        else:
            ridge_init = Ridge(alpha=alpha)

        ridge_init.fit(X_lagged, y_values)
        beta_initial = np.asarray(ridge_init.coef_, dtype=float).reshape(-1)

        # Zou-style adaptive weights:
        # weaker initial coefficients receive stronger penalty.
        adaptive_weights = 1.0 / (
            (np.abs(beta_initial) + adaptive_epsilon) ** adaptive_gamma
        )

        # Weighted lasso via column scaling:
        # min ||y - X beta||^2 + lambda * sum_j w_j |beta_j|
        # Let theta_j = w_j * beta_j and X_weighted_j = X_j / w_j.
        X_weighted = X_lagged / adaptive_weights.reshape(1, -1)

        if use_cv:
            lasso_alphas = np.logspace(-5, 1, 60)
            lasso_model = LassoCV(
                alphas=lasso_alphas,
                cv=cv_obj,
                max_iter=lasso_max_iter,
                fit_intercept=True,
            )
        else:
            lasso_model = Lasso(
                alpha=alpha,
                max_iter=lasso_max_iter,
                fit_intercept=True,
            )

        lasso_model.fit(X_weighted, y_values)

        theta = np.asarray(lasso_model.coef_, dtype=float).reshape(-1)
        beta_flat = theta / adaptive_weights

        metadata["initial_ridge_alpha_used"] = (
            float(ridge_init.alpha_) if hasattr(ridge_init, "alpha_") else float(alpha)
        )
        metadata["refit_alpha_used"] = (
            float(lasso_model.alpha_) if hasattr(lasso_model, "alpha_") else float(alpha)
        )
        metadata["adaptive_gamma"] = float(adaptive_gamma)
        metadata["adaptive_epsilon"] = float(adaptive_epsilon)
        metadata["adaptive_weight_min"] = float(np.min(adaptive_weights))
        metadata["adaptive_weight_max"] = float(np.max(adaptive_weights))

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

    metadata["nonzero_beta_count"] = int(np.count_nonzero(beta_flat))
    metadata["beta_abs_max"] = float(np.max(np.abs(beta_flat))) if beta_flat.size else 0.0

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

    valid_refit_methods = {"ridge", "adaptive_lasso"}
    if config.v_weighting == "refit" and config.refit_method not in valid_refit_methods:
        raise ValueError(
            f"Unknown refit_method={config.refit_method!r}. "
            "Use 'ridge' or 'adaptive_lasso'."
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
        "refit_method": config.refit_method if config.v_weighting == "refit" else None,
        "refit_metadata": {},
        "diagnostics": [],
        "causal_lags": {},
        "causal_scores": {},
        "contemporaneous_links": {},
        "full_interval_contemporaneous_links": {},
        "selected_cause_shift_scores": {},
        "autoregressive_parent_in_B2": False,
        "autoregressive_parent_full_interval": False,
    }

    backend = make_causal_backend(config)

    split_index = find_increase_split(
        X[config.y_t],
        min_I1_length=config.min_I1_length,
        min_I2_length=config.min_I2_length,
    )

    result["split_index"] = split_index

    if split_index is not None:
        result["split_timestamp"] = X.index[split_index]

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
        result["stop_reason"] = "No valid I1/I2 split with |mean(y)_I2| > |mean(y)_I1|."
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

    if config.v_weighting == "backend":
        # HMML paper baseline: use backend-provided beta_star.
        # For PCMCI/PCMCI+, this is only a diagnostic/sensitivity option,
        # because their values are not structural regression coefficients.
        beta_2 = discovery_2.beta
        result_refit_metadata = {
            "v_weighting": "backend",
            "source": config.causal_backend,
        }

    elif config.v_weighting == "refit":
        # PCMCI/PCMCI+ thesis extension:
        # use backend for B2, then estimate beta_star by regression on selected parents.
        beta_2, result_refit_metadata = refit_beta_for_selected_parents(
            X=I_2,
            y_t=config.y_t,
            selected_parents=B_2,
            lags=config.lags,
            method=config.refit_method,
            alpha=config.refit_alpha,
            cv=config.refit_cv,
            cv_folds=config.refit_cv_folds,
            lasso_max_iter=config.refit_lasso_max_iter,
            adaptive_gamma=config.adaptive_gamma,
            adaptive_epsilon=config.adaptive_epsilon,
        )

    else:
        raise ValueError(
            f"Unknown v_weighting={config.v_weighting!r}. "
            "Use 'backend', or 'refit'."
        )

    result["autoregressive_parent_in_B2"] = config.y_t in B_2
    result["backend"] = config.causal_backend
    result["causal_lags"] = discovery_2.lags
    result["causal_scores"] = discovery_2.scores
    result["contemporaneous_links"] = discovery_2.scores.get(
        "_contemporaneous_links", {}
    )
    result["B_1"] = list(discovery_1.parents)
    result["B_2"] = B_2
    result["refit_metadata"] = result_refit_metadata

    # Trigger candidates
    T_candidates = []
    # x_s ∈ B2, x_s ≠ y_t, and |E(x_s)|_I2 > |E(x_s)|_I1
    for col in B_2:
        if col == config.y_t: # Exclude target variable from trigger candidates
            continue
        if abs(I_2[col].mean()) > abs(I_1[col].mean()): # Alg. line 10: |E(x_s^t)|_I2 > |E(x_s^t)|_I1
            T_candidates.append(col)

    result["T_candidates"] = T_candidates

    if len(B_2) < 2:
        result["stop_reason"] = "Not enough B2 variables for cause selection."
        return result
        

    for x_s in T_candidates:
        B_2_without_trigger = [
            v for v in B_2
            if v != x_s and v != config.y_t
        ]

        if len(B_2_without_trigger) == 0:
            result["diagnostics"].append({
                "trigger": x_s,
                "cause": None,
                "accepted": False,
                "reason": "No B2 variables left after removing candidate trigger and target.",
            })
            continue

        # cause selection: (line 14 in Alg.)
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