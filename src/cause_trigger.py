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
from sklearn.linear_model import LinearRegression
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

    beta_is_ones: bool = False
    # added:
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
        if not config.beta_is_ones:
            warnings.warn(
                "PCMCI/PCMCI+ backends do not return HMML-style regression beta coefficients. "
                "Their beta matrix stores causal-discovery test-statistic strengths, not β*. "
                "For paper-compatible May-16 runs, use causal_backend='hmml'. "
                "PCMCI/PCMCI+ with beta_is_ones=False is not paper-equivalent unless "
                "a separate regression refit is used to estimate β*.",
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

    if config.beta_is_ones:
        warnings.warn(
            "beta_is_ones=True is a non-paper variant for the May-16 version. "
            "The May-16 paper baseline uses V := X_without-trigger @ beta_star.",
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

    beta_2 = discovery_2.beta
    B_2 = list(discovery_2.parents)

    result["autoregressive_parent_in_B2"] = config.y_t in B_2
    result["backend"] = config.causal_backend
    result["causal_lags"] = discovery_2.lags
    result["causal_scores"] = discovery_2.scores
    result["contemporaneous_links"] = discovery_2.scores.get(
        "_contemporaneous_links", {}
    )
    result["B_1"] = list(discovery_1.parents)
    result["B_2"] = B_2

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
            beta_is_ones=config.beta_is_ones,
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