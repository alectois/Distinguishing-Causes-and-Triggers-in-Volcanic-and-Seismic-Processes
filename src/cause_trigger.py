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
    min_interval_length: int = 30
    beta_is_ones: bool = True
    exclude_target_from_triggers: bool = False

    # Step-8 causal discovery backend
    causal_backend: str = "hmml"  # "hmml" or "pcmci"

    # PCMCI settings
    pcmci_pc_alpha: float = 0.05
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: str = "fdr_bh"
    pcmci_cond_ind_test: str = "parcorr"
    pcmci_verbosity: int = 0

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
        parents = X.columns[np.where(adjacency == 1)[0]].to_list()

        return BackendResult(
            parents=parents,
            adjacency=adjacency,
            beta=beta,
            lags={},
            scores={},
            raw_results=None,
        )


def make_causal_backend(config: CauseTriggerConfig):
    if config.causal_backend == "hmml":
        return HMMLBackend(
            lags=config.lags,
            distribution=config.distribution,
        )

    if config.causal_backend == "pcmci":
        return PCMCIBackend(
            tau_max=config.lags,
            pc_alpha=config.pcmci_pc_alpha,
            alpha_level=config.pcmci_alpha_level,
            fdr_method=config.pcmci_fdr_method,
            cond_ind_test=config.pcmci_cond_ind_test,
            verbosity=config.pcmci_verbosity,
        )

    raise ValueError(
        f"Unknown causal_backend={config.causal_backend!r}. "
        "Use 'hmml' or 'pcmci'."
    )

def find_increase_split(y: pd.Series, min_interval_length: int = 30):
    # find I1, I2 such that |E(y_t)|_I2 > |E(y_t)|_I1
    best_split = None
    max_abs_mean_difference = float("-inf")

    for i in range(1, len(y) - 1):
        interval_1 = y.iloc[:i]
        interval_2 = y.iloc[i:]

        if len(interval_2) <= min_interval_length:
            continue

        abs_mean_difference = abs(interval_2.mean()) - abs(interval_1.mean())

        if abs_mean_difference > max_abs_mean_difference and abs_mean_difference > 0:
            max_abs_mean_difference = abs_mean_difference
            best_split = i

    return best_split


def build_lagged_design_matrix(X_values, beta_values, lags, beta_is_ones=True):
    design_cols = []

    for i in range(X_values.shape[1]):
        for j in range(lags, 0, -1):
            col = X_values[j - 1 : X_values.shape[0] - lags + j - 1, i]
            design_cols.append(col.reshape(-1, 1))

    design_matrix = np.hstack(design_cols)

    beta_col = beta_values.reshape(-1, 1)
    if beta_is_ones:
        beta_col = np.ones_like(beta_col)

    return design_matrix, beta_col


def residual_sum_of_squares(y_true, X_features, model):
    y_hat = model.predict(X_features)
    return np.sum((y_true - y_hat) ** 2)


def f_statistic(rss_reduced, rss_full, r):
    """
    Paper Eq. (10), pages 19-20:

        S = ((RSS1 - RSS2) * (r - 2)) / RSS2

    Interpreted consistently as:
        RSS1 = RSS_reduced from Eq. (4)
        RSS2 = RSS_full from Eq. (3)
    """
    return ((rss_reduced - rss_full) * (r - 2)) / rss_full


def test_moderation(y_response, V, x_s_values, alpha=0.05):
    reduced_model = LinearRegression().fit(V, y_response)

    features_full = np.hstack([V, V * x_s_values])
    full_model = LinearRegression().fit(features_full, y_response)

    rss_reduced = residual_sum_of_squares(y_response, V, reduced_model)
    rss_full = residual_sum_of_squares(y_response, features_full, full_model)

    r = V.shape[0]
    f_stat = f_statistic(rss_reduced, rss_full, r)
    critical_f = f.ppf(1 - alpha, dfn=1, dfd=(r - 2))

    rss_reduction = rss_reduced - rss_full

    if rss_reduced != 0:
        rss_reduction_ratio = rss_reduction / rss_reduced
    else:
        rss_reduction_ratio = np.nan

    # Eq. (3): y_t = gamma_0 + gamma_1 V_t + gamma_2 V_t x_s^t + epsilon
    # sklearn full_model.coef_[0] corresponds to gamma_1,
    # full_model.coef_[1] corresponds to gamma_2.
    gamma_1 = full_model.coef_[0]
    gamma_2 = full_model.coef_[1]

    return {
        "accepted": f_stat > critical_f,
        "f_stat": f_stat,
        "critical_f": critical_f,
        "rss_reduced": rss_reduced,
        "rss_full": rss_full,
        "rss_reduction": rss_reduction,
        "rss_reduction_ratio": rss_reduction_ratio,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "r": r,
    }


def run_cause_trigger(X: pd.DataFrame, config: CauseTriggerConfig):
    if config.y_t not in X.columns:
        raise ValueError(f"Target variable {config.y_t!r} is not in X.columns")

    result = {
        "C": [],
        "T": [],
        "pairs": [],
        "B_2": [],
        "T_candidates": [],
        "split_index": None,
        "diagnostics": [],
    }

    backend = make_causal_backend(config)

    split_index = find_increase_split(
        X[config.y_t],
        min_interval_length=config.min_interval_length,
    )

    result["split_index"] = split_index

    if split_index is None:
        discovery = backend.discover(X, y_t=config.y_t)
        causes = list(discovery.parents)

        if config.exclude_target_from_triggers:
            causes = [c for c in causes if c != config.y_t]

        result["C"] = causes
        result["backend"] = config.causal_backend
        result["causal_lags"] = discovery.lags
        result["causal_scores"] = discovery.scores
        return result

    I_1 = X.iloc[:split_index]
    I_2 = X.iloc[split_index:]

    # cause selection:
    # x_u := arg max_k |E(x_k)_I1 - E(x_k)_I2|
    mu_before = I_1.mean()
    mu_after = I_2.mean()
    delta_mu = (mu_after - mu_before).abs()
    x_u = delta_mu.idxmax()

    discovery_2 = backend.discover(I_2, y_t=config.y_t)

    beta_2 = discovery_2.beta
    B_2 = list(discovery_2.parents)

    result["backend"] = config.causal_backend
    result["causal_lags"] = discovery_2.lags
    result["causal_scores"] = discovery_2.scores

    if config.exclude_target_from_triggers:
        B_2 = [v for v in B_2 if v != config.y_t]

    result["B_2"] = B_2

    T_candidates = []
    for col in B_2:
        if abs(I_2[col].mean()) > abs(I_1[col].mean()): # Alg. line 10: |E(x_s^t)|_I2 > |E(x_s^t)|_I1
            T_candidates.append(col)

    result["T_candidates"] = T_candidates

    if len(B_2) < 2:
        return result

    for x_s in T_candidates:
        B_2_without_trigger = [v for v in B_2 if v != x_s]

        X_without_trigger = I_2[B_2_without_trigger]
        beta_without_trigger = beta_2[B_2_without_trigger]

        design_matrix, beta_col = build_lagged_design_matrix(
            X_without_trigger.to_numpy(),
            beta_without_trigger.to_numpy(),
            config.lags,
            beta_is_ones=config.beta_is_ones,
        )

        V = design_matrix @ beta_col

        y_response = I_2[config.y_t].iloc[config.lags:].to_numpy()
        x_s_values = I_2[x_s].iloc[config.lags:].to_numpy().reshape(-1, 1)

        moderation = test_moderation(
            y_response=y_response,
            V=V,
            x_s_values=x_s_values,
            alpha=config.alpha,
        )

        moderation["trigger"] = x_s
        moderation["cause"] = x_u
        result["diagnostics"].append(moderation)

        if moderation["accepted"]:
            result["T"].append(x_s)
            result["C"].append(x_u)
            result["pairs"].append((x_u, x_s))

    return result