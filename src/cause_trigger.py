"""
Cause–Trigger algorithm implementation for thesis experiments.

This algorithm is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Zenodo. DOI: 10.5281/zenodo.15109084

Original material: CC BY 4.0.
""" 

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from hmml_runner import HMMLRunner
from pcmci_runner import PCMCIBackend


def paper_abs_mean_increase(mean_after: float, mean_before: float) -> bool:
    """Return the condition |E[x] in I2| > |E[x] in I1|."""
    mean_after = float(mean_after)
    mean_before = float(mean_before)
    difference = abs(mean_after) - abs(mean_before)
    scale = max(1.0, abs(mean_after), abs(mean_before))
    tolerance = 100.0 * np.finfo(float).eps * scale
    return difference > tolerance


def validate_regular_time_index(
    df: pd.DataFrame,
    expected_step: str = "1h",
) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a pandas DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Time index is not sorted.")
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique()[:5]
        raise ValueError(
            f"Time index contains duplicate timestamps, e.g. {list(duplicates)}"
        )

    expected_delta = pd.Timedelta(expected_step)
    invalid = df.index.to_series().diff().dropna()
    invalid = invalid[invalid != expected_delta]
    if not invalid.empty:
        examples = {
            timestamp: str(delta)
            for timestamp, delta in invalid.head(5).items()
        }
        raise ValueError(
            "Time index is not a complete regular grid. "
            f"Expected step={expected_delta}; irregular transitions={examples}"
        )


def standard_scale_from_reference(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize analysis_df with mean/std estimated from reference_df."""
    reference = reference_df.apply(pd.to_numeric, errors="coerce")
    analysis = analysis_df.apply(pd.to_numeric, errors="coerce")

    if list(reference.columns) != list(analysis.columns):
        raise ValueError(
            "Reference and analysis dataframes must have identical columns "
            "in identical order."
        )
    if reference.empty:
        raise ValueError("Reference dataframe is empty.")

    for name, frame in (("Reference", reference), ("Analysis", analysis)):
        if frame.isna().any().any():
            missing = frame.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            raise ValueError(f"{name} dataframe contains NaNs:\n{missing}")
        if not np.isfinite(frame.to_numpy()).all():
            bad = [
                column
                for column in frame.columns
                if not np.isfinite(frame[column].to_numpy()).all()
            ]
            raise ValueError(
                f"{name} dataframe contains non-finite values in columns: {bad}"
            )

    reference_mean = reference.mean()
    reference_std = reference.std(ddof=0)
    invalid_scale = reference_std[
        (~np.isfinite(reference_std)) | (reference_std <= 0)
    ]
    if not invalid_scale.empty:
        raise ValueError(
            "Cannot reference-standardize constant or invalid columns:\n"
            f"{invalid_scale}"
        )

    scaled = (
        analysis
        .subtract(reference_mean, axis="columns")
        .divide(reference_std, axis="columns")
    )
    report = pd.DataFrame(
        {
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "reference_n": len(reference),
            "case_mean_after_scaling": scaled.mean(),
            "case_std_after_scaling": scaled.std(ddof=0),
        }
    )
    return scaled, report


@dataclass(frozen=True)
class CauseTriggerConfig:
    y_t: str
    lags: int
    distribution: str = "gaussian"
    alpha: float = 0.05
    min_I1_length: int = 12
    min_I2_length: int = 24
    causal_backend: str = "hmml"

    # Ridge refit used after PCMCI/PCMCI+ parent selection.
    refit_alpha: float = 1.0
    refit_cv: bool = True
    refit_cv_folds: int = 3

    # PCMCI / PCMCI+.
    pcmci_pc_alpha: float = 0.2
    pcmci_plus_pc_alpha: float = 0.01
    pcmci_alpha_level: float = 0.05
    pcmci_fdr_method: str | None = "fdr_bh"
    pcmci_cond_ind_test: str = "parcorr"
    pcmci_verbosity: int = 0

    # PCMCI+ contemporaneous-link settings.
    pcmci_contemp_collider_rule: str = "majority"
    pcmci_conflict_resolution: bool = True
    pcmci_plus_use_contemporaneous_triggers: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lags, int) or self.lags < 1:
            raise ValueError("lags must be a positive integer.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.min_I1_length < 1 or self.min_I2_length < 1:
            raise ValueError("Minimum interval lengths must be positive.")
        if self.causal_backend not in {"hmml", "pcmci", "pcmci_plus"}:
            raise ValueError(f"Unsupported backend: {self.causal_backend!r}")
        if self.refit_alpha < 0:
            raise ValueError("refit_alpha must be non-negative.")
        if self.refit_cv_folds < 2:
            raise ValueError("refit_cv_folds must be at least 2.")
        if self.pcmci_fdr_method not in {None, "fdr_bh"}:
            raise ValueError("pcmci_fdr_method must be None or 'fdr_bh'.")
        if self.pcmci_cond_ind_test not in {
            "parcorr",
            "robust_parcorr",
        }:
            raise ValueError(
                f"Unsupported conditional-independence test: "
                f"{self.pcmci_cond_ind_test!r}"
            )


@dataclass
class DiscoveryResult:
    parents: list[str]
    beta: pd.DataFrame | None
    lags: dict[str, list[int]]
    contemporaneous_links: dict[str, dict]


class HMMLBackend:
    def __init__(self, lags: int, distribution: str):
        self.runner = HMMLRunner(lags=lags, distribution=distribution)

    def discover(self, X: pd.DataFrame, y_t: str) -> DiscoveryResult:
        beta, adjacency = self.runner.get_betas_and_adjacency(X, y_t=y_t)
        parents = X.columns[np.asarray(adjacency).reshape(-1) == 1].to_list()
        return DiscoveryResult(
            parents=parents,
            beta=beta,
            lags={},
            contemporaneous_links={},
        )


def make_causal_backend(config: CauseTriggerConfig):
    if config.causal_backend == "hmml":
        return HMMLBackend(config.lags, config.distribution)
    
    pc_alpha = (
        config.pcmci_plus_pc_alpha
        if config.causal_backend == "pcmci_plus"
        else config.pcmci_pc_alpha
    )

    return PCMCIBackend(
        tau_max=config.lags,
        pc_alpha=pc_alpha,
        alpha_level=config.pcmci_alpha_level,
        fdr_method=config.pcmci_fdr_method,
        cond_ind_test=config.pcmci_cond_ind_test,
        verbosity=config.pcmci_verbosity,
        method=config.causal_backend,
        contemp_collider_rule=config.pcmci_contemp_collider_rule,
        conflict_resolution=config.pcmci_conflict_resolution,
    )


def _discover(backend, X: pd.DataFrame, y_t: str) -> DiscoveryResult:
    result = backend.discover(X, y_t=y_t)
    if isinstance(result, DiscoveryResult):
        return result

    return DiscoveryResult(
        parents=list(result.parents),
        beta=getattr(result, "beta", None),
        lags=dict(result.lags),
        contemporaneous_links=dict(
            getattr(result, "contemporaneous_links", {})
        ),
    )


def find_effect_split(
    y: pd.Series,
    min_I1_length: int = 12,
    min_I2_length: int = 30,
    *,
    return_info: bool = False,
):
    """
    Select the split maximizing |mean(I2)| - |mean(I1)|, subject to the
    increase condition and minimum interval lengths.
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    if y.isna().any():
        raise ValueError("find_effect_split received NaNs.")

    n = len(y)
    lower = int(min_I1_length)
    upper = n - int(min_I2_length)
    if lower > upper:
        raise ValueError(
            f"Not enough rows for min_I1_length={lower} and "
            f"min_I2_length={min_I2_length}: n={n}."
        )

    best = None
    best_score = float("-inf")

    for split_index in range(lower, upper + 1):
        mean_1 = float(y.iloc[:split_index].mean())
        mean_2 = float(y.iloc[split_index:].mean())
        if not paper_abs_mean_increase(mean_2, mean_1):
            continue

        score = abs(mean_2) - abs(mean_1)
        if score > best_score:
            boundary_distance = min(
                split_index - lower,
                upper - split_index,
            )
            best_score = score
            best = {
                "split_index": split_index,
                "score": float(score),
                "target_mean_I1": mean_1,
                "target_mean_I2": mean_2,
                "I1_length": split_index,
                "I2_length": n - split_index,
                "boundary_split": boundary_distance == 0,
            }

    if best is None:
        best = {
            "split_index": None,
            "score": None,
            "target_mean_I1": None,
            "target_mean_I2": None,
            "I1_length": None,
            "I2_length": None,
            "boundary_split": None,
        }

    return best if return_info else best["split_index"]


def build_lagged_design_matrix_for_V(
    X_values,
    beta_values,
    lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct X_without_trigger and V in lag-major column order."""
    X_values = np.asarray(X_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)

    if X_values.ndim != 2:
        raise ValueError("X_values must be a 2D array.")
    n, p = X_values.shape
    if n <= lags:
        raise ValueError(f"Need len(I2) > lags, got len={n}, lags={lags}.")
    if p == 0:
        raise ValueError("Cannot build V with zero non-trigger variables.")
    if beta_values.shape != (lags, p):
        raise ValueError(
            f"beta_values must have shape ({lags}, {p}), "
            f"got {beta_values.shape}."
        )

    lagged_blocks = [
        X_values[lags - lag:n - lag, :]
        for lag in range(1, lags + 1)
    ]
    lagged = np.hstack(lagged_blocks)
    beta_star = beta_values.reshape(-1, 1)
    return lagged @ beta_star, lagged, beta_star


def build_selected_lag_matrix(
    X: pd.DataFrame,
    selected_parents: list[str],
    selected_lags: dict[str, list[int]],
    max_lag: int,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Build a design matrix from the parent-lag terms selected by PCMCI."""
    blocks = []
    terms = []
    n = len(X)

    for parent in selected_parents:
        lags = sorted(
            {
                int(lag)
                for lag in selected_lags.get(parent, [])
                if 1 <= int(lag) <= max_lag
            }
        )
        values = X[parent].to_numpy(dtype=float)
        for lag in lags:
            blocks.append(values[max_lag - lag:n - lag])
            terms.append((parent, lag))

    if not blocks:
        raise ValueError("No selected parent-lag terms are available for refitting.")
    return np.column_stack(blocks), terms


def refit_beta_for_selected_parents(
    X: pd.DataFrame,
    y_t: str,
    selected_parents: list[str],
    selected_lags: dict[str, list[int]],
    lags: int,
    *,
    alpha: float = 1.0,
    cv: bool = True,
    cv_folds: int = 3,
) -> pd.DataFrame:
    """Ridge-refit coefficients for PCMCI/PCMCI+ selected parent-lag terms."""
    selected_parents = list(selected_parents)
    index = [f"Lag_{lag}" for lag in range(1, lags + 1)]
    if not selected_parents:
        return pd.DataFrame(index=index)

    X_lagged, terms = build_selected_lag_matrix(
        X,
        selected_parents,
        selected_lags,
        lags,
    )
    y = X[y_t].iloc[lags:].to_numpy(dtype=float)
    if len(y) != len(X_lagged):
        raise ValueError("Refit response/design length mismatch.")

    n_splits = min(int(cv_folds), 3)
    use_cv = bool(
        cv
        and n_splits >= 2
        and len(X_lagged) > 2 * (lags + 1)
    )

    if use_cv:
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=lags)
        try:
            model = RidgeCV(
                alphas=np.logspace(-4, 4, 40),
                cv=splitter,
                scoring="neg_mean_squared_error",
                fit_intercept=True,
            ).fit(X_lagged, y)
        except ValueError:
            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
            ).fit(X_lagged, y)
    else:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
        ).fit(X_lagged, y)

    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    beta = np.zeros((lags, len(selected_parents)), dtype=float)
    parent_index = {
        parent: position
        for position, parent in enumerate(selected_parents)
    }
    for coefficient, (parent, lag) in zip(coefficients, terms):
        beta[lag - 1, parent_index[parent]] = coefficient

    return pd.DataFrame(beta, index=index, columns=selected_parents)


def f_statistic(rss_reduced: float, rss_full: float, n: int, d: int) -> float:
    """Paper statistic: (RSS1 - RSS2) * (n - d - 3) / RSS2."""
    denominator_df = n - d - 3
    if denominator_df <= 0:
        raise ValueError(
            f"Need len(I2) > lags + 3; got n={n}, d={d}."
        )
    if rss_full <= 0:
        raise ValueError("Full-model RSS must be positive.")
    return ((rss_reduced - rss_full) * denominator_df) / rss_full


def test_moderation(
    y_response,
    V,
    x_s_values,
    n_I2: int,
    d: int,
    alpha: float = 0.05,
) -> dict:
    """Fit the paper's reduced/full moderation regressions and F-test."""
    y = np.asarray(y_response, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    x_s = np.asarray(x_s_values, dtype=float)

    if V.ndim != 2 or V.shape[1] != 1:
        raise ValueError(f"V must have shape (n-d, 1); got {V.shape}.")
    if x_s.ndim != 2 or x_s.shape[1] != 1:
        raise ValueError(f"x_s_values must have shape (n-d, 1); got {x_s.shape}.")
    if not (len(y) == len(V) == len(x_s)):
        raise ValueError("Response and feature lengths do not match.")
    if not (
        np.isfinite(y).all()
        and np.isfinite(V).all()
        and np.isfinite(x_s).all()
    ):
        raise ValueError("Moderation inputs contain non-finite values.")
    if np.isclose(np.std(V), 0.0):
        raise ValueError("V is constant; moderation is not identifiable.")

    interaction = V * x_s
    reduced = LinearRegression().fit(V, y)
    full_features = np.hstack([V, interaction])
    full = LinearRegression().fit(full_features, y)

    rss_reduced = float(np.sum((y - reduced.predict(V)) ** 2))
    rss_full = float(np.sum((y - full.predict(full_features)) ** 2))
    statistic = f_statistic(rss_reduced, rss_full, n_I2, d)
    denominator_df = n_I2 - d - 3
    critical = float(f.ppf(1 - alpha, 1, denominator_df))
    p_value = float(1 - f.cdf(statistic, 1, denominator_df))

    return {
        "accepted": bool(statistic > critical),
        "f_stat": float(statistic),
        "critical_f": critical,
        "p_value": p_value,
        "rss_reduced": rss_reduced,
        "rss_full": rss_full,
        "rss_reduction_ratio": (
            (rss_reduced - rss_full) / rss_reduced
            if rss_reduced > 0
            else np.nan
        ),
        "gamma_2": float(full.coef_[1]),
    }


def _validate_analysis_frames(
    X_model: pd.DataFrame,
    X_mean: pd.DataFrame,
    target: str,
) -> None:
    if not X_model.index.equals(X_mean.index):
        raise ValueError("X_model and X_mean must have identical time indices.")
    if list(X_model.columns) != list(X_mean.columns):
        raise ValueError(
            "X_model and X_mean must have identical columns in identical order."
        )
    if X_model.shape != X_mean.shape:
        raise ValueError("X_model and X_mean must have identical shapes.")
    if target not in X_model.columns:
        raise ValueError(f"Target variable {target!r} is not in the data.")

    for name, frame in (("X_model", X_model), ("X_mean", X_mean)):
        if frame.isna().any().any():
            raise ValueError(f"{name} contains NaNs.")
        if not all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes):
            raise ValueError(f"{name} must contain only numeric columns.")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{name} contains non-finite values.")


def run_cause_trigger(
    X_model: pd.DataFrame,
    config: CauseTriggerConfig,
    *,
    X_mean: pd.DataFrame,
) -> dict:
    """
    Run the Cause–Trigger algorithm.

    X_model is used for causal discovery, coefficient estimation, V, and
    moderation. X_mean is used only for the split, trigger mean screen, and
    final cause mean-shift ranking.
    """
    _validate_analysis_frames(X_model, X_mean, config.y_t)

    result = {
        "C": [],
        "T": [],
        "pairs": [],
        "B_1": [],
        "B_2": [],
        "T_candidates": [],
        "T_candidates_lagged": [],
        "T_candidates_contemporaneous": [],
        "split_index": None,
        "split_timestamp": None,
        "I1_length": None,
        "I2_length": None,
        "split_score": None,
        "boundary_split": None,
        "target_abs_mean_I1": None,
        "target_abs_mean_I2": None,
        "backend": config.causal_backend,
        "lag": config.lags,
        "distribution": config.distribution,
        "contemporaneous_links": {},
        "autoregressive_parent_in_B2": False,
        "diagnostics": [],
        "stop_reason": None,
    }

    backend = make_causal_backend(config)
    split = find_effect_split(
        X_mean[config.y_t],
        min_I1_length=config.min_I1_length,
        min_I2_length=config.min_I2_length,
        return_info=True,
    )
    split_index = split["split_index"]
    result["split_index"] = split_index
    result["split_score"] = split["score"]
    result["boundary_split"] = split["boundary_split"]

    if split_index is None:
        discovery = _discover(backend, X_model, config.y_t)
        result["C"] = [
            parent
            for parent in discovery.parents
            if parent != config.y_t
        ]
        result["stop_reason"] = "No valid I1/I2 split found."
        return result

    result["split_timestamp"] = X_model.index[split_index]

    I_1 = X_model.iloc[:split_index]
    I_2 = X_model.iloc[split_index:]
    M_1 = X_mean.iloc[:split_index]
    M_2 = X_mean.iloc[split_index:]

    result["I1_length"] = len(I_1)
    result["I2_length"] = len(I_2)
    result["target_abs_mean_I1"] = float(abs(M_1[config.y_t].mean()))
    result["target_abs_mean_I2"] = float(abs(M_2[config.y_t].mean()))

    discovery_1 = _discover(backend, I_1, config.y_t)
    discovery_2 = _discover(backend, I_2, config.y_t)

    B_1_model = list(discovery_1.parents)
    B_2_model = list(discovery_2.parents)
    B_1 = [variable for variable in B_1_model if variable != config.y_t]
    B_2 = [variable for variable in B_2_model if variable != config.y_t]

    result["B_1"] = B_1
    result["B_2"] = B_2
    result["autoregressive_parent_in_B2"] = config.y_t in B_2_model
    result["contemporaneous_links"] = discovery_2.contemporaneous_links

    lagged_candidates = [
        variable
        for variable in B_2
        if paper_abs_mean_increase(
            M_2[variable].mean(),
            M_1[variable].mean(),
        )
    ]

    contemporaneous_candidates = []
    if (
        config.causal_backend == "pcmci_plus"
        and config.pcmci_plus_use_contemporaneous_triggers
    ):
        for variable, link in discovery_2.contemporaneous_links.items():
            if (
                variable != config.y_t
                and variable in X_model.columns
                and link.get("eligible_as_trigger_candidate", False)
                and paper_abs_mean_increase(
                    M_2[variable].mean(),
                    M_1[variable].mean(),
                )
            ):
                contemporaneous_candidates.append(variable)

    candidates = list(
        dict.fromkeys(lagged_candidates + contemporaneous_candidates)
    )
    result["T_candidates_lagged"] = lagged_candidates
    result["T_candidates_contemporaneous"] = contemporaneous_candidates
    result["T_candidates"] = candidates

    if not B_2:
        result["stop_reason"] = (
            "No lagged non-target B2 variables are available to construct V."
        )
        return result
    if not candidates:
        result["stop_reason"] = "No trigger candidates satisfy the mean condition."
        return result

    if config.causal_backend == "hmml":
        if discovery_2.beta is None:
            raise RuntimeError("HMML did not return beta coefficients.")
        beta_2 = discovery_2.beta
    else:
        selected_lags = {
            parent: discovery_2.lags.get(parent, [])
            for parent in B_2_model
        }
        beta_2 = refit_beta_for_selected_parents(
            I_2,
            config.y_t,
            B_2_model,
            selected_lags,
            config.lags,
            alpha=config.refit_alpha,
            cv=config.refit_cv,
            cv_folds=config.refit_cv_folds,
        )

    for trigger in candidates:
        cause_candidates = [
            variable
            for variable in B_2
            if variable != trigger
        ]
        if not cause_candidates:
            result["diagnostics"].append(
                {
                    "trigger": trigger,
                    "cause": None,
                    "trigger_source": (
                        "contemporaneous"
                        if trigger in contemporaneous_candidates
                        else "lagged"
                    ),
                    "accepted": False,
                    "reason": "No B2 cause candidate remains after removing trigger.",
                }
            )
            continue

        shifts = (
            M_2[cause_candidates].mean()
            - M_1[cause_candidates].mean()
        ).abs()
        cause = shifts.idxmax()

        model_variables = [
            variable
            for variable in B_2_model
            if variable != trigger
        ]
        X_without_trigger = I_2[model_variables]
        beta_without_trigger = beta_2[model_variables]

        V, _, _ = build_lagged_design_matrix_for_V(
            X_without_trigger.to_numpy(),
            beta_without_trigger.to_numpy(),
            config.lags,
        )
        if np.isclose(np.std(V), 0.0):
            result["diagnostics"].append(
                {
                    "trigger": trigger,
                    "cause": cause,
                    "trigger_source": (
                        "contemporaneous"
                        if trigger in contemporaneous_candidates
                        else "lagged"
                    ),
                    "accepted": False,
                    "reason": "V is constant.",
                }
            )
            continue

        moderation = test_moderation(
            y_response=I_2[config.y_t].iloc[config.lags:].to_numpy(),
            V=V,
            x_s_values=(
                I_2[trigger]
                .iloc[config.lags:]
                .to_numpy()
                .reshape(-1, 1)
            ),
            n_I2=len(I_2),
            d=config.lags,
            alpha=config.alpha,
        )
        moderation.update(
            {
                "trigger": trigger,
                "cause": cause,
                "trigger_source": (
                    "lagged_and_contemporaneous"
                    if (
                        trigger in lagged_candidates
                        and trigger in contemporaneous_candidates
                    )
                    else (
                        "contemporaneous"
                        if trigger in contemporaneous_candidates
                        else "lagged"
                    )
                ),
                "reason": None,
            }
        )
        result["diagnostics"].append(moderation)

        if moderation["accepted"]:
            if trigger not in result["T"]:
                result["T"].append(trigger)
            if cause not in result["C"]:
                result["C"].append(cause)
            pair = (cause, trigger)
            if pair not in result["pairs"]:
                result["pairs"].append(pair)

    return result


def diagnostics_to_dataframe(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result.get("diagnostics", []))
