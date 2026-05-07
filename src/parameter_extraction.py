"""
Parameter extraction for Cause–Trigger thesis experiments.

Selects:
1. VAR lag order using an information criterion.
2. Target distribution for HMML using distfit.

This file is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Zenodo. DOI: 10.5281/zenodo.15109084
"""

import numpy as np
import pandas as pd
from distfit import distfit
from statsmodels.tsa.api import VAR


def select_var_lag(df, max_lags=3, criterion="aic", fallback_lag=1):
    """
    Select the optimal lag for a multivariate time series using VAR.

    Returns fallback_lag if VAR selection fails or returns an invalid lag.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if len(numeric_df) <= max_lags + 2:
        return fallback_lag

    try:
        model = VAR(numeric_df)
        selected = model.select_order(maxlags=max_lags)
        lag = getattr(selected, criterion)

        if lag is None or np.isnan(lag) or lag < 1:
            return fallback_lag

        return int(lag)

    except Exception:
        return fallback_lag


def convert_name(name: str):
    convert = {
        "gamma": "gamma",
        "norm": "gaussian",
        "invgauss": "inverse_gaussian",
    }

    if name not in convert:
        return "gaussian"

    return convert[name]


def find_distribution(series, fallback_distribution="gaussian"):
    """
    Fit distribution for the target series and convert to HMML-compatible name.
    """
    clean_series = pd.Series(series).dropna()

    if len(clean_series) < 10:
        return fallback_distribution

    try:
        dist = distfit(
            distr=["gamma", "invgauss", "norm"],
            random_state=0,
            verbose=False,
        )
        dv = dist.fit_transform(clean_series.values)
        best_distribution = dv["model"]["name"]
        return convert_name(best_distribution)

    except Exception:
        return fallback_distribution


def find_parameters(
    X,
    target_series,
    max_lags=3,
    criterion="aic",
    fallback_lag=1,
    fallback_distribution="gaussian",
):
    """
    Return distribution and lag for CauseTriggerConfig.
    """
    lag = select_var_lag(
        X,
        max_lags=max_lags,
        criterion=criterion,
        fallback_lag=fallback_lag,
    )

    distribution = find_distribution(
        target_series,
        fallback_distribution=fallback_distribution,
    )

    return distribution, lag