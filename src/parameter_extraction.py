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

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


_VALID_CRITERIA = {"aic", "bic", "hqic", "fpe"}


def select_var_lag(
    df: pd.DataFrame,
    max_lags: int = 12,
    criterion: str = "aic",
    fallback_lag: int = 1,
) -> int:
    """Select a positive VAR lag using an information criterion."""
    if criterion not in _VALID_CRITERIA:
        raise ValueError(
            f"criterion must be one of {sorted(_VALID_CRITERIA)}, "
            f"got {criterion!r}."
        )
    if not isinstance(max_lags, int) or max_lags < 1:
        raise ValueError("max_lags must be a positive integer.")
    if not isinstance(fallback_lag, int) or not 1 <= fallback_lag <= max_lags:
        raise ValueError("fallback_lag must be between 1 and max_lags.")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("VAR lag selection received missing or non-numeric values.")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("VAR lag selection received non-finite values.")
    if numeric.shape[1] < 2:
        raise ValueError("VAR lag selection requires at least two variables.")

    if len(numeric) <= max_lags + 2:
        warnings.warn(
            "Too few rows for the requested VAR maximum lag; "
            f"using fallback lag {fallback_lag}.",
            RuntimeWarning,
        )
        return fallback_lag

    try:
        selected = VAR(numeric).select_order(maxlags=max_lags)
        lag = getattr(selected, criterion)
    except Exception as exc:
        warnings.warn(
            f"VAR-{criterion.upper()} lag selection failed: {exc}. "
            f"Using fallback lag {fallback_lag}.",
            RuntimeWarning,
        )
        return fallback_lag

    if lag is None or not np.isfinite(lag) or int(lag) < 1:
        warnings.warn(
            f"VAR-{criterion.upper()} returned invalid lag {lag!r}; "
            f"using fallback lag {fallback_lag}.",
            RuntimeWarning,
        )
        return fallback_lag

    return int(lag)