"""
This file is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Original source: https://doi.org/10.5281/zenodo.15109084
Original license: CC BY 4.0
This implementation has been substantially modified and extended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HMMLRunner:
    """Run HMML and return lag coefficients plus adjacency."""

    def __init__(self, lags: int, distribution: str = "gaussian"):
        if not isinstance(lags, int) or lags < 1:
            raise ValueError(f"HMMLRunner requires lags >= 1, got {lags!r}.")
        if not isinstance(distribution, str) or not distribution:
            raise ValueError("distribution must be a non-empty string.")

        try:
            import hmml
        except ImportError as exc:
            raise ImportError(
                "The 'hmml' package is required for causal_backend='hmml'."
            ) from exc

        self.lags = lags
        self.distribution = distribution
        self._algorithm = hmml.HmmlExh

    def _validate_input(self, X: pd.DataFrame, y_t: str) -> None:
        if y_t not in X.columns:
            raise ValueError(f"Target variable {y_t!r} is not in X.columns.")
        if len(X) <= self.lags:
            raise ValueError(
                f"HMML requires more rows than lags; got n={len(X)}, "
                f"lags={self.lags}."
            )
        if X.isna().any().any():
            raise ValueError("HMMLRunner received NaNs.")
        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            raise ValueError("HMMLRunner expects numeric columns only.")
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("HMMLRunner received non-finite values.")

    def _beta_matrix(
        self,
        result: dict,
        columns: pd.Index,
        adjacency: np.ndarray,
    ) -> pd.DataFrame:
        beta = np.zeros((len(columns), self.lags), dtype=float)
        beta_i = result.get("beta_i")

        active = adjacency.astype(bool)
        if beta_i is None and active.any():
            raise RuntimeError("HMML selected parents but returned no beta coefficients.")

        if beta_i is not None:
            expected = int(active.sum()) * self.lags
            values = np.asarray(beta_i, dtype=float).reshape(-1)
            if values.size != expected:
                raise RuntimeError(
                    "HMML beta length is inconsistent with adjacency and lag count."
                )
            beta[active, :] = values.reshape(int(active.sum()), self.lags)

        if not np.isfinite(beta).all():
            raise RuntimeError("HMML returned non-finite beta coefficients.")

        return pd.DataFrame(
            beta.T,
            index=[f"Lag_{lag}" for lag in range(1, self.lags + 1)],
            columns=columns,
        )

    def get_betas_and_adjacency(
        self,
        X: pd.DataFrame,
        y_t: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Fit HMML once and return beta shaped (lags, variables)."""
        self._validate_input(X, y_t)

        target_indices = [X.columns.get_loc(y_t)]
        model = self._algorithm(
            X.to_numpy(dtype=float).T,
            self.lags,
            [self.distribution],
            target_indices,
        )
        results = model.fit()

        if not results:
            raise RuntimeError(
                f"HMML returned no result for target={y_t!r}, "
                f"lags={self.lags}, distribution={self.distribution!r}."
            )

        result = results[0]
        if "adjacency" not in result:
            raise RuntimeError("HMML result does not contain adjacency.")

        adjacency = np.asarray(result["adjacency"]).reshape(-1)
        if adjacency.size != X.shape[1]:
            raise RuntimeError(
                "HMML adjacency length does not match the number of variables."
            )
        if not np.isfinite(adjacency).all():
            raise RuntimeError("HMML returned non-finite adjacency values.")

        adjacency = (adjacency != 0).astype(int)
        beta = self._beta_matrix(result, X.columns, adjacency)
        return beta, adjacency
