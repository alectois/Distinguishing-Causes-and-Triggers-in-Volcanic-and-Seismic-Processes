"""
This file is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Zenodo. DOI: 10.5281/zenodo.15109084
"""

import hmml
import numpy as np
import pandas as pd
"""
This class handles all computations and data transformations necessary for the HMML algorithm.
"""

class HMMLRunner():
    def __init__(self, lags: int = None, distribution: str = None, mode: str = "exhaustive"):
        self.lags = lags
        self.distribution = distribution
        self.requested_distribution = distribution
        self.used_distribution = distribution
        self.used_fallback = False

        algorithms = {
            "exhaustive": hmml.HmmlExh,
            "genetic": hmml.HmmlGa,
        }
        self.HMML = algorithms[mode]
        
    def run_hmml(self, X: pd.DataFrame, y_t: str):

        target_indices = [X.columns.get_loc(y_t)]
        hmmlga = self.HMML(np.transpose(X.to_numpy()), self.lags, [self.distribution], target_indices)
        return hmmlga.fit()

    def transform_beta_matrix(self,result_dict,X_columns):
        beta = np.zeros([len(X_columns), self.lags])

        if "beta_i" in result_dict.keys():
            beta_i = result_dict["beta_i"]
            sequence = result_dict["adjacency"]

            if beta_i is not None:
                beta[sequence.astype(bool), :] = beta_i.reshape(np.count_nonzero(sequence), self.lags)

        columns = [f"Lag_{lag}" for lag in range(1,self.lags+1)]
        return pd.DataFrame(beta,index=X_columns, columns=columns)

    def get_betas_and_adjacency(self, X: pd.DataFrame, y_t: str):
        if y_t not in X.columns:
            raise ValueError(f"Target variable {y_t!r} is not in X.columns")

        if X.isna().any().any():
            raise ValueError("HMMLRunner received NaNs. Drop or impute missing values first.")

        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            raise ValueError("HMMLRunner expects all columns to be numeric.")

        original_distribution = self.distribution

        self.used_distribution = original_distribution
        self.used_fallback = False

        results = self.run_hmml(X, y_t=y_t)

        if len(results) == 0:
            self.used_fallback = True
            self.used_distribution = "gaussian"

            self.distribution = "gaussian"
            results = self.run_hmml(X, y_t=y_t)

            # Restore requested distribution after fallback attempt.
            self.distribution = original_distribution

        if len(results) == 0:
            raise RuntimeError(
                f"HMML returned no results for target={y_t}, "
                f"lags={self.lags}, requested_distribution={self.requested_distribution}, "
                f"used_distribution={self.used_distribution}"
            )

        adjacency = results[0]["adjacency"]
        betas = self.transform_beta_matrix(results[0], X.columns).T

        return betas, adjacency