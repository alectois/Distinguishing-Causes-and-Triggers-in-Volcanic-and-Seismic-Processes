from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class PCMCIResult:
    parents: List[str]
    adjacency: np.ndarray
    lags: Dict[str, List[int]]
    scores: Dict[str, Dict[str, float]]
    beta: pd.DataFrame
    raw_results: Optional[Dict[str, Any]] = None


class PCMCIBackend:
    """
    PCMCI backend for paper Step 8:
        Find causal variables B2 for target y_t on interval I2.

    This backend returns B2 as a set of variables. If any lag of x_j significantly
    predicts y_t, then x_j is included in B2.
    """

    def __init__(
        self,
        tau_max: int,
        pc_alpha: float = 0.05,
        alpha_level: float = 0.05,
        fdr_method: str = "fdr_bh",
        cond_ind_test: str = "parcorr",
        verbosity: int = 0,
        keep_raw_results: bool = False,
    ):
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.fdr_method = fdr_method
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity
        self.keep_raw_results = keep_raw_results

    def _make_cond_ind_test(self):
        if self.cond_ind_test == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr

            return ParCorr(significance="analytic")

        raise ValueError(
            f"Unsupported cond_ind_test={self.cond_ind_test!r}. "
            "Start with 'parcorr'."
        )

    def discover(self, X: pd.DataFrame, y_t: str) -> PCMCIResult:
        if y_t not in X.columns:
            raise ValueError(f"Target variable {y_t!r} is not in X.columns")

        if X.isna().any().any():
            raise ValueError(
                "PCMCIBackend received NaNs. Drop or impute missing values before running PCMCI."
            )

        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        var_names = list(X.columns)
        target_idx = var_names.index(y_t)

        dataframe = pp.DataFrame(
            X.to_numpy(),
            var_names=var_names,
        )

        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self._make_cond_ind_test(),
            verbosity=self.verbosity,
        )

        results = pcmci.run_pcmci(
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
        )

        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]

        if self.fdr_method is not None:
            q_matrix = pcmci.get_corrected_pvalues(
                p_matrix=p_matrix,
                tau_max=self.tau_max,
                fdr_method=self.fdr_method,
            )
            significance_matrix = q_matrix
            score_name = "min_q"
        else:
            q_matrix = None
            significance_matrix = p_matrix
            score_name = "min_p"

        parents = []
        lags = {}
        scores = {}

        # beta-like matrix with rows Lag_1 ... Lag_tau_max and columns variable names.
        # For PCMCI this is not a regression beta. We store val_matrix strengths so that
        # beta_is_ones=True remains the default for paper-compatible V.
        beta_values = np.zeros((self.tau_max, len(var_names)))

        for source_idx, source_name in enumerate(var_names):
            selected_lags = []
            selected_pvals = []
            selected_vals = []

            # Standard PCMCI lagged links: tau=1..tau_max.
            # tau=0 is contemporaneous and is not part of the paper's lagged Granger-style setup.
            for tau in range(1, self.tau_max + 1):
                sig_value = significance_matrix[source_idx, target_idx, tau]
                val = val_matrix[source_idx, target_idx, tau]

                beta_values[tau - 1, source_idx] = val

                if sig_value <= self.alpha_level:
                    selected_lags.append(tau)
                    selected_pvals.append(sig_value)
                    selected_vals.append(val)

            if selected_lags:
                parents.append(source_name)
                lags[source_name] = selected_lags
                scores[source_name] = {
                    score_name: float(np.min(selected_pvals)),
                    "max_abs_val": float(np.max(np.abs(selected_vals))),
                    "best_lag": int(selected_lags[int(np.argmax(np.abs(selected_vals)))]),
                }

        adjacency = np.array([name in parents for name in var_names], dtype=int)

        beta = pd.DataFrame(
            beta_values,
            index=[f"Lag_{lag}" for lag in range(1, self.tau_max + 1)],
            columns=var_names,
        )

        return PCMCIResult(
            parents=parents,
            adjacency=adjacency,
            lags=lags,
            scores=scores,
            beta=beta,
            raw_results=results if self.keep_raw_results else None,
        )