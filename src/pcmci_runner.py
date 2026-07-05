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
    PCMCI / PCMCI+ backend for paper Step 8:
        Find causal variables B2 for target y_t on interval I2.

    For the May-paper Cause–Trigger algorithm, only lagged tau>=1 links are used
    as B2 parents. If method='pcmci_plus', contemporaneous tau=0 links are stored
    in scores['_contemporaneous_links'] for diagnostics, but are not used in B2.
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
        method: str = "pcmci",
        contemp_collider_rule: str = "majority",
        conflict_resolution: bool = True,
    ):
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.fdr_method = fdr_method
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity
        self.keep_raw_results = keep_raw_results
        self.method = method
        self.contemp_collider_rule = contemp_collider_rule
        self.conflict_resolution = conflict_resolution

        if self.method not in {"pcmci", "pcmci_plus"}:
            raise ValueError(
                f"Unsupported PCMCIBackend method={self.method!r}. "
                "Use 'pcmci' or 'pcmci_plus'."
            )

    def _make_cond_ind_test(self):
        if self.cond_ind_test == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr(significance="analytic")

        if self.cond_ind_test == "robust_parcorr":
            from tigramite.independence_tests.robust_parcorr import RobustParCorr
            return RobustParCorr(significance="analytic")

        if self.cond_ind_test == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC
            return GPDC(significance="analytic")

        if self.cond_ind_test == "cmiknn":
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn(significance="shuffle_test")

        raise ValueError(
            f"Unsupported cond_ind_test={self.cond_ind_test!r}. "
            "Use 'parcorr', 'robust_parcorr', 'gpdc', or 'cmiknn'."
        )
    
    def _run_method(self, pcmci):
        if self.method == "pcmci":
            return pcmci.run_pcmci(
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
            )

        if self.method == "pcmci_plus":
            return pcmci.run_pcmciplus(
                tau_min=0,
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
                contemp_collider_rule=self.contemp_collider_rule,
                conflict_resolution=self.conflict_resolution,
                fdr_method=self.fdr_method if self.fdr_method is not None else "none",
            )

        raise ValueError(f"Unsupported method={self.method!r}")
    
    def discover(self, X: pd.DataFrame, y_t: str) -> PCMCIResult:
        if y_t not in X.columns:
            raise ValueError(f"Target variable {y_t!r} is not in X.columns")

        if X.isna().any().any():
            raise ValueError(
                "PCMCIBackend received NaNs. Drop or impute missing values before running PCMCI."
            )
        
        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            non_numeric = X.columns[
                [not np.issubdtype(dtype, np.number) for dtype in X.dtypes]
            ].to_list()
            raise ValueError(
                f"PCMCIBackend expects all columns to be numeric. "
                f"Non-numeric columns: {non_numeric}"
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

        results = self._run_method(pcmci)

        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]
        graph = results.get("graph", None)

        # For PCMCI, we do FDR correction here manually, preserving the current behavior.
        # For PCMCI+, run_pcmciplus can already apply fdr_method internally, and the final
        # graph is the safest object for selecting links. We therefore keep p_matrix for
        # scores but use graph orientation/adjacency for PCMCI+ parent selection.
        if self.method == "pcmci" and self.fdr_method is not None:
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
        beta_values = np.zeros((self.tau_max, len(var_names)))

        contemporaneous_links = {}

        for source_idx, source_name in enumerate(var_names):
            selected_lags = []
            selected_pvals = []
            selected_vals = []

            # Store PCMCI+ contemporaneous source-target links separately.
            # These are NOT ordinary B2 parents. They can only be used later as
            # same-bin / immediate trigger candidates in the Cause--Trigger extension.
            if self.method == "pcmci_plus" and graph is not None and source_idx != target_idx:
                tau0_link = str(graph[source_idx, target_idx, 0])
                tau0_p = p_matrix[source_idx, target_idx, 0]
                tau0_val = val_matrix[source_idx, target_idx, 0]

                # Conservative choice:
                #   "-->" means source -> target.
                #   "o-o" or "x-x" are unresolved/ambiguous contemporaneous adjacency.
                # Do not use "<--", because that means target -> source in this array slot.
                eligible_tau0_links = {"-->", "o-o", "x-x"}

                if tau0_link in eligible_tau0_links and tau0_p <= self.alpha_level:
                    contemporaneous_links[source_name] = {
                        "graph_link": tau0_link,
                        "p_value": float(tau0_p),
                        "val": float(tau0_val),
                        "tau": 0,
                        "role": "contemporaneous_trigger_candidate_only",
                    }

            # Lagged links tau=1..tau_max are eligible for B2.
            for tau in range(1, self.tau_max + 1):
                sig_value = significance_matrix[source_idx, target_idx, tau]
                val = val_matrix[source_idx, target_idx, tau]

                beta_values[tau - 1, source_idx] = val

                if self.method == "pcmci_plus" and graph is not None:
                    graph_link = graph[source_idx, target_idx, tau]
                    is_selected = graph_link != ""
                else:
                    is_selected = sig_value <= self.alpha_level

                if is_selected:
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
                
        scores["_pcmci_metadata"] = {
            "method": self.method,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "alpha_level": self.alpha_level,
            "fdr_method": self.fdr_method,
            "cond_ind_test": self.cond_ind_test,
            "contemp_collider_rule": self.contemp_collider_rule,
            "conflict_resolution": self.conflict_resolution,
        }

        scores["_contemporaneous_links"] = contemporaneous_links
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