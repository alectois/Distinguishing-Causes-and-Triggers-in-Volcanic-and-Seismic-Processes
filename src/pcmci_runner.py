from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCMCIResult:
    parents: list[str]
    lags: dict[str, list[int]]
    contemporaneous_links: dict[str, dict]


class PCMCIBackend:
    """Find lagged parents and optional PCMCI+ contemporaneous links."""

    def __init__(
        self,
        tau_max: int,
        pc_alpha: float = 0.05,
        alpha_level: float = 0.05,
        fdr_method: str | None = "fdr_bh",
        cond_ind_test: str = "parcorr",
        verbosity: int = 0,
        method: str = "pcmci",
        contemp_collider_rule: str = "majority",
        conflict_resolution: bool = True,
    ):
        if not isinstance(tau_max, int) or tau_max < 1:
            raise ValueError("tau_max must be a positive integer.")
        if method not in {"pcmci", "pcmci_plus"}:
            raise ValueError("method must be 'pcmci' or 'pcmci_plus'.")
        if cond_ind_test not in {"parcorr", "robust_parcorr"}:
            raise ValueError(
                "cond_ind_test must be 'parcorr' or 'robust_parcorr'."
            )
        if fdr_method not in {None, "fdr_bh"}:
            raise ValueError("fdr_method must be None or 'fdr_bh'.")
        if not 0 < pc_alpha < 1 or not 0 < alpha_level < 1:
            raise ValueError("pc_alpha and alpha_level must be between 0 and 1.")

        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.fdr_method = fdr_method
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity
        self.method = method
        self.contemp_collider_rule = contemp_collider_rule
        self.conflict_resolution = conflict_resolution

    def _make_cond_ind_test(self):
        if self.cond_ind_test == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr

            return ParCorr(significance="analytic")

        from tigramite.independence_tests.robust_parcorr import RobustParCorr

        return RobustParCorr(significance="analytic")

    def _run(self, pcmci):
        if self.method == "pcmci":
            return pcmci.run_pcmci(
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
            )

        return pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
            contemp_collider_rule=self.contemp_collider_rule,
            conflict_resolution=self.conflict_resolution,
            fdr_method=self.fdr_method or "none",
        )

    def _validate_input(self, X: pd.DataFrame, y_t: str) -> None:
        if y_t not in X.columns:
            raise ValueError(f"Target variable {y_t!r} is not in X.columns.")
        if len(X) <= self.tau_max:
            raise ValueError(
                f"PCMCI requires more rows than tau_max; got n={len(X)}, "
                f"tau_max={self.tau_max}."
            )
        if X.isna().any().any():
            raise ValueError("PCMCIBackend received NaNs.")
        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            raise ValueError("PCMCIBackend expects numeric columns only.")
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("PCMCIBackend received non-finite values.")

    def discover(self, X: pd.DataFrame, y_t: str) -> PCMCIResult:
        self._validate_input(X, y_t)

        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        names = list(X.columns)
        target_idx = names.index(y_t)
        dataframe = pp.DataFrame(X.to_numpy(dtype=float), var_names=names)
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self._make_cond_ind_test(),
            verbosity=self.verbosity,
        )
        results = self._run(pcmci)

        p_matrix = np.asarray(results["p_matrix"], dtype=float)
        graph = results.get("graph")

        if self.method == "pcmci" and self.fdr_method is not None:
            significance = pcmci.get_corrected_pvalues(
                p_matrix=p_matrix,
                tau_max=self.tau_max,
                fdr_method=self.fdr_method,
            )
        else:
            significance = p_matrix

        parents: list[str] = []
        selected_lags: dict[str, list[int]] = {}
        contemporaneous_links: dict[str, dict] = {}

        for source_idx, source_name in enumerate(names):
            lags = []

            if (
                self.method == "pcmci_plus"
                and graph is not None
                and source_idx != target_idx
            ):
                link = str(graph[source_idx, target_idx, 0])
                p_value = float(p_matrix[source_idx, target_idx, 0])
                if link and p_value <= self.alpha_level:
                    directed = link == "-->"
                    contemporaneous_links[source_name] = {
                        "graph_link": link,
                        "p_value": p_value,
                        "tau": 0,
                        "directed_source_to_target": directed,
                        "eligible_as_trigger_candidate": directed,
                    }

            for tau in range(1, self.tau_max + 1):
                if self.method == "pcmci_plus" and graph is not None:
                    selected = str(graph[source_idx, target_idx, tau]) == "-->"
                else:
                    selected = (
                        float(significance[source_idx, target_idx, tau])
                        <= self.alpha_level
                    )

                if selected:
                    lags.append(tau)

            if lags:
                parents.append(source_name)
                selected_lags[source_name] = lags

        return PCMCIResult(
            parents=parents,
            lags=selected_lags,
            contemporaneous_links=contemporaneous_links,
        )
