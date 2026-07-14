from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from etna.etna_dataset import (
    fit_cause_trigger_transform_parameters as fit_etna_transform_parameters,
    standard_scale_dataframe as standard_scale_etna_dataframe,
    transform_for_cause_trigger_scaling as transform_etna_for_scaling,
)
from whakaari.whakaari_dataset import (
    fit_cause_trigger_transform_parameters as fit_whakaari_transform_parameters,
    standard_scale_dataframe as standard_scale_whakaari_dataframe,
    transform_for_cause_trigger_scaling as transform_whakaari_for_scaling,
)


@dataclass(frozen=True)
class CaseSpec:
    """Case-specific variables and preprocessing functions."""

    name: str
    effect: str
    raw_effect: str
    index_name: str
    model_columns: Sequence[str]
    variable_labels: Mapping[str, str]
    fit_transform_parameters: Callable[[pd.DataFrame], dict]
    transform_for_scaling: Callable[..., pd.DataFrame]
    standard_scale_dataframe: Callable[[pd.DataFrame], pd.DataFrame]


ETNA_CASE = CaseSpec(
    name="Etna",
    effect="local_event_rate_response_scaled",
    raw_effect="local_event_rate_response",
    index_name="time",
    model_columns=(
        "teleseismic",
        "local_event_rate_state",
        "local_event_rate_response",
        "CO2_3",
        "rainfall_mm",
        "pressure_drop",
    ),
    variable_labels={
        "teleseismic_scaled": "Teleseismic RMS",
        "local_event_rate_state_scaled": "Past local seismicity",
        "local_event_rate_response_scaled": "Catalogue response (effect)",
        "CO2_3_scaled": "Soil CO₂",
        "rainfall_mm_scaled": "Rainfall",
        "pressure_drop_scaled": "Pressure drop",
    },
    fit_transform_parameters=fit_etna_transform_parameters,
    transform_for_scaling=transform_etna_for_scaling,
    standard_scale_dataframe=standard_scale_etna_dataframe,
)


WHAKAARI_CASE = CaseSpec(
    name="Whakaari",
    effect="effect_tremor_5_15_scaled",
    raw_effect="effect_tremor_5_15",
    index_name="time",
    model_columns=(
        "hydro_2_5",
        "spectral_log_ratio_4p5_8_over_8_16",
        "effect_tremor_5_15",
        "rainfall_mm",
        "pressure_drop",
        "GNSS_deformation_rate",
    ),
    variable_labels={
        "hydro_2_5_scaled": "Hydrothermal RMS (2–5 Hz)",
        "spectral_log_ratio_4p5_8_over_8_16_scaled": "Past spectral contrast",
        "effect_tremor_5_15_scaled": "Tremor anomaly (effect)",
        "rainfall_mm_scaled": "Rainfall",
        "pressure_drop_scaled": "Pressure drop",
        "GNSS_deformation_rate_scaled": "Lagged GNSS deformation change",
    },
    fit_transform_parameters=fit_whakaari_transform_parameters,
    transform_for_scaling=transform_whakaari_for_scaling,
    standard_scale_dataframe=standard_scale_whakaari_dataframe,
)
