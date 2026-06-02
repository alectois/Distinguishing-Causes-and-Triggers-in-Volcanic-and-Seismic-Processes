from pathlib import Path

import numpy as np
import pandas as pd

def robust_scale_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")

    med = s.median()
    mad = np.median(np.abs(s - med))

    if pd.isna(mad) or mad == 0:
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / std

    return (s - med) / (1.4826 * mad)


def build_master_dataframe(
    wave,
    weather_vars,
    so2,
    gnss,
    start,
    end,
    master_freq="1h",
    local_eq=None,
):
    master_index = pd.date_range(
        start=start,
        end=pd.Timestamp(end) + pd.Timedelta(hours=23),
        freq=master_freq,
        tz="UTC",
    )

    wave_1h = wave.resample(master_freq).mean()
    weather_1h = weather_vars.resample(master_freq).interpolate()
    so2_1h = so2.resample(master_freq).mean()
    gnss_1h = gnss.resample("1D").mean().resample(master_freq).ffill()

    wave_1h = wave_1h.reindex(master_index)
    weather_1h = weather_1h.reindex(master_index)
    so2_1h = so2_1h.reindex(master_index)
    gnss_1h = gnss_1h.reindex(master_index)

    frames = [wave_1h, weather_1h, so2_1h, gnss_1h]

    if local_eq is not None:
        local_eq_1h = local_eq.resample(master_freq).mean()
        local_eq_1h = local_eq_1h.reindex(master_index)
        local_eq_1h = local_eq_1h.fillna(0)
        frames.append(local_eq_1h)

    whakaari_raw = pd.concat(frames, axis=1)

    whakaari_raw.index.name = "timestamp"

    return whakaari_raw


def prepare_analysis_dataframe(whakaari_raw):
    whakaari_dataset = whakaari_raw.copy()

    if "SO2_flux" in whakaari_dataset.columns:
        whakaari_dataset["SO2_flux"] = whakaari_dataset["SO2_flux"].interpolate(limit=3)

    if "API" in whakaari_dataset.columns:
        whakaari_dataset["API"] = whakaari_dataset["API"].interpolate()

    if "pressure_drop" in whakaari_dataset.columns:
        whakaari_dataset["pressure_drop"] = whakaari_dataset["pressure_drop"].fillna(0)

    if "event_rate_2_5" in whakaari_dataset.columns:
        whakaari_dataset["event_rate_2_5"] = whakaari_dataset["event_rate_2_5"].fillna(0)

    if "GNSS_deformation" in whakaari_dataset.columns:
        whakaari_dataset["GNSS_deformation"] = whakaari_dataset["GNSS_deformation"].ffill()

    if "effect_tremor_rms_5_15" in whakaari_dataset.columns:
        whakaari_dataset["effect_tremor_rms_5_15"] = whakaari_dataset["effect_tremor_rms_5_15"].interpolate()

    if "local_eq_count_1h" in whakaari_dataset.columns:
        whakaari_dataset["local_eq_count_1h"] = whakaari_dataset["local_eq_count_1h"].fillna(0)

    required_waveform_cols = [
        "hydro_rms_2_5",
        "ratio_4p5_8_over_8_16",
        "event_rate_2_5",
        "effect_tremor_rms_5_15",
        "local_eq_count_1h",
    ]

    required_existing = [
        c for c in required_waveform_cols
        if c in whakaari_dataset.columns
    ]

    whakaari_dataset = whakaari_dataset.dropna(subset=required_existing)

    return whakaari_dataset

def scale_analysis_dataframe(
    whakaari_dataset,
    log_transform_cols=None,
):
    """
    Prepare the scaled Whakaari dataset for causal analysis.

    Raw variables are not modified. For the scaled dataset:
    - non-negative, bursty/skewed variables are log1p-transformed;
    - all variables are then robustly scaled using median/MAD.
    """

    if log_transform_cols is None:
        log_transform_cols = [
            "SO2_flux",
            "hydro_rms_2_5",
            "event_rate_2_5",
            "API",
            "effect_tremor_rms_5_15",
            "local_eq_count_1h",
        ]

    analysis_prepped = whakaari_dataset.copy()

    # Log-transform only non-negative skewed variables.
    # Do not include pressure_drop or GNSS_deformation because they are signed.
    for col in log_transform_cols:
        if col in analysis_prepped.columns:
            analysis_prepped[col] = np.log1p(
                pd.to_numeric(analysis_prepped[col], errors="coerce").clip(lower=0)
            )

    feature_cols = analysis_prepped.columns.tolist()

    whakaari_dataset_scaled = pd.DataFrame(index=analysis_prepped.index)
    for col in feature_cols:
        whakaari_dataset_scaled[col + "_scaled"] = robust_scale_series(analysis_prepped[col])

    return whakaari_dataset_scaled, analysis_prepped, None

def save_whakaari_datasets(
    whakaari_raw,
    whakaari_dataset,
    whakaari_dataset_scaled,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    whakaari_raw.to_csv(output_dir / "whakaari_raw.csv")
    whakaari_dataset.to_csv(output_dir / "whakaari_dataset.csv")
    whakaari_dataset_scaled.to_csv(output_dir / "whakaari_dataset_scaled.csv")