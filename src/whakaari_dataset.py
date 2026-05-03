from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_master_dataframe(
    wave,
    weather_vars,
    so2,
    gnss,
    start,
    end,
    master_freq="1h",
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

    master_df = pd.concat(
        [wave_1h, weather_1h, so2_1h, gnss_1h],
        axis=1,
    )

    master_df.index.name = "timestamp"

    return master_df


def prepare_analysis_dataframe(master_df):
    analysis_df = master_df.copy()

    if "SO2_flux" in analysis_df.columns:
        analysis_df["SO2_flux"] = analysis_df["SO2_flux"].interpolate(limit=3)

    if "API" in analysis_df.columns:
        analysis_df["API"] = analysis_df["API"].interpolate()

    if "pressure_drop" in analysis_df.columns:
        analysis_df["pressure_drop"] = analysis_df["pressure_drop"].fillna(0)

    if "hf_event_rate_2_5" in analysis_df.columns:
        analysis_df["hf_event_rate_2_5"] = analysis_df["hf_event_rate_2_5"].fillna(0)

    if "GNSS_deformation" in analysis_df.columns:
        analysis_df["GNSS_deformation"] = analysis_df["GNSS_deformation"].ffill()

    if "effect_tremor_rms_5_15" in analysis_df.columns:
        analysis_df["effect_tremor_rms_5_15"] = analysis_df["effect_tremor_rms_5_15"].interpolate()

    required_waveform_cols = [
        "hydro_rms_2_5",
        "ratio_4p5_8_over_8_16",
        "hf_event_rate_2_5",
        "effect_tremor_rms_5_15",
    ]

    required_existing = [
        c for c in required_waveform_cols
        if c in analysis_df.columns
    ]

    analysis_df = analysis_df.dropna(subset=required_existing)

    return analysis_df

def scale_analysis_dataframe(
    analysis_df,
    log_transform_cols=None,
):
    if log_transform_cols is None:
        log_transform_cols = [
            "SO2_flux",
            "hydro_rms_2_5",
            "hf_event_rate_2_5",
            "API",
            "effect_tremor_rms_5_15",
        ]

    analysis_prepped = analysis_df.copy()

    for col in log_transform_cols:
        if col in analysis_prepped.columns:
            analysis_prepped[col] = np.log1p(
                analysis_prepped[col].clip(lower=0)
            )

    feature_cols = analysis_prepped.columns.tolist()

    scaler = StandardScaler()
    analysis_scaled = analysis_prepped.copy()
    analysis_scaled[feature_cols] = scaler.fit_transform(
        analysis_prepped[feature_cols]
    )

    return analysis_scaled, analysis_prepped, scaler

def save_whakaari_datasets(
    master_df,
    analysis_df,
    analysis_scaled,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_df.to_pickle(output_dir / "whakaari_raw.pkl")
    analysis_df.to_pickle(output_dir / "whakaari_dataset.pkl")
    analysis_scaled.to_pickle(output_dir / "whakaari_dataset_scaled.pkl")

    master_df.to_csv(output_dir / "whakaari_raw.csv")
    analysis_df.to_csv(output_dir / "whakaari_dataset.csv")
    analysis_scaled.to_csv(output_dir / "whakaari_dataset_scaled.csv")