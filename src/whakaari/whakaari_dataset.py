from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _utc_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="raise")
    out = out.sort_index()
    out.index.name = name
    if out.index.has_duplicates:
        raise ValueError(f"{name} dataframe contains duplicate timestamps.")
    return out


def past_rolling_median_state(
    series: pd.Series,
    *,
    window: int = 6,
    min_periods: int = 3,
) -> pd.Series:
    """Past-only rolling median, shifted by one hourly sample."""
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(window=window, min_periods=min_periods)
        .median()
        .shift(1)
    )


def positive_log_epsilon(series: pd.Series) -> float:
    """Estimate a small positive log offset from a reference interval."""
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"Cannot estimate log epsilon because {series.name!r} contains NaNs."
        )
    if (values < 0).any():
        raise ValueError(
            f"Cannot estimate log epsilon because {series.name!r} contains negative values."
        )

    positive = values[values > 0]
    if positive.empty:
        return 1e-30

    return max(float(positive.quantile(0.01)) * 0.1, 1e-30)


def safe_log_positive(
    series: pd.Series,
    eps: float | None = None,
) -> pd.Series:
    """Log-transform a non-negative physical amplitude series."""
    values = pd.to_numeric(series, errors="coerce")
    if (values < 0).any():
        raise ValueError(
            f"safe_log_positive received negative values in {series.name!r}."
        )

    if eps is None:
        eps = positive_log_epsilon(values)

    return np.log(values.clip(lower=0) + float(eps))


def positive_past_log_anomaly(
    series: pd.Series,
    *,
    baseline_window: int = 24,
    min_periods: int = 12,
    eps: float = 1e-30,
) -> pd.Series:
    """Positive log excess above a past-only rolling-median baseline."""
    log_values = safe_log_positive(series, eps=eps)
    baseline = (
        log_values
        .rolling(window=baseline_window, min_periods=min_periods)
        .median()
        .shift(1)
    )
    return (log_values - baseline).clip(lower=0)


def fit_cause_trigger_transform_parameters(
    reference: pd.DataFrame,
) -> dict:
    """Fit data-dependent transformation parameters on the reference interval."""
    parameters = {"log_positive_eps": {}}

    if "hydro_2_5" in reference.columns:
        parameters["log_positive_eps"]["hydro_2_5"] = (
            positive_log_epsilon(reference["hydro_2_5"])
        )

    return parameters


def transform_for_cause_trigger_scaling(
    final: pd.DataFrame,
    *,
    transform_parameters: dict | None = None,
) -> pd.DataFrame:
    """Apply variable-family transformations before standardization."""
    transformed = final.copy()

    if transform_parameters is None:
        transform_parameters = fit_cause_trigger_transform_parameters(transformed)

    log_positive_eps = transform_parameters.get("log_positive_eps", {})

    if "hydro_2_5" in transformed.columns:
        if "hydro_2_5" not in log_positive_eps:
            raise ValueError("Missing reference-fitted log epsilon for 'hydro_2_5'.")
        transformed["hydro_2_5"] = safe_log_positive(
            transformed["hydro_2_5"],
            eps=float(log_positive_eps["hydro_2_5"]),
        )

    if "rainfall_mm" in transformed.columns:
        transformed["rainfall_mm"] = np.log1p(
            _numeric_series(
                transformed,
                "rainfall_mm",
            ).clip(lower=0)
        )

    for column in ("pressure_drop", "GNSS_deformation_rate"):
        if column in transformed.columns:
            transformed[column] = np.arcsinh(
                _numeric_series(transformed, column)
            )

    return transformed


def standard_scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return population-standardized numeric columns."""
    numeric = df.apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        missing = numeric.isna().sum()
        raise ValueError(
            "Cannot standardize because transformed data contain NaNs:\n"
            f"{missing[missing > 0]}"
        )

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Cannot standardize non-finite transformed values.")

    constant_columns = [
        column
        for column in numeric.columns
        if numeric[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        raise ValueError(
            f"Cannot standardize constant columns: {constant_columns}"
        )

    scaled = StandardScaler().fit_transform(numeric)
    return pd.DataFrame(
        scaled,
        index=numeric.index,
        columns=numeric.columns,
    )


def _full_day_hourly_index(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str,
) -> pd.DatetimeIndex:
    start_timestamp = pd.to_datetime(start, utc=True)
    end_timestamp = pd.to_datetime(end, utc=True)

    if end_timestamp == end_timestamp.normalize():
        end_exclusive = end_timestamp + pd.Timedelta(days=1)
    else:
        end_exclusive = end_timestamp

    return pd.date_range(
        start=start_timestamp,
        end=end_exclusive,
        freq=frequency,
        inclusive="left",
    )


def build_master_dataframe(
    wave: pd.DataFrame,
    weather_vars: pd.DataFrame,
    gnss: pd.DataFrame,
    start,
    end,
    master_freq: str = "1h",
) -> pd.DataFrame:
    """
    Align waveform, weather, and past-only GNSS variables on one hourly grid.

    The GNSS value assigned to day D is the displacement change from D-2 to
    D-1, so the entire hourly step for D uses information available before D.
    """
    master_index = _full_day_hourly_index(start, end, master_freq)

    wave_hourly = _utc_index(wave, "time").resample(master_freq).mean()
    wave_hourly = wave_hourly.reindex(master_index)

    if "spectral_log_ratio_4p5_8_over_8_16" in wave_hourly.columns:
        wave_hourly["spectral_log_ratio_4p5_8_over_8_16"] = (
            past_rolling_median_state(
                wave_hourly["spectral_log_ratio_4p5_8_over_8_16"],
                window=6,
                min_periods=3,
            )
        )

    if "effect_tremor_5_15" in wave_hourly.columns:
        wave_hourly["effect_tremor_5_15"] = positive_past_log_anomaly(
            wave_hourly["effect_tremor_5_15"],
            baseline_window=24,
            min_periods=12,
        )

    weather_hourly = (
        _utc_index(weather_vars, "timestamp")
        .resample(master_freq)
        .mean()
        .reindex(master_index)
    )

    gnss_daily = _utc_index(gnss, "timestamp").resample("1D").mean()
    if "GNSS_deformation" not in gnss_daily.columns:
        raise ValueError("GNSS dataframe must contain 'GNSS_deformation'.")

    gnss_daily["GNSS_deformation_rate"] = (
        gnss_daily["GNSS_deformation"]
        .diff()
        .shift(1)
    )

    # Reindex first, then forward-fill within each day. Resampling alone stops at
    # the final daily timestamp and previously created 23 artificial end NaNs.
    gnss_hourly = (
        gnss_daily[["GNSS_deformation_rate"]]
        .reindex(master_index)
        .ffill(limit=23)
    )

    master = pd.concat(
        [wave_hourly, weather_hourly, gnss_hourly],
        axis=1,
    )
    master.index = master_index
    master.index.name = "time"
    return master


def prepare_analysis_dataframe(
    master: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep all retained variables and remove only rows unresolved in any of them.

    Missing waveform hours are not interpolated. They remain explicit missing
    timestamps in the canonical complete-case dataframe.
    """
    analysis = _utc_index(master, "time")

    if required_columns is None:
        required_columns = [
            "hydro_2_5",
            "spectral_log_ratio_4p5_8_over_8_16",
            "effect_tremor_5_15",
            "rainfall_mm",
            "pressure_drop",
            "GNSS_deformation_rate",
        ]

    missing_columns = sorted(set(required_columns) - set(analysis.columns))
    if missing_columns:
        raise ValueError(
            f"Whakaari master dataframe is missing columns: {missing_columns}"
        )

    analysis = analysis[required_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    analysis = analysis.dropna(subset=required_columns)

    if not np.isfinite(analysis.to_numpy()).all():
        raise ValueError("Whakaari analysis dataframe contains non-finite values.")

    return analysis


def _timestamp_ranges(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert sorted hourly timestamps into compact continuous ranges."""
    index = pd.DatetimeIndex(index).sort_values().unique()

    if len(index) == 0:
        return pd.DataFrame(columns=["start", "end", "hours"])

    rows: list[dict[str, object]] = []
    start = previous = index[0]

    for timestamp in index[1:]:
        if timestamp == previous + pd.Timedelta(hours=1):
            previous = timestamp
            continue

        rows.append(
            {
                "start": start,
                "end": previous,
                "hours": int((previous - start) / pd.Timedelta("1h")) + 1,
            }
        )
        start = previous = timestamp

    rows.append(
        {
            "start": start,
            "end": previous,
            "hours": int((previous - start) / pd.Timedelta("1h")) + 1,
        }
    )
    return pd.DataFrame(rows)


def preprocessing_report(
    rawest_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return compact construction, missing-value, and dropped-range tables."""
    master = _utc_index(rawest_df, "time")
    prepared = _utc_index(prepared_df, "time")
    dropped = master.index.difference(prepared.index)

    summary = pd.DataFrame(
        [
            {
                "master_rows": len(master),
                "analysis_rows": len(prepared),
                "rows_removed": len(dropped),
                "analysis_start": prepared.index.min(),
                "analysis_end": prepared.index.max(),
                "duplicate_timestamps": int(prepared.index.duplicated().sum()),
                "remaining_missing_values": int(prepared.isna().sum().sum()),
            }
        ]
    )

    missing_by_variable = pd.DataFrame(
        {
            "missing_in_master": master.isna().sum(),
            "missing_in_analysis": prepared.isna().sum().reindex(
                master.columns,
                fill_value=0,
            ),
        }
    )
    missing_by_variable.index.name = "variable"
    missing_by_variable = missing_by_variable.reset_index()

    return {
        "summary": summary,
        "missing_by_variable": missing_by_variable,
        "dropped_ranges": _timestamp_ranges(dropped),
    }


def save_whakaari_analysis_dataset(
    analysis: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Validate and save the canonical complete-case Whakaari dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _utc_index(analysis, "time")

    scaled_columns = [
        column for column in frame.columns if column.endswith("_scaled")
    ]
    if scaled_columns:
        raise ValueError(
            "Whakaari canonical data must remain unstandardized. "
            f"Found: {scaled_columns}"
        )

    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(
            "Whakaari analysis data contain missing values:\n"
            f"{missing[missing > 0]}"
        )

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Whakaari analysis data contain non-finite values.")

    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Whakaari timestamps must be unique and sorted.")

    if not (
        (frame.index.minute == 0)
        & (frame.index.second == 0)
        & (frame.index.microsecond == 0)
    ).all():
        raise ValueError("Whakaari timestamps must lie on the hourly grid.")

    output_path = output_dir / "whakaari_dataset.csv"
    frame.to_csv(output_path, index_label="time")
    return output_path
