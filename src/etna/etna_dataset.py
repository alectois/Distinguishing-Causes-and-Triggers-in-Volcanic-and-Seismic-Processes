import pandas as pd
import numpy as np
from pathlib import Path
import requests
from sklearn.preprocessing import StandardScaler

def merge_data(base, ext, base_time, ext_time, value_cols, tolerance_hours):
    base = base.copy()
    ext = ext.copy()

    base[base_time] = pd.to_datetime(base[base_time], utc=True, errors="coerce")
    ext[ext_time] = pd.to_datetime(ext[ext_time], utc=True, errors="coerce")

    base = base.dropna(subset=[base_time]).sort_values(base_time)
    ext = ext.dropna(subset=[ext_time]).sort_values(ext_time)
    missing_value_cols = sorted(set(value_cols) - set(ext.columns))

    if missing_value_cols:
        raise ValueError(
            f"External dataframe is missing requested columns: {missing_value_cols}"
        )
    keep = [ext_time] + [c for c in value_cols if c in ext.columns]
    ext = ext[keep].copy()

    merged = pd.merge_asof(
        base,
        ext,
        left_on=base_time,
        right_on=ext_time,
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )

    if ext_time != base_time and ext_time in merged.columns:
        merged = merged.drop(columns=[ext_time])

    return merged

def load_etna_event_catalog_xls(
    path: str | Path,
    *,
    sheet_name=0,
    quality_filter: bool = False,
) -> pd.DataFrame:
    """
    Load the EtnaSC 2000--2010 catalogue and construct UTC timestamps.

    Expected columns:
        YE, MO, DA, HR, MI, SE, MD, ML, LAT, LON,
        DEPSL, DEPGL, N.O., RMS, GAP, ERZ, ERH

    The returned dataframe contains a 'timestamp' column.
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)

    required = ["YE", "MO", "DA", "HR", "MI", "SE"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing catalogue time columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    base_time = pd.to_datetime(
        {
            "year": df["YE"],
            "month": df["MO"],
            "day": df["DA"],
            "hour": df["HR"],
            "minute": df["MI"],
        },
        utc=True,
        errors="coerce",
    )

    df["timestamp"] = base_time + pd.to_timedelta(df["SE"], unit="s")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [
        "MD",
        "ML",
        "LAT",
        "LON",
        "DEPSL",
        "DEPGL",
        "N.O.",
        "RMS",
        "GAP",
        "ERZ",
        "ERH",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if quality_filter:
        required_quality = ["N.O.", "RMS", "GAP"]
        missing_quality = [c for c in required_quality if c not in df.columns]
        if missing_quality:
            raise ValueError(f"Cannot quality-filter catalogue; missing: {missing_quality}")

        df = df[
            (df["N.O."] >= 8)
            & (df["RMS"] <= 0.30)
            & (df["GAP"] <= 250)
        ].copy()

    return df

def catalogue_hourly_counts(
    catalog_df: pd.DataFrame,
    master_index: pd.DatetimeIndex,
) -> pd.Series:
    events = catalog_df.copy()

    if "timestamp" not in events.columns:
        if "time" in events.columns:
            events = events.rename(columns={"time": "timestamp"})
        else:
            raise ValueError(
                "Catalogue dataframe must contain 'timestamp' or 'time'."
            )

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        utc=True,
        errors="coerce",
    )
    events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

    master_index = pd.DatetimeIndex(
        pd.to_datetime(master_index, utc=True)
    )

    return (
        events
        .set_index("timestamp")
        .assign(count=1)["count"]
        .resample("1h")
        .sum()
        .reindex(master_index, fill_value=0)
        .astype(float)
    )

def catalogue_event_rate_response(
    catalog_df: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    *,
    response_window: int = 6,
    baseline_window: int = 24,
    min_periods: int = 12,
) -> pd.Series:
    """
    Short-term local seismic response.

    At time t:
    - response window: counts from t-5 through t;
    - baseline: earlier six-hour count windows ending at least six hours
      before t.

    The current response window therefore does not overlap with the
    contemporaneous background-state window.
    """
    hourly_count = catalogue_hourly_counts(catalog_df, master_index)

    recent_count = hourly_count.rolling(
        window=response_window,
        min_periods=1,
    ).sum()

    log_recent_count = np.log1p(recent_count)

    past_baseline = (
        log_recent_count
        .shift(response_window)
        .rolling(
            window=baseline_window,
            min_periods=min_periods,
        )
        .median()
    )

    response = (log_recent_count - past_baseline).clip(lower=0)
    response.name = "local_event_rate_response"
    return response

def catalogue_event_rate_state(
    catalog_df: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    *,
    state_window: int = 48,
    exclusion_hours: int = 6,
    min_periods: int = 24,
) -> pd.Series:
    """
    Past local-seismicity state.

    At time t the state uses only counts ending six hours before t.
    It therefore excludes the six-hour window used by the response proxy.
    """
    hourly_count = catalogue_hourly_counts(catalog_df, master_index)

    past_count_sum = (
        hourly_count
        .shift(exclusion_hours)
        .rolling(
            window=state_window,
            min_periods=min_periods,
        )
        .sum()
    )

    state = np.log1p(past_count_sum)
    state.name = "local_event_rate_state"
    return state


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def safe_log_positive(s: pd.Series, eps: float | None = None) -> pd.Series:
    """
    Log-transform strictly non-negative physical amplitudes.

    better than log1p for seismic RMS values because RMS velocities
    can be much smaller than 1, where log1p(x) is almost identical to x.
    """
    x = pd.to_numeric(s, errors="coerce")

    if (x < 0).any():
        raise ValueError(f"safe_log_positive received negative values in {s.name!r}")

    positive = x[x > 0]

    if eps is None:
        if len(positive) == 0:
            eps = 1e-30
        else:
            eps = max(float(positive.quantile(0.01)) * 0.1, 1e-30)

    return np.log(x.clip(lower=0) + eps)


def transform_for_cause_trigger_scaling(final: pd.DataFrame) -> pd.DataFrame:
    """
    Apply family-aware transformations before StandardScaler.

    final_raw remains unchanged. This transformed copy is used only for
    constructing etna_final.csv.
    """
    transformed = final.copy()

    log_positive_cols = [
        "teleseismic",
    ]

    log1p_cols = [
        "rainfall_mm",
        "CO2_3",
    ]

    asinh_cols = [
        "pressure_drop",
    ]

    for col in log_positive_cols:
        if col in transformed.columns:
            transformed[col] = safe_log_positive(transformed[col])

    for col in log1p_cols:
        if col in transformed.columns:
            transformed[col] = np.log1p(
                _numeric_series(transformed, col).clip(lower=0)
            )

    for col in asinh_cols:
        if col in transformed.columns:
            transformed[col] = np.arcsinh(
                _numeric_series(transformed, col)
            )

    return transformed

def standard_scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScaler-compatible z-standardization:
        mean 0, variance 1.

    Raises an error for constant or invalid columns instead of silently creating
    unusable model input.
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        missing = numeric.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        raise ValueError(f"Cannot standardize because transformed data contain NaNs:\n{missing}")

    if not np.isfinite(numeric.to_numpy()).all():
        bad_cols = [
            col
            for col in numeric.columns
            if not np.isfinite(numeric[col].to_numpy()).all()
        ]
        raise ValueError(
            "Cannot standardize because transformed data contain infinite values: "
            f"{bad_cols}"
        )
    
    constant_cols = [
        col for col in numeric.columns
        if numeric[col].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        raise ValueError(f"Cannot standardize constant columns: {constant_cols}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    return pd.DataFrame(
        scaled,
        index=numeric.index,
        columns=numeric.columns,
    )

def create_etna_dataset(
    wave_df: pd.DataFrame,
    station_name: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    catalog_df: pd.DataFrame | None = None,
    etnagas_df: pd.DataFrame | None = None,
    etnagas_cols: list[str] | None = None,
    etnagas_buffer_hours: int = 6,
    etnagas_tolerance_hours: int = 1,
    weather_df: pd.DataFrame | None = None,
    weather_cols: list[str] | None = None,
    weather_buffer_hours: int = 6,
    weather_tolerance_hours: int = 1,
    output_dir: str | Path | None = None,
):
    # ---- load station waveform features ----
    df = wave_df.copy().reset_index()

    if "time" not in df.columns:
        if "index" in df.columns:
            df = df.rename(columns={"index": "time"})
        else:
            raise KeyError("Waveform dataframe must have a DatetimeIndex or a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    seismic_cols = [
        "teleseismic",
    ]

    for c in seismic_cols:
        if c not in df.columns:
            raise KeyError(f"Missing '{c}' in waveform dataframe.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    analysis_start = (
        pd.to_datetime(start_time, utc=True)
        if start_time is not None
        else df["time"].min().floor("h")
    )

    analysis_end = (
        pd.to_datetime(end_time, utc=True)
        if end_time is not None
        else df["time"].max().ceil("h") + pd.Timedelta(hours=1)
    )

    df = df[
        (df["time"] >= analysis_start)
        & (df["time"] < analysis_end)
    ].copy()

    master_index = pd.date_range(
        start=analysis_start,
        end=analysis_end,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )

    base = (
        df
        .set_index("time")[seismic_cols]
        .resample("1h")
        .mean()
        .reindex(master_index)
        .rename_axis("time")
        .reset_index()
    )

    if catalog_df is None:
        raise ValueError(
            "catalog_df is required because the final Etna effect is "
            "local_event_rate_response from the Etna event catalogue."
        )

    base["local_event_rate_state"] = catalogue_event_rate_state(
        catalog_df,
        pd.DatetimeIndex(base["time"]),
        state_window=48,
        exclusion_hours=6,
        min_periods=24,
    ).to_numpy()

    base["local_event_rate_response"] = catalogue_event_rate_response(
        catalog_df,
        pd.DatetimeIndex(base["time"]),
        response_window=6,
        baseline_window=24,
        min_periods=12,
    ).to_numpy()

    # Drop only the first rows where past-only state/response construction is undefined.
    base = base.dropna(
        subset=["local_event_rate_state", "local_event_rate_response"]
    ).reset_index(drop=True)

    model_cols = [
        "time",
        "teleseismic",
        "local_event_rate_state",
        "local_event_rate_response",
    ]

    base["station"] = station_name
    base = base[["station", *model_cols]]

    # ---- ETNAGAS ----
    if etnagas_df is not None and etnagas_cols:
        g = etnagas_df.copy()

        tmin = base["time"].iloc[0] - pd.Timedelta(hours=etnagas_buffer_hours)
        tmax = base["time"].iloc[-1] + pd.Timedelta(hours=etnagas_buffer_hours)

        g = g[(g["timestamp"] >= tmin) & (g["timestamp"] <= tmax)].copy()

        base = merge_data(
            base=base,
            ext=g,
            base_time="time",
            ext_time="timestamp",
            value_cols=etnagas_cols,
            tolerance_hours=etnagas_tolerance_hours,
        )

    # ---- Open-Meteo weather ----
    if weather_df is not None and weather_cols:
        w = weather_df.copy()

        tmin = base["time"].iloc[0] - pd.Timedelta(hours=weather_buffer_hours)
        tmax = base["time"].iloc[-1] + pd.Timedelta(hours=weather_buffer_hours)

        w = w[(w["timestamp"] >= tmin) & (w["timestamp"] <= tmax)].copy()

        base = merge_data(
            base=base,
            ext=w,
            base_time="time",
            ext_time="timestamp",
            value_cols=weather_cols,
            tolerance_hours=weather_tolerance_hours,
        )

    analysis = base.sort_values("time").reset_index(drop=True)

    unexpected_suffix_columns = [
        column
        for column in analysis.columns
        if column.endswith("_scaled")
    ]

    if unexpected_suffix_columns:
        raise ValueError(
            "Etna analysis dataset must remain unstandardized. "
            f"Found scaled columns: {unexpected_suffix_columns}"
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "etna_dataset.csv"
        analysis.to_csv(output_path, index=False)

    return analysis

def load_etnagas_csv(path, value_cols):
    df = pd.read_csv(path).replace("NULL", np.nan)
    df["timestamp"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    source_cols = [c for c in value_cols if c in df.columns]
    if "pressure_drop" in value_cols and "Patm_3" in df.columns:
        source_cols = list(dict.fromkeys(source_cols + ["Patm_3"]))

    df = (
        df[["timestamp"] + source_cols]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
    )

    for col in source_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    hourly = (
        df
        .set_index("timestamp")
        .resample("1h")
        .mean()
    )

    continuous_cols = [
        column
        for column in source_cols
        if column != "Patm_3"
    ]

    if continuous_cols:
        hourly[continuous_cols] = hourly[continuous_cols].interpolate(
            method="time",
            limit=1,
            limit_area="inside",
        )

    if "Patm_3" in hourly.columns:
        pressure = hourly["Patm_3"].interpolate(
            method="time",
            limit=1,
            limit_area="inside",
        )
        hourly["Patm_3"] = pressure

    df = hourly.reset_index()

    if "pressure_drop" in value_cols:
        if "Patm_3" not in df.columns:
            raise KeyError(
                "Cannot compute pressure_drop because 'Patm_3' is missing."
            )

        df["pressure_drop"] = -df["Patm_3"].diff()

    final_cols = ["timestamp"] + [c for c in value_cols if c in df.columns]
    return df[final_cols]


def load_openmeteo_etna_weather(
    start_date: str,
    end_date: str,
    latitude: float = 37.75,
    longitude: float = 14.99,
):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=precipitation"
        "&timezone=UTC"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "rainfall_mm": data["hourly"]["precipitation"],
    }).sort_values("timestamp")

    weather["rainfall_mm"] = pd.to_numeric(weather["rainfall_mm"], errors="coerce")

    if weather["rainfall_mm"].isna().any():
        missing = int(weather["rainfall_mm"].isna().sum())
        raise ValueError(
            f"Open-Meteo Etna precipitation contains {missing} missing values."
        )
    return weather[["timestamp", "rainfall_mm"]]