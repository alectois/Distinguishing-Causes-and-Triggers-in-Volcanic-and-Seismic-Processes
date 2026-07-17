from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler


def _to_utc_timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def merge_hourly_data(
    base: pd.DataFrame,
    external: pd.DataFrame,
    *,
    base_time: str,
    external_time: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Join an external hourly dataframe by exact timestamp."""
    missing = sorted(set([external_time, *value_columns]) - set(external.columns))
    if missing:
        raise ValueError(f"External dataframe is missing columns: {missing}")

    left = base.copy()
    right = external[[external_time, *value_columns]].copy()

    left[base_time] = pd.to_datetime(left[base_time], utc=True, errors="coerce")
    right[external_time] = pd.to_datetime(
        right[external_time],
        utc=True,
        errors="coerce",
    )

    if left[base_time].isna().any() or right[external_time].isna().any():
        raise ValueError("Cannot merge data containing invalid timestamps.")

    if left[base_time].duplicated().any():
        raise ValueError("Base dataframe contains duplicate timestamps.")

    if right[external_time].duplicated().any():
        duplicates = right.loc[
            right[external_time].duplicated(keep=False),
            external_time,
        ].unique()[:5]
        raise ValueError(
            "External dataframe contains duplicate timestamps, e.g. "
            f"{list(duplicates)}"
        )

    right = right.rename(columns={external_time: base_time})

    return left.merge(
        right,
        on=base_time,
        how="left",
        validate="one_to_one",
        sort=True,
    )


def load_etna_event_catalog_xls(
    path: str | Path,
    *,
    sheet_name=0,
    quality_filter: bool = False,
) -> pd.DataFrame:
    """Load the EtnaSC 2000–2010 catalogue and construct UTC timestamps."""
    try:
        dataframe = pd.read_excel(Path(path), sheet_name=sheet_name)
    except ImportError as exc:
        raise ImportError(
            "Reading the .xls catalogue requires the 'xlrd' package."
        ) from exc

    time_columns = ["YE", "MO", "DA", "HR", "MI", "SE"]
    missing = sorted(set(time_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing catalogue time columns: {missing}")

    for column in time_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    base_time = pd.to_datetime(
        {
            "year": dataframe["YE"],
            "month": dataframe["MO"],
            "day": dataframe["DA"],
            "hour": dataframe["HR"],
            "minute": dataframe["MI"],
        },
        utc=True,
        errors="coerce",
    )

    dataframe["timestamp"] = (
        base_time
        + pd.to_timedelta(dataframe["SE"], unit="s")
    )

    dataframe = (
        dataframe
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    numeric_columns = [
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

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    if quality_filter:
        required_quality = ["N.O.", "RMS", "GAP"]
        missing_quality = [
            column
            for column in required_quality
            if column not in dataframe.columns
        ]
        if missing_quality:
            raise ValueError(
                "Cannot quality-filter catalogue; missing columns: "
                f"{missing_quality}"
            )

        dataframe = dataframe.loc[
            (dataframe["N.O."] >= 8)
            & (dataframe["RMS"] <= 0.30)
            & (dataframe["GAP"] <= 250)
        ].copy()

    return dataframe


def catalogue_hourly_counts(
    catalogue: pd.DataFrame,
    master_index: pd.DatetimeIndex,
) -> pd.Series:
    events = catalogue.copy()

    if "timestamp" not in events.columns:
        if "time" not in events.columns:
            raise ValueError(
                "Catalogue dataframe must contain 'timestamp' or 'time'."
            )
        events = events.rename(columns={"time": "timestamp"})

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        utc=True,
        errors="coerce",
    )
    events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

    index = pd.DatetimeIndex(pd.to_datetime(master_index, utc=True))

    return (
        events
        .set_index("timestamp")
        .assign(count=1)["count"]
        .resample("1h")
        .sum()
        .reindex(index, fill_value=0)
        .astype(float)
        .rename("catalogue_count")
    )


def catalogue_event_rate_response(
    catalogue: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    *,
    response_window: int = 6,
    baseline_window: int = 24,
    min_periods: int = 12,
) -> pd.Series:
    """
    Positive short-term local-seismicity response.

    At hour t, the response uses counts from t-5 through t. Its baseline uses
    earlier six-hour response windows ending at least six hours before t.
    """
    hourly_count = catalogue_hourly_counts(catalogue, master_index)

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

    return (
        (log_recent_count - past_baseline)
        .clip(lower=0)
        .rename("local_event_rate_response")
    )


def catalogue_event_rate_state(
    catalogue: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    *,
    state_window: int = 48,
    exclusion_hours: int = 6,
    min_periods: int = 24,
) -> pd.Series:
    """
    Past local-seismicity state.

    At hour t, the state uses a 48-hour count window ending six hours before t,
    so it does not overlap the contemporaneous six-hour response window.
    """
    hourly_count = catalogue_hourly_counts(catalogue, master_index)

    past_count = (
        hourly_count
        .shift(exclusion_hours)
        .rolling(
            window=state_window,
            min_periods=min_periods,
        )
        .sum()
    )

    return np.log1p(past_count).rename("local_event_rate_state")


def _numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="coerce")


def _log1p_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    if (values.dropna() < 0).any():
        raise ValueError(
            f"Expected non-negative values in {series.name!r} before log1p."
        )

    return np.log1p(values)


def positive_log_epsilon(series: pd.Series) -> float:
    """Estimate a positive log offset on a reference interval."""
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"Cannot estimate log epsilon because {series.name!r} contains NaNs."
        )

    if (values < 0).any():
        raise ValueError(
            f"Cannot estimate log epsilon because {series.name!r} contains "
            "negative values."
        )

    positive = values[values > 0]
    if positive.empty:
        return 1e-30

    return max(float(positive.quantile(0.01)) * 0.1, 1e-30)


def safe_log_positive(
    series: pd.Series,
    eps: float | None = None,
) -> pd.Series:
    """Log-transform a non-negative physical-amplitude series."""
    values = pd.to_numeric(series, errors="coerce")

    if (values.dropna() < 0).any():
        raise ValueError(
            f"safe_log_positive received negative values in {series.name!r}."
        )

    if eps is None:
        eps = positive_log_epsilon(values)

    return np.log(values + float(eps))


def fit_cause_trigger_transform_parameters(
    reference: pd.DataFrame,
) -> dict:
    """Fit data-dependent transformations on the pre-case reference interval."""
    parameters = {"log_positive_eps": {}}

    if "teleseismic" in reference.columns:
        parameters["log_positive_eps"]["teleseismic"] = (
            positive_log_epsilon(reference["teleseismic"])
        )

    return parameters


def transform_for_cause_trigger_scaling(
    dataframe: pd.DataFrame,
    *,
    transform_parameters: dict | None = None,
) -> pd.DataFrame:
    """Apply variable-family transformations before standardization."""
    transformed = dataframe.copy()

    if transform_parameters is None:
        transform_parameters = fit_cause_trigger_transform_parameters(
            transformed
        )

    epsilons = transform_parameters.get("log_positive_eps", {})

    if "teleseismic" in transformed.columns:
        if "teleseismic" not in epsilons:
            raise ValueError(
                "Missing reference-fitted log epsilon for 'teleseismic'."
            )
        transformed["teleseismic"] = safe_log_positive(
            transformed["teleseismic"],
            eps=float(epsilons["teleseismic"]),
        )

    for column in ("rainfall_mm", "CO2_3"):
        if column in transformed.columns:
            transformed[column] = _log1p_nonnegative(
                _numeric_series(transformed, column)
            )

    if "pressure_drop" in transformed.columns:
        transformed["pressure_drop"] = np.arcsinh(
            _numeric_series(transformed, "pressure_drop")
        )

    return transformed


def standard_scale_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return StandardScaler-compatible z-scores with strict validation."""
    numeric = dataframe.apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        missing = numeric.isna().sum()
        raise ValueError(
            "Cannot standardize data containing NaNs:\n"
            f"{missing[missing > 0].sort_values(ascending=False)}"
        )

    if not np.isfinite(numeric.to_numpy()).all():
        invalid = [
            column
            for column in numeric.columns
            if not np.isfinite(numeric[column].to_numpy()).all()
        ]
        raise ValueError(
            "Cannot standardize data containing non-finite values in: "
            f"{invalid}"
        )

    constant = [
        column
        for column in numeric.columns
        if numeric[column].nunique(dropna=True) <= 1
    ]
    if constant:
        raise ValueError(f"Cannot standardize constant columns: {constant}")

    scaled = StandardScaler().fit_transform(numeric)

    return pd.DataFrame(
        scaled,
        index=numeric.index,
        columns=numeric.columns,
    )


def validate_etna_dataset(dataframe: pd.DataFrame) -> None:
    required = {
        "time",
        "teleseismic",
        "local_event_rate_state",
        "local_event_rate_response",
    }
    missing_required = sorted(required - set(dataframe.columns))
    if missing_required:
        raise ValueError(
            f"Final Etna dataset is missing required columns: {missing_required}"
        )

    time = pd.to_datetime(dataframe["time"], utc=True, errors="coerce")

    if time.isna().any():
        raise ValueError("Final Etna dataset contains invalid timestamps.")

    if time.duplicated().any():
        raise ValueError("Final Etna dataset contains duplicate timestamps.")

    if not time.is_monotonic_increasing:
        raise ValueError("Final Etna dataset is not sorted by time.")

    deltas = time.diff().dropna()
    if not deltas.eq(pd.Timedelta("1h")).all():
        raise ValueError("Final Etna dataset is not a complete hourly grid.")

    numeric = dataframe.drop(columns="time").apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        missing = numeric.isna().sum()
        raise ValueError(
            "Final Etna dataset contains missing values:\n"
            f"{missing[missing > 0].sort_values(ascending=False)}"
        )

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Final Etna dataset contains non-finite values.")

    scaled_columns = [
        column
        for column in dataframe.columns
        if column.endswith("_scaled")
    ]
    if scaled_columns:
        raise ValueError(
            "The Etna dataset must remain unstandardized. "
            f"Found scaled columns: {scaled_columns}"
        )

def fill_isolated_teleseismic_hours(
    series: pd.Series,
) -> pd.Series:
    """
    Fill only isolated one-hour gaps in the positive teleseismic proxy.

    The replacement is the geometric mean of the immediately preceding
    and following valid hourly values. Consecutive missing hours and gaps
    at the boundaries remain missing.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float).copy()

    previous = values.shift(1)
    following = values.shift(-1)

    fillable = (
        values.isna()
        & previous.notna()
        & following.notna()
        & previous.gt(0)
        & following.gt(0)
    )

    values.loc[fillable] = np.sqrt(
        previous.loc[fillable]
        * following.loc[fillable]
    )

    return values

def _waveform_frame(waveform: pd.DataFrame) -> pd.DataFrame:
    frame = waveform.copy()

    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(
            frame["time"],
            utc=True,
            errors="coerce",
        )
        frame = frame.set_index("time")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    else:
        raise ValueError(
            "Waveform dataframe must have a DatetimeIndex or a 'time' column."
        )

    if "teleseismic" not in frame.columns:
        raise ValueError("Waveform dataframe is missing 'teleseismic'.")

    frame["teleseismic"] = pd.to_numeric(
        frame["teleseismic"],
        errors="coerce",
    )

    return frame[["teleseismic"]].sort_index()


def create_etna_dataset(
    wave_df: pd.DataFrame,
    *,
    start_time,
    end_time,
    catalog_df: pd.DataFrame,
    etnagas_df: pd.DataFrame | None = None,
    etnagas_cols: list[str] | None = None,
    weather_df: pd.DataFrame | None = None,
    weather_cols: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    analysis_start = _to_utc_timestamp(start_time)
    analysis_end = _to_utc_timestamp(end_time)

    if analysis_end <= analysis_start:
        raise ValueError("end_time must be later than start_time.")

    master_index = pd.date_range(
        start=analysis_start,
        end=analysis_end,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )

    waveform = _waveform_frame(wave_df)

    waveform_hourly = (
        waveform
        .resample("1h")
        .mean()
        .reindex(master_index)
    )

    waveform_hourly["teleseismic"] = (
        fill_isolated_teleseismic_hours(
            waveform_hourly["teleseismic"]
        )
    )

    base = (
        waveform_hourly
        .rename_axis("time")
        .reset_index()
    )

    base["local_event_rate_state"] = catalogue_event_rate_state(
        catalog_df,
        master_index,
        state_window=48,
        exclusion_hours=6,
        min_periods=24,
    ).to_numpy()

    base["local_event_rate_response"] = catalogue_event_rate_response(
        catalog_df,
        master_index,
        response_window=6,
        baseline_window=24,
        min_periods=12,
    ).to_numpy()

    # Only the leading rows lacking the past-only catalogue windows are removed.
    base = base.dropna(
        subset=[
            "local_event_rate_state",
            "local_event_rate_response",
        ]
    ).reset_index(drop=True)

    if etnagas_df is not None and etnagas_cols:
        base = merge_hourly_data(
            base,
            etnagas_df,
            base_time="time",
            external_time="timestamp",
            value_columns=list(etnagas_cols),
        )

    if weather_df is not None and weather_cols:
        base = merge_hourly_data(
            base,
            weather_df,
            base_time="time",
            external_time="timestamp",
            value_columns=list(weather_cols),
        )

    ordered_columns = [
        "time",
        "teleseismic",
        "local_event_rate_state",
        "local_event_rate_response",
        *(etnagas_cols or []),
        *(weather_cols or []),
    ]
    ordered_columns = list(dict.fromkeys(ordered_columns))

    analysis = (
        base[ordered_columns]
        .sort_values("time")
        .reset_index(drop=True)
    )

    validate_etna_dataset(analysis)

    if output_dir is not None:
        output_path = Path(output_dir) / "etna_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(output_path, index=False)

    return analysis


def load_etnagas_csv(
    path: str | Path,
    value_cols: list[str],
    *,
    start_time=None,
    end_time=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path).replace("NULL", np.nan)

    if "Time" not in raw.columns:
        raise ValueError("ETNAGAS CSV is missing the 'Time' column.")

    raw["timestamp"] = pd.to_datetime(
        raw["Time"],
        utc=True,
        errors="coerce",
    )
    raw = (
        raw
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="first")
    )

    direct_columns = [
        column
        for column in value_cols
        if column != "pressure_drop"
    ]

    source_columns = list(direct_columns)
    if "pressure_drop" in value_cols:
        source_columns.append("Patm_3")

    missing_source = sorted(set(source_columns) - set(raw.columns))
    if missing_source:
        raise ValueError(
            f"ETNAGAS CSV is missing source columns: {missing_source}"
        )

    for column in source_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    if start_time is None:
        requested_start = raw["timestamp"].min().floor("h")
    else:
        requested_start = _to_utc_timestamp(start_time).floor("h")

    if end_time is None:
        requested_end = raw["timestamp"].max().ceil("h") + pd.Timedelta("1h")
    else:
        requested_end = _to_utc_timestamp(end_time).ceil("h")

    source_start = (
        requested_start - pd.Timedelta("1h")
        if "pressure_drop" in value_cols
        else requested_start
    )

    source = raw.loc[
        (raw["timestamp"] >= source_start)
        & (raw["timestamp"] < requested_end),
        ["timestamp", *source_columns],
    ].copy()

    full_index = pd.date_range(
        source_start,
        requested_end,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )

    hourly = (
        source
        .set_index("timestamp")
        .resample("1h")
        .mean()
        .reindex(full_index)
    )

    missing_before = hourly[source_columns].isna()

    hourly[source_columns] = hourly[source_columns].interpolate(
        method="time",
        limit=1,
        limit_area="inside",
    )

    report_rows = []
    for timestamp in hourly.index:
        filled = [
            column
            for column in source_columns
            if missing_before.at[timestamp, column]
            and pd.notna(hourly.at[timestamp, column])
        ]

        if filled and requested_start <= timestamp < requested_end:
            report_rows.append({
                "timestamp": timestamp,
                "interpolated_variables": ", ".join(filled),
                "method": "linear time interpolation",
                "maximum_gap": "1 hour",
            })

    remaining = hourly.loc[
        (hourly.index >= requested_start)
        & (hourly.index < requested_end),
        source_columns,
    ].isna()

    if remaining.any().any():
        missing = remaining.sum()
        raise ValueError(
            "ETNAGAS data remain incomplete after one-hour interpolation:\n"
            f"{missing[missing > 0].sort_values(ascending=False)}"
        )

    if "pressure_drop" in value_cols:
        hourly["pressure_drop"] = -hourly["Patm_3"].diff()

    output = (
        hourly
        .loc[
            (hourly.index >= requested_start)
            & (hourly.index < requested_end)
        ]
        .rename_axis("timestamp")
        .reset_index()
    )

    final_columns = ["timestamp", *value_cols]
    report = pd.DataFrame(
        report_rows,
        columns=[
            "timestamp",
            "interpolated_variables",
            "method",
            "maximum_gap",
        ],
    )

    return output[final_columns], report


def load_openmeteo_etna_weather(
    start_date: str,
    end_date: str,
    *,
    latitude: float = 37.75,
    longitude: float = 14.99,
) -> pd.DataFrame:
    """Download hourly Open-Meteo precipitation for the Etna proxy point."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation",
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(
            data["hourly"]["time"],
            utc=True,
        ),
        "rainfall_mm": pd.to_numeric(
            data["hourly"]["precipitation"],
            errors="coerce",
        ),
    }).sort_values("timestamp")

    if weather["timestamp"].duplicated().any():
        raise ValueError("Open-Meteo returned duplicate hourly timestamps.")

    if weather["rainfall_mm"].isna().any():
        missing = int(weather["rainfall_mm"].isna().sum())
        raise ValueError(
            f"Open-Meteo precipitation contains {missing} missing values."
        )

    return weather[["timestamp", "rainfall_mm"]]
