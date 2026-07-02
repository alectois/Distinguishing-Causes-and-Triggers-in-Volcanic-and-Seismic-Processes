import xlrd
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from sklearn.preprocessing import StandardScaler

def extract_plume_co2so2_xls(xls_path, sheet_index=2, time_col=1, ratio_col=10):
    wb = xlrd.open_workbook(xls_path)
    sh = wb.sheet_by_index(sheet_index)

    first_data_row = None
    for r in range(sh.nrows):
        v = sh.cell_value(r, time_col)
        # Check type BEFORE comparison
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1000:
            first_data_row = r
            break
    if first_data_row is None:
        raise RuntimeError("Could not find numeric Excel time serials in the time column.")

    times = []
    ratios = []
    datemode = wb.datemode

    for r in range(first_data_row, sh.nrows):
        t_val = sh.cell_value(r, time_col)
        y_val = sh.cell_value(r, ratio_col)

        # Type guard with explicit conversion
        if not isinstance(t_val, (int, float)) or isinstance(t_val, bool):
            continue
        
        # Explicitly cast to float to satisfy type checker
        t_val = float(t_val)
        
        if t_val <= 0:
            continue

        try:
            t_dt = xlrd.xldate_as_datetime(t_val, datemode)
        except Exception:
            continue

        try:
            y = float(y_val)
        except Exception:
            y = np.nan

        times.append(t_dt)
        ratios.append(y)

    g = pd.DataFrame({
        "timestamp": pd.to_datetime(times, utc=True),
        "CO2_SO2": pd.to_numeric(ratios, errors="coerce")
    })

    g = g.sort_values("timestamp").dropna(subset=["CO2_SO2"]).reset_index(drop=True)
    return g

def merge_data(base, ext, base_time, ext_time, value_cols, tolerance_hours):
    base = base.copy()
    ext = ext.copy()

    base[base_time] = pd.to_datetime(base[base_time], utc=True, errors="coerce")
    ext[ext_time] = pd.to_datetime(ext[ext_time], utc=True, errors="coerce")

    base = base.dropna(subset=[base_time]).sort_values(base_time)
    ext = ext.dropna(subset=[ext_time]).sort_values(ext_time)

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

    # Positive waveform-amplitude variables.
    # Use log, not log1p, because RMS velocities can be much smaller than 1.
    log_positive_cols = [
        "teleseismic",
        "background_seismic",
        "effect_seismic",
    ]

    # Positive accumulation / gas variables.
    log1p_cols = [
        "API",
        "CO2_3",
        "CO2_SO2",
    ]

    # Signed burst-like variables.
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

def create_etna_final_dataset(
    wave_df: pd.DataFrame,
    station_name: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    etnagas_df: pd.DataFrame | None = None,
    etnagas_cols: list[str] | None = None,
    etnagas_buffer_hours: int = 6,
    etnagas_tolerance_hours: int = 6,
    plume_df: pd.DataFrame | None = None,
    plume_buffer_hours: int = 12,
    plume_tolerance_hours: int = 6,
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
        "background_seismic",
        "effect_seismic",
    ]

    for c in seismic_cols:
        if c not in df.columns:
            raise KeyError(f"Missing '{c}' in waveform dataframe.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if start_time is not None:
        df = df[df["time"] >= pd.to_datetime(start_time, utc=True)]
    if end_time is not None:
        df = df[df["time"] < pd.to_datetime(end_time, utc=True)]

    base_cols = [
        "time",
        *seismic_cols,
    ]

    base = df[base_cols].copy()

    base["station"] = station_name
    base = base[["station", *base_cols]]

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

    # ---- plume ----
    if plume_df is not None:
        gpl = plume_df.copy()

        tmin = base["time"].iloc[0] - pd.Timedelta(hours=plume_buffer_hours)
        tmax = base["time"].iloc[-1] + pd.Timedelta(hours=plume_buffer_hours)
        gpl = gpl[(gpl["timestamp"] >= tmin) & (gpl["timestamp"] <= tmax)].copy()

        base = merge_data(
            base=base,
            ext=gpl,
            base_time="time",
            ext_time="timestamp",
            value_cols=["CO2_SO2"],
            tolerance_hours=plume_tolerance_hours,
        )

    # final sorted dataframe
    final = base.sort_values("time").reset_index(drop=True)

    # ---- final transformation + StandardScaler ----
    # Raw variables remain unchanged in final_raw.
    # The scaled dataset is built from a transformed copy.

    exclude_from_scaling = ["station", "time"]

    scale_input = transform_for_cause_trigger_scaling(final)

    scale_cols = [
        c for c in scale_input.columns
        if c not in exclude_from_scaling
    ]

    scale_input_numeric = scale_input[scale_cols].copy()
    scaled_numeric = standard_scale_dataframe(scale_input_numeric)

    final_scaled = final[["time"]].copy()

    for col in scale_cols:
        final_scaled[col + "_scaled"] = scaled_numeric[col]

    if final.isna().any().any():
        print("Warning: final dataset contains missing values.")
        print(final.isna().mean().sort_values())

    final_raw = final.copy()

    unexpected_raw_suffix_cols = [
        c for c in final_raw.columns
        if c.endswith("_raw")
    ]

    if unexpected_raw_suffix_cols:
        raise ValueError(
            "etna_raw should not contain *_raw columns after the Etna-Whakaari "
            f"schema cleanup. Found: {unexpected_raw_suffix_cols}"
        )

    non_scaled_model_cols = [
        c for c in final_scaled.columns
        if c != "time" and not c.endswith("_scaled")
    ]

    if non_scaled_model_cols:
        raise ValueError(
            "etna_final should contain only 'time' and *_scaled variables. "
            f"Found unexpected columns: {non_scaled_model_cols}"
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = output_dir / "etna_raw.csv"
        final_path = output_dir / "etna_final.csv"

        final_raw.to_csv(raw_path, index=False)
        final_scaled.to_csv(final_path, index=False)

        print("Saved:", raw_path)
        print("Saved:", final_path)

    return final_raw, final_scaled

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
        .reset_index(drop=True)
    )

    for col in source_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pressure_drop" in value_cols:
        if "Patm_3" not in df.columns:
            raise KeyError("Cannot compute pressure_drop because 'Patm_3' is missing.")
        df["pressure_change"] = df["Patm_3"].diff()
        df["pressure_drop"] = -df["pressure_change"]
        df["pressure_drop"] = df["pressure_drop"].fillna(0)

    final_cols = ["timestamp"] + [c for c in value_cols if c in df.columns]
    return df[final_cols]


def load_openmeteo_etna_weather(
    start_date: str,
    end_date: str,
    latitude: float = 37.75,
    longitude: float = 14.99,
    api_alpha: float = 0.05,
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

    weather["API"] = (
        weather["rainfall_mm"]
        .fillna(0)
        .ewm(alpha=api_alpha, adjust=False)
        .mean()
    )

    return weather[["timestamp", "API"]]