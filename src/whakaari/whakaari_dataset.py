from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def past_rolling_median_state(
    s: pd.Series,
    *,
    window: int = 6,
    min_periods: int = 3,
) -> pd.Series:
    """
    Convert an immediate hourly variable into a past-only state proxy.

    The value at time t depends only on observations before t:
        rolling median over previous values, shifted by one sample.
    """
    return (
        pd.to_numeric(s, errors="coerce")
        .rolling(window=window, min_periods=min_periods)
        .median()
        .shift(1)
    )

def safe_log_positive(s: pd.Series, eps: float | None = None) -> pd.Series:
    """
    Log-transform strictly non-negative physical amplitudes/ratios.

    This is better than log1p for seismic RMS values because RMS velocities
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

def positive_past_log_anomaly(
    s: pd.Series,
    *,
    baseline_window: int = 24,
    min_periods: int = 12,
) -> pd.Series:
    """
    Convert a positive amplitude series into a positive past-baseline anomaly.

    The value at time t is:
        max(log(x_t + eps) - median(log(x_{t-24:t-1} + eps)), 0)

    This is used for effect proxies where the target should represent
    a positive response anomaly rather than raw absolute amplitude.
    """
    log_s = safe_log_positive(s)

    past_baseline = (
        log_s
        .rolling(window=baseline_window, min_periods=min_periods)
        .median()
        .shift(1)
    )

    return (log_s - past_baseline).clip(lower=0)

def transform_for_cause_trigger_scaling(final: pd.DataFrame) -> pd.DataFrame:
    """
    Apply neutral, variable-family transformations before StandardScaler.

    This makes heterogeneous
    observables numerically comparable for HMML/PCMCI and the Cause--Trigger
    split/F-test.
    """
    transformed = final.copy()

    # Positive amplitude / ratio variables.
    # Use log, not log1p, because waveform RMS values are often << 1.
    log_positive_cols = [
        "hydro_2_5",
        "ratio_4p5_8_over_8_16",
    ]

    # Positive count / accumulation / flux variables.
    log1p_cols = [
        "event_rate_2_5",
        "SO2_flux",
        "API",
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

    Raises an error for NaNs, infinite values, or constant columns instead of
    silently producing unusable causal-model input.
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        missing = numeric.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        raise ValueError(
            "Cannot standardize because transformed data contain NaNs:\n"
            f"{missing}"
        )

    if not np.isfinite(numeric.to_numpy()).all():
        bad_cols = [
            col for col in numeric.columns
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

    # Replace the immediate spectral ratio with a past-only smoothed
    # spectral-state proxy. 
    if "ratio_4p5_8_over_8_16" in wave_1h.columns:
        wave_1h["ratio_4p5_8_over_8_16"] = past_rolling_median_state(
            wave_1h["ratio_4p5_8_over_8_16"],
            window=6,
            min_periods=3,
        )
    weather_1h = weather_vars.resample(master_freq).mean()
    so2_1h = so2.resample(master_freq).mean()
    gnss_1h = (
        gnss
        .resample("1D")
        .mean()
        .resample(master_freq)
        .ffill()
    )

    # Replace raw 5--15 Hz RMS with a positive tremor-response anomaly.
    # positive values represent eruption-response
    # excess above the recent past baseline.
    if "effect_tremor_5_15" in wave_1h.columns:
        wave_1h["effect_tremor_5_15"] = positive_past_log_anomaly(
            wave_1h["effect_tremor_5_15"],
            baseline_window=24,
            min_periods=12,
        )

    wave_1h = wave_1h.reindex(master_index)
    weather_1h = weather_1h.reindex(master_index)
    # Carry the last pre-window SO2 observation into the modelling window.
    so2_1h = (
        so2_1h
        .reindex(so2_1h.index.union(master_index))
        .sort_index()
        .ffill()
        .reindex(master_index)
    )
    gnss_1h = (
        gnss_1h
        .reindex(master_index)
        .ffill()
    )

    frames = [wave_1h, weather_1h, so2_1h, gnss_1h]

    whakaari_master = pd.concat(frames, axis=1)
    whakaari_master.index.name = "timestamp"
    return whakaari_master


def prepare_raw_analysis_dataframe(whakaari_master):
    whakaari_raw = whakaari_master.copy()

    if "SO2_flux" in whakaari_raw.columns:
        # SO2 is sparse and slow-changing.
        # Prepare it as a past-only step function:
        # each observed value is carried forward until the next observation.
        whakaari_raw["SO2_flux"] = (
            pd.to_numeric(whakaari_raw["SO2_flux"], errors="coerce")
            .clip(lower=0)
            .ffill()
        )

        # Remove only the part before the first real SO2 observation.
        whakaari_raw = whakaari_raw.dropna(subset=["SO2_flux"])

    if "API" in whakaari_raw.columns:
        whakaari_raw["API"] = (
            pd.to_numeric(whakaari_raw["API"], errors="coerce")
            .ffill(limit=1)
        )

    if "pressure_drop" in whakaari_raw.columns:
        whakaari_raw["pressure_drop"] = (
            pd.to_numeric(whakaari_raw["pressure_drop"], errors="coerce")
            .ffill(limit=1)
        )

    if "GNSS_deformation" in whakaari_raw.columns:
        whakaari_raw["GNSS_deformation"] = (
            pd.to_numeric(whakaari_raw["GNSS_deformation"], errors="coerce")
            .ffill()
        )

    required_waveform_cols = [
        "hydro_2_5",
        "ratio_4p5_8_over_8_16",
        "event_rate_2_5",
        "effect_tremor_5_15",
    ]

    required_existing = [
        c for c in required_waveform_cols
        if c in whakaari_raw.columns
    ]

    whakaari_raw = whakaari_raw.dropna(subset=required_existing)

    return whakaari_raw

def preprocessing_report(rawest_df, prepared_df):
    """
    Notebook-only preprocessing audit.

    Compares:
    - rawest_df: output of build_master_dataframe()
    - prepared_df: output of prepare_raw_analysis_dataframe()

    It reports:
    - rows before/after
    - dropped timestamps
    - missing values before/after
    - how many NaNs were filled
    - how many existing values changed
    - how many non-missing values were lost because rows were dropped
    """

    rawest = rawest_df.copy()
    prepared = prepared_df.copy()

    if "timestamp" in rawest.columns:
        rawest = rawest.set_index("timestamp")
    if "timestamp" in prepared.columns:
        prepared = prepared.set_index("timestamp")

    rawest.index = pd.to_datetime(rawest.index, utc=True)
    prepared.index = pd.to_datetime(prepared.index, utc=True)

    common_index = rawest.index.intersection(prepared.index)
    dropped_timestamps = rawest.index.difference(prepared.index)

    rows = []

    all_cols = sorted(set(rawest.columns).union(set(prepared.columns)))

    for col in all_cols:
        if col not in rawest.columns:
            rows.append({
                "variable": col,
                "status": "added",
                "missing_before": np.nan,
                "missing_after": prepared[col].isna().sum(),
                "filled_from_nan": np.nan,
                "changed_existing": np.nan,
                "lost_by_dropped_rows": np.nan,
            })
            continue

        if col not in prepared.columns:
            rows.append({
                "variable": col,
                "status": "removed",
                "missing_before": rawest[col].isna().sum(),
                "missing_after": np.nan,
                "filled_from_nan": np.nan,
                "changed_existing": np.nan,
                "lost_by_dropped_rows": rawest[col].notna().sum(),
            })
            continue

        before = rawest.loc[common_index, col]
        after = prepared.loc[common_index, col]

        before_num = pd.to_numeric(before, errors="coerce")
        after_num = pd.to_numeric(after, errors="coerce")

        filled_from_nan = int((before.isna() & after.notna()).sum())

        both_present = before.notna() & after.notna()

        changed_existing = int((
            both_present
            & ~np.isclose(
                before_num,
                after_num,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        ).sum())

        lost_by_dropped_rows = int(rawest.loc[dropped_timestamps, col].notna().sum())

        rows.append({
            "variable": col,
            "status": "kept",
            "missing_before": int(rawest[col].isna().sum()),
            "missing_after": int(prepared[col].isna().sum()),
            "filled_from_nan": filled_from_nan,
            "changed_existing": changed_existing,
            "lost_by_dropped_rows": lost_by_dropped_rows,
        })

    report = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Rows before preprocessing: {len(rawest)}")
    print(f"Rows after preprocessing:  {len(prepared)}")
    print(f"Rows dropped:              {len(dropped_timestamps)}")

    if len(dropped_timestamps) > 0:
        print("\nDropped timestamps:")
        for ts in dropped_timestamps:
            print(f"  {ts}")

    print("\nVariables with preprocessing changes:")
    changed = report[
        (report["filled_from_nan"].fillna(0) > 0)
        | (report["changed_existing"].fillna(0) > 0)
        | (report["lost_by_dropped_rows"].fillna(0) > 0)
        | (report["status"] != "kept")
    ]

    if len(changed) == 0:
        print("  None")
    else:
        try:
            from IPython.display import display
            display(changed)
        except Exception:
            print(changed.to_string(index=False))

    return report

def build_final_causal_dataframe(
    whakaari_raw: pd.DataFrame,
    include_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
):
    """
    Prepare the standardized Whakaari dataset for causal analysis.
    """

    analysis_input = whakaari_raw.copy()

    if analysis_input.index.has_duplicates:
        duplicates = analysis_input.index[analysis_input.index.duplicated()].unique()[:5]
        raise ValueError(
            f"Whakaari causal dataframe has duplicate timestamps, e.g. {list(duplicates)}"
        )

    if not analysis_input.index.is_monotonic_increasing:
        analysis_input = analysis_input.sort_index()

    if include_columns is not None:
        missing = sorted(set(include_columns) - set(analysis_input.columns))
        if missing:
            raise ValueError(f"Requested columns not found: {missing}")

        feature_cols = list(include_columns)

    else:
        drop_columns = [] if drop_columns is None else list(drop_columns)

        feature_cols = [
            c for c in analysis_input.columns
            if pd.api.types.is_numeric_dtype(analysis_input[c])
            and c not in drop_columns
        ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric columns found for Whakaari causal dataframe.")

    non_numeric_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(analysis_input[c])
    ]

    if non_numeric_cols:
        raise ValueError(
            f"Selected columns must be numeric. Non-numeric columns: {non_numeric_cols}"
        )

    model_input = analysis_input[feature_cols].copy()

    if model_input.isna().any().any():
        missing = model_input.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        raise ValueError(
            "Selected Whakaari model columns still contain NaNs:\n"
            f"{missing}"
        )

    transformed = transform_for_cause_trigger_scaling(model_input)

    scaled_numeric = standard_scale_dataframe(transformed)

    whakaari_final = pd.DataFrame(index=analysis_input.index)

    for col in feature_cols:
        whakaari_final[col + "_scaled"] = scaled_numeric[col]

    return whakaari_final

def save_whakaari_datasets(
    whakaari_raw,
    whakaari_final,
    output_dir,
):
    """
    Save the two final Whakaari datasets.

    whakaari_raw:
        Filled, analysis-ready dataframe in physical units.

    whakaari_final:
        Transformed and mean-zero/unit-variance standardized dataframe
        for the causal algorithm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    whakaari_raw.to_csv(output_dir / "whakaari_raw.csv", index_label="timestamp")
    whakaari_final.to_csv(output_dir / "whakaari_final.csv", index_label="timestamp")