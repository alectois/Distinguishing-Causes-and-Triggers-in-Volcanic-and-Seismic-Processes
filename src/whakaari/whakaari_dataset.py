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

    weather_1h = (
        weather_vars
        .resample(master_freq)
        .interpolate(limit_direction="both")
    )

    so2_1h = so2.resample(master_freq).mean()

    gnss_1h = (
        gnss
        .resample("1D")
        .mean()
        .resample(master_freq)
        .ffill()
    )

    wave_1h = wave_1h.reindex(master_index)
    weather_1h = weather_1h.reindex(master_index)
    so2_1h = so2_1h.reindex(master_index)

    gnss_1h = (
        gnss_1h
        .reindex(master_index)
        .ffill()
        .bfill()
    )

    frames = [wave_1h, weather_1h, so2_1h, gnss_1h]

    if local_eq is not None:
        local_eq_1h = local_eq.resample(master_freq).mean()
        local_eq_1h = local_eq_1h.reindex(master_index)
        local_eq_1h = local_eq_1h.fillna(0)
        frames.append(local_eq_1h)

    whakaari_master = pd.concat(frames, axis=1)
    whakaari_master.index.name = "timestamp"
    return whakaari_master


def prepare_raw_analysis_dataframe(whakaari_master):
    whakaari_raw = whakaari_master.copy()

    if "SO2_flux" in whakaari_raw.columns:
        # Fill only short SO₂ gaps; long gaps remain NaN.
        whakaari_raw["SO2_flux"] = whakaari_raw["SO2_flux"].interpolate(limit=3)

    if "API" in whakaari_raw.columns:
        whakaari_raw["API"] = whakaari_raw["API"].interpolate()

    if "pressure_drop" in whakaari_raw.columns:
        whakaari_raw["pressure_drop"] = whakaari_raw["pressure_drop"].fillna(0)

    if "event_rate_2_5" in whakaari_raw.columns:
        whakaari_raw["event_rate_2_5"] = whakaari_raw["event_rate_2_5"].fillna(0)

    if "GNSS_deformation" in whakaari_raw.columns:
        whakaari_raw["GNSS_deformation"] = whakaari_raw["GNSS_deformation"].ffill()

    if "local_eq_count_1h" in whakaari_raw.columns:
        whakaari_raw["local_eq_count_1h"] = whakaari_raw["local_eq_count_1h"].fillna(0)

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
    whakaari_raw,
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
            "hydro_2_5",
            "event_rate_2_5",
            "API",
            "effect_tremor_5_15",
            "local_eq_count_1h",
        ]

    analysis_prepped = whakaari_raw.copy()

    # Log-transform only non-negative skewed variables.
    # Do not include pressure_drop or GNSS_deformation because they are signed.
    for col in log_transform_cols:
        if col in analysis_prepped.columns:
            analysis_prepped[col] = np.log1p(
                pd.to_numeric(analysis_prepped[col], errors="coerce").clip(lower=0)
            )

    feature_cols = analysis_prepped.columns.tolist()

    whakaari_final = pd.DataFrame(index=analysis_prepped.index)
    for col in feature_cols:
        whakaari_final[col + "_scaled"] = robust_scale_series(analysis_prepped[col])

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
        Log-transformed and robust-scaled dataframe for the causal algorithm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    whakaari_raw.to_csv(output_dir / "whakaari_raw.csv")
    whakaari_final.to_csv(output_dir / "whakaari_final.csv")