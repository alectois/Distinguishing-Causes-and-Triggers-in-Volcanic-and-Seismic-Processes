import xlrd
import pandas as pd
import numpy as np
from pathlib import Path

def extract_plume_co2so2_xls(xls_path, sheet_index=2, time_col=1, ratio_col=10):
    wb = xlrd.open_workbook(xls_path)
    sh = wb.sheet_by_index(sheet_index)

    first_data_row = None
    for r in range(sh.nrows):
        v = sh.cell_value(r, time_col)
        if isinstance(v, (int, float)) and v > 1000:
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

        if not isinstance(t_val, (int, float)) or t_val <= 0:
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

def create_etna_final_dataset(
    wave_df: pd.DataFrame,
    station_name: str,
    out_csv: str,
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
    save_pickle: bool = True,
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

    for c in ["S_log", "T_log", "Y_log"]:
        if c not in df.columns:
            raise KeyError(f"Missing '{c}' in waveform dataframe.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if start_time is not None:
        df = df[df["time"] >= pd.to_datetime(start_time, utc=True)]
    if end_time is not None:
        df = df[df["time"] < pd.to_datetime(end_time, utc=True)]

    # waveform scaling
    df["S_log_scaled"] = robust_scale_series(df["S_log"])
    df["T_log_scaled"] = robust_scale_series(df["T_log"])
    df["Y_log_scaled"] = robust_scale_series(df["Y_log"])

    # keep BOTH raw and scaled waveform variables
    base = df[[
        "time",
        "S_log", "T_log", "Y_log",
        "S_log_scaled", "T_log_scaled", "Y_log_scaled"
    ]].copy()

    base["station"] = station_name
    base = base[[
        "station", "time",
        "S_log", "T_log", "Y_log",
        "S_log_scaled", "T_log_scaled", "Y_log_scaled"
    ]]

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

        for c in etnagas_cols:
            if c in base.columns:
                base[c + "_scaled"] = robust_scale_series(base[c])

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

        if "CO2_SO2" in base.columns:
            base["CO2_SO2_scaled"] = robust_scale_series(base["CO2_SO2"])

    # final sorted dataframe
    final = base.sort_values("time").reset_index(drop=True)

    if final.isna().any().any():
        print("Warning: final dataset contains missing values.")
        print(final.isna().mean().sort_values())

    # file stems
    stem = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    Path(stem).parent.mkdir(parents=True, exist_ok=True)

    # optional full checkpoint
    if save_pickle:
        final.to_pickle(f"{stem}_full.pkl")

    # save RAW dataset
    raw_cols = [c for c in final.columns if not c.endswith("_scaled")]
    final_raw = final[raw_cols].copy()
    final_raw.to_csv(f"{stem}_raw.csv", index=False)
    if save_pickle:
        final_raw.to_pickle(f"{stem}_raw.pkl")

    # save SCALED dataset
    scaled_cols = ["time", "station"] + [c for c in final.columns if c.endswith("_scaled")]
    final_scaled = final[scaled_cols].copy()
    final_scaled.to_csv(f"{stem}_scaled.csv", index=False)
    if save_pickle:
        final_scaled.to_pickle(f"{stem}_scaled.pkl")

    print("Saved:", f"{stem}_raw.csv")
    print("Saved:", f"{stem}_scaled.csv")
    if save_pickle:
        print("Saved:", f"{stem}_full.pkl")
        print("Saved:", f"{stem}_raw.pkl")
        print("Saved:", f"{stem}_scaled.pkl")

    return final_raw, final_scaled

def load_etnagas_csv(path, value_cols):
    df = pd.read_csv(path).replace("NULL", np.nan)
    df["timestamp"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    df = (
        df[["timestamp"] + value_cols]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df