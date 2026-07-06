import numpy as np
import pandas as pd
from pathlib import Path
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def _interpolate_only_short_gaps(x, max_gap_samples):
    """
    Interpolate only NaN gaps shorter than or equal to max_gap_samples.
    Longer gaps remain NaN.
    """
    x = np.asarray(x, dtype=float)
    isnan = np.isnan(x)

    if not isnan.any():
        return x

    idx = np.arange(len(x))
    valid = ~isnan

    if not valid.any():
        return x

    nan_groups = []
    in_gap = False
    start = None

    for i, flag in enumerate(isnan):
        if flag and not in_gap:
            start = i
            in_gap = True
        elif not flag and in_gap:
            nan_groups.append((start, i - 1))
            in_gap = False

    if in_gap:
        nan_groups.append((start, len(x) - 1))

    for g0, g1 in nan_groups:
        gap_len = g1 - g0 + 1

        if gap_len <= max_gap_samples:
            left_ok = g0 > 0 and not np.isnan(x[g0 - 1])
            right_ok = g1 < len(x) - 1 and not np.isnan(x[g1 + 1])

            if left_ok and right_ok:
                x[g0:g1 + 1] = np.interp(
                    idx[g0:g1 + 1],
                    idx[valid],
                    x[valid],
                )

    return x

def _count_nan_groups(x):
    isnan = np.isnan(x)
    groups = []
    in_gap = False
    start = None

    for i, flag in enumerate(isnan):
        if flag and not in_gap:
            start = i
            in_gap = True
        elif not flag and in_gap:
            groups.append((start, i - 1))
            in_gap = False

    if in_gap:
        groups.append((start, len(x) - 1))

    return groups

def get_day_trace(client, day_start, cfg):
    pad = cfg.get("pad_sec", 0)

    day_start_utc = UTCDateTime(day_start)
    t1 = day_start_utc - pad
    t2 = day_start_utc + 24 * 3600 + pad

    st = client.get_waveforms(
        cfg["network"],
        cfg["station"],
        cfg["location"],
        cfg["channel"],
        t1,
        t2,
        attach_response=True,
    )

    st = st.copy()
    st.sort()

    if len(st) == 0:
        raise ValueError(f"Empty waveform stream for {day_start}")

    # Conservative merge: do not automatically interpolate all gaps.
    st.merge(method=0, fill_value=None)

    if len(st) != 1:
        raise ValueError(f"Expected 1 merged trace for {day_start}, got {len(st)}")

    tr = st[0]

    # Convert possible masked-array gaps to NaN.
    data = np.ma.masked_invalid(np.asarray(tr.data, dtype=float))
    x = data.filled(np.nan)

    sr = tr.stats.sampling_rate
    max_gap_sec = cfg.get("max_interp_gap_sec", 2.0)
    max_gap_samples = int(max_gap_sec * sr)

    gap_groups_before = _count_nan_groups(x)

    x = _interpolate_only_short_gaps(x, max_gap_samples)

    gap_groups_after = _count_nan_groups(x)

    tr.stats.processing.append(
        f"gap_policy: short_gap_limit={max_gap_sec}s, "
        f"gaps_before={len(gap_groups_before)}, "
        f"gaps_after={len(gap_groups_after)}"
    )

    tr.data = x.astype(np.float64)

    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.02)

    tr.remove_response(
        output=cfg.get("response_output", "VEL"),
        pre_filt=cfg.get("pre_filt", (0.5, 1.0, 20.0, 25.0)),
    )

    return tr


# helper: RMS in a band. this will be used for: 2–5 Hz RMS, 4.5–8 Hz RMS, 8–16 Hz RMS. 
# using 10-minute windows (600 seconds) -> each RMS value summarizes 10 minutes of seismic energy

def band_rms_series(trace, fmin, fmax, win_sec=600):
    tr = trace.copy()
    tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    sr = tr.stats.sampling_rate
    nwin = int(win_sec * sr)
    x = tr.data.astype(float)

    times, vals = [], []
    for i in range(0, len(x) - nwin + 1, nwin):
        seg = x[i:i+nwin]
        rms = np.sqrt(np.mean(seg**2))
        times.append(tr.stats.starttime + i / sr)
        vals.append(rms)

    if len(vals) == 0:
        return pd.Series(dtype=float)

    return pd.Series(
        vals,
        index=pd.to_datetime([t.datetime for t in times], utc=True),
    )

# Hydrothermal tremor RMS 2–5 Hz
# direct hydrothermal-state variable
def hydro_rms_2_5(trace, win_sec=600):
    s = band_rms_series(trace, 2.0, 5.0, win_sec=win_sec)
    s.name = "hydro_2_5"
    return s

# Spectral ratio 4.5–8 / 8–16 Hz
# Whakaari-specific precursor/state variable
def spectral_ratio_4p5_8_over_8_16(trace, win_sec=600):
    low = band_rms_series(trace, 4.5, 8.0, win_sec=win_sec)
    high = band_rms_series(trace, 8.0, 16.0, win_sec=win_sec)

    ratio = low / high.replace(0, np.nan)
    ratio.name = "ratio_4p5_8_over_8_16"
    return ratio

# Continuous effect variable: eruption/tremor response energy
# Uses 5–15 Hz to avoid directly containing hydro_2_5 as a sub-band.
def effect_tremor_rms_5_15(trace, win_sec=600):
    s = band_rms_series(trace, 5.0, 15.0, win_sec=win_sec)
    s.name = "effect_tremor_5_15"
    return s

# HF event rate in 2–5 Hz only
def event_rate_2_5(
    trace,
    sta_sec=1.0,
    lta_sec=5.0,
    on_thres=3.0,
    off_thres=1.5,
    out_freq="1h",
):
    tr = trace.copy()
    tr.filter("bandpass", freqmin=2.0, freqmax=5.0, corners=4, zerophase=True)

    x = tr.data.astype(float)
    sr = tr.stats.sampling_rate

    start = pd.to_datetime(tr.stats.starttime.datetime, utc=True).floor(out_freq)
    end = pd.to_datetime(tr.stats.endtime.datetime, utc=True).ceil(out_freq)

    full_index = pd.date_range(
        start=start,
        end=end - pd.Timedelta(out_freq),
        freq=out_freq,
        tz="UTC",
    )

    # Do not run STA/LTA on traces containing NaNs from long gaps.
    # But return NaN, not an empty series, so missing data is not later
    # confused with zero detected events.
    if np.isnan(x).any():
        return pd.Series(np.nan, index=full_index, name="event_rate_2_5")

    cft = classic_sta_lta(
        x,
        int(sta_sec * sr),
        int(lta_sec * sr),
    )

    on_off = trigger_onset(cft, on_thres, off_thres)

    if len(on_off) == 0:
        return pd.Series(0, index=full_index, name="event_rate_2_5")

    event_times = pd.to_datetime(
        [tr.stats.starttime.datetime + pd.Timedelta(seconds=on / sr) for on, _ in on_off],
        utc=True,
    )

    s = (
        pd.Series(1, index=event_times)
        .resample(out_freq)
        .sum()
        .reindex(full_index, fill_value=0)
    )

    s.name = "event_rate_2_5"
    return s


# Extract all features for one day
def extract_features_for_day(client, day_start, cfg):
    tr = get_day_trace(client, day_start, cfg)

    win_sec = cfg.get("rms_window_sec", 600)
    out_freq = cfg.get("master_freq", "1h")

    hydro = hydro_rms_2_5(tr, win_sec=win_sec)
    ratio = spectral_ratio_4p5_8_over_8_16(tr, win_sec=win_sec)
    hf_rate = event_rate_2_5(tr, out_freq=out_freq)
    effect = effect_tremor_rms_5_15(tr, win_sec=win_sec)

    hydro_h = hydro.resample(out_freq).mean()
    ratio_h = ratio.resample(out_freq).mean()
    effect_h = effect.resample(out_freq).mean()

    df = pd.concat([hydro_h, ratio_h, hf_rate, effect_h], axis=1)

    left = pd.Timestamp(UTCDateTime(day_start).datetime, tz="UTC")
    right = left + pd.Timedelta(days=1)

    df = df.loc[(df.index >= left) & (df.index < right)].copy()

    return df


def build_waveform_dataset(
    client,
    start,
    end,
    cfg,
    save_path=None,
    overwrite=False,
):
    if save_path is not None:
        save_path = Path(save_path)

        if save_path.exists() and not overwrite:
            cached = pd.read_pickle(save_path)
            print(f"loaded cached waveform dataset: {save_path}")

            if isinstance(cached, dict):
                return cached["waveform_df"], cached.get("failures", [])

            # Backward-compatible fallback if only the dataframe was saved.
            return cached, []

    days = pd.date_range(start, end, freq="D")

    all_days = []
    failures = []

    for day in days:
        try:
            df_day = extract_features_for_day(client, day, cfg)
            all_days.append(df_day)
            print("done:", day.date())
        except Exception as e:
            failures.append((day.date(), str(e)))
            print("failed:", day.date(), e)

    if len(all_days) == 0:
        raise RuntimeError("No waveform days were successfully processed.")

    waveform_df = pd.concat(all_days).sort_index()
    waveform_df = waveform_df[~waveform_df.index.duplicated(keep="first")]

    waveform_df.index = pd.to_datetime(waveform_df.index, utc=True)
    waveform_df = waveform_df.sort_index()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pd.to_pickle(
            {
                "waveform_df": waveform_df,
                "failures": failures,
                "start": start,
                "end": end,
                "cfg": cfg,
            },
            save_path,
        )

        print(f"saved waveform dataset: {save_path}")

    return waveform_df, failures