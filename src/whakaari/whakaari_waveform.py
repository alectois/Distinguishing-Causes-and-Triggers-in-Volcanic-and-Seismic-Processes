import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def get_day_trace(client, day_start, cfg):
    t1 = UTCDateTime(day_start)
    t2 = t1 + 24 * 3600

    st = client.get_waveforms(
        cfg["network"],
        cfg["station"],
        cfg["location"],
        cfg["channel"],
        t1,
        t2,
        attach_response=True,
    )

    st.merge(fill_value="interpolate")

    tr = st[0]
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.02)
    tr.remove_response(output="VEL")

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
    s.name = "hydro_rms_2_5"
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
# Uses 5–15 Hz to avoid directly containing hydro_rms_2_5 as a sub-band.
def effect_tremor_rms_5_15(trace, win_sec=600):
    s = band_rms_series(trace, 5.0, 15.0, win_sec=win_sec)
    s.name = "effect_tremor_rms_5_15"
    return s

# HF event rate in 2–5 Hz only
def hf_event_rate_2_5(trace,
                      sta_sec=1.0,
                      lta_sec=5.0,
                      on_thres=3.0,
                      off_thres=1.5,
                      out_freq="1h"):
    tr = trace.copy()
    tr.filter("bandpass", freqmin=2.0, freqmax=5.0, corners=4, zerophase=True)

    sr = tr.stats.sampling_rate
    cft = classic_sta_lta(
        tr.data.astype(float),
        int(sta_sec * sr),
        int(lta_sec * sr)
    )

    on_off = trigger_onset(cft, on_thres, off_thres)

    event_times = []
    for on, off in on_off:
        event_times.append(tr.stats.starttime + on / sr)

    if len(event_times) == 0:
        idx = pd.date_range(
            pd.to_datetime(tr.stats.starttime.datetime, utc=True).floor(out_freq),
            pd.to_datetime(tr.stats.endtime.datetime, utc=True).floor(out_freq),
            freq=out_freq,
        )
        return pd.Series(0, index=idx, name="hf_event_rate_2_5")

    event_times = pd.to_datetime([t.datetime for t in event_times], utc=True)

    s = (
        pd.Series(1, index=event_times)
        .resample(out_freq)
        .sum()
        .fillna(0)
    )
    s.name = "hf_event_rate_2_5"
    return s


# Extract all features for one day
def extract_features_for_day(client, day_start, cfg):
    tr = get_day_trace(client, day_start, cfg)

    win_sec = cfg.get("rms_window_sec", 600)
    out_freq = cfg.get("master_freq", "1h")

    hydro = hydro_rms_2_5(tr, win_sec=win_sec)
    ratio = spectral_ratio_4p5_8_over_8_16(tr, win_sec=win_sec)
    hf_rate = hf_event_rate_2_5(tr, out_freq=out_freq)
    effect = effect_tremor_rms_5_15(tr, win_sec=win_sec)

    hydro_h = hydro.resample(out_freq).mean()
    ratio_h = ratio.resample(out_freq).mean()
    effect_h = effect.resample(out_freq).mean()

    df = pd.concat([hydro_h, ratio_h, hf_rate, effect_h], axis=1)

    return df


def build_waveform_dataset(client, start, end, cfg):
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

    waveform_df["hf_event_rate_2_5"] = waveform_df["hf_event_rate_2_5"].fillna(0)

    return waveform_df, failures