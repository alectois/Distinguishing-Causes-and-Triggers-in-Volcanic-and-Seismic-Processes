"""
ETNA-specific waveform processing and feature extraction.
"""

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream
from pathlib import Path

def fetch_waveform_chunk(client, station: str, t1: UTCDateTime, t2: UTCDateTime, cfg: dict):
    st = client.get_waveforms(
        network=cfg["network"],
        station=station,
        location=cfg["location"],
        channel=cfg["channel"],
        starttime=t1,
        endtime=t2,
        attach_response=True,
    )
    return st

def preprocess_stream_safely(st: Stream, cfg: dict):
    st = st.copy()
    st.sort()

    if len(st) == 0:
        raise ValueError("Empty stream")

    st.merge(method=0, fill_value=None)

    if len(st) != 1:
        raise ValueError(f"Expected 1 merged trace, got {len(st)}")

    tr = st[0]

    # convert to float first so NaNs are allowed
    x = np.asarray(tr.data, dtype=float)

    # mask invalid values if any
    data = np.ma.masked_invalid(x)

    sr = tr.stats.sampling_rate
    max_gap_samples = int(cfg["max_interp_gap_sec"] * sr)

    x = data.filled(np.nan)
    isnan = np.isnan(x)

    if isnan.any():
        idx = np.arange(len(x))
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

        valid = ~np.isnan(x)

        for g0, g1 in nan_groups:
            gap_len = g1 - g0 + 1
            if gap_len <= max_gap_samples and valid.any():
                left_ok = g0 > 0 and not np.isnan(x[g0 - 1])
                right_ok = g1 < len(x) - 1 and not np.isnan(x[g1 + 1])
                if left_ok and right_ok:
                    x[g0:g1 + 1] = np.interp(
                        idx[g0:g1 + 1],
                        idx[valid],
                        x[valid]
                    )

    tr.data = x.astype(np.float64)

    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.02)

    tr.remove_response(
        output=cfg["response_output"],
        pre_filt=cfg["pre_filt"]
    )

    return tr

def windowed_metric(trace, fmin, fmax, window_sec, out_freq="1min", metric="rms", agg="mean"):
    tr = trace.copy()
    tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    sr = tr.stats.sampling_rate
    nwin = int(window_sec * sr)
    if nwin <= 0:
        raise ValueError("window_sec too small")

    x = np.asarray(tr.data, dtype=float)

    times = []
    vals = []

    for i in range(0, len(x) - nwin + 1, nwin):
        seg = x[i:i+nwin]

        if np.isnan(seg).any():
            val = np.nan
        else:
            if metric == "rms":
                val = np.sqrt(np.mean(seg ** 2))
            elif metric == "absmean":
                val = np.mean(np.abs(seg))
            elif metric == "maxabs":
                val = np.max(np.abs(seg))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        times.append(tr.stats.starttime + i / sr)
        vals.append(val)

    if len(vals) == 0:
        return pd.Series(dtype=float)

    s = pd.Series(vals, index=pd.to_datetime([t.datetime for t in times], utc=True))
    s = s.resample(out_freq).agg(agg)
    return s

def extract_etna_features_for_chunk(client, station: str, day_start: UTCDateTime, cfg: dict):
    pad = cfg["pad_sec"]
    t1 = day_start - pad
    t2 = day_start + cfg["chunk_sec"] + pad

    st = fetch_waveform_chunk(client, station, t1, t2, cfg)
    tr = preprocess_stream_safely(st, cfg)

    # T: teleseismic trigger band
    T = windowed_metric(
        tr,
        fmin=cfg["bands"]["T"][0],
        fmax=cfg["bands"]["T"][1],
        window_sec=cfg["windows_sec"]["T"],
        out_freq=cfg["base_freq"],
        metric="rms",
        agg="max",     # preserve sharp arrivals
    ).rename("teleseismic_band_raw")

    # S: background / state band
    S = windowed_metric(
        tr,
        fmin=cfg["bands"]["S"][0],
        fmax=cfg["bands"]["S"][1],
        window_sec=cfg["windows_sec"]["S"],
        out_freq=cfg["base_freq"],
        metric="rms",
        agg="mean",
    ).rename("background_seismic_raw")

    # Y: HF response / effect band
    # Use hourly max to preserve burst-like response energy.
    Y = windowed_metric(
        tr,
        fmin=cfg["bands"]["Y"][0],
        fmax=cfg["bands"]["Y"][1],
        window_sec=cfg["windows_sec"]["Y"],
        out_freq=cfg["base_freq"],
        metric="rms",
        agg="max",
    ).rename("effect_seismic_raw")

    df = pd.concat([S, T, Y], axis=1)

    # Trim padding back to exact target chunk
    left = pd.Timestamp(day_start.datetime, tz="UTC")
    right = left + pd.Timedelta(seconds=cfg["chunk_sec"])
    df = df.loc[(df.index >= left) & (df.index < right)].copy()

    # Log-transform positive amplitudes safely
    for raw, logc in [("background_seismic_raw", "background_seismic"), ("teleseismic_band_raw", "teleseismic_band"), ("effect_seismic_raw", "effect_seismic")]:
        df[logc] = np.log1p(df[raw].clip(lower=0))

    return df[["background_seismic", "teleseismic_band", "effect_seismic"]]

def build_station_waveform_dataset(
    client,
    station: str,
    cfg: dict,
    *,
    cache_path: str | None = None,
    redownload: bool = False,
):
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not redownload:
            print(f"{station}: loading cached waveform features from {cache_path}")
            df = pd.read_pickle(cache_path)
            failures = []
            return df, failures

    starts = []
    t = cfg["start"]
    while t < cfg["end"]:
        starts.append(t)
        t += cfg["chunk_sec"]

    chunks = []
    failures = []

    for day_start in starts:
        try:
            df_chunk = extract_etna_features_for_chunk(client, station, day_start, cfg)
            chunks.append(df_chunk)
            print(f"{station} OK  {day_start.date}")
        except Exception as e:
            failures.append((day_start.date, str(e)))
            print(f"{station} FAIL {day_start.date}: {e}")

    if not chunks:
        raise RuntimeError(f"No chunks extracted for station {station}")

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "time"

    if cache_path is not None:
        df.to_pickle(cache_path)
        print(f"{station}: saved waveform features to {cache_path}")

    return df, failures