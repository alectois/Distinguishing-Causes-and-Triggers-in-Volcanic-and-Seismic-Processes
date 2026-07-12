from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime



def _nan_groups(values: np.ndarray) -> list[tuple[int, int]]:
    missing = np.isnan(values)
    groups: list[tuple[int, int]] = []
    start: int | None = None

    for index, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            groups.append((start, index - 1))
            start = None

    if start is not None:
        groups.append((start, len(values) - 1))

    return groups


def _interpolate_short_gaps(
    values: np.ndarray,
    *,
    max_gap_samples: int,
) -> np.ndarray:
    """Interpolate only bounded gaps no longer than ``max_gap_samples``."""
    values = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(values)

    if valid.all() or not valid.any():
        return values

    positions = np.arange(len(values))

    for start, end in _nan_groups(values):
        gap_length = end - start + 1
        bounded = (
            start > 0
            and end < len(values) - 1
            and np.isfinite(values[start - 1])
            and np.isfinite(values[end + 1])
        )
        if gap_length <= max_gap_samples and bounded:
            values[start : end + 1] = np.interp(
                positions[start : end + 1],
                positions[valid],
                values[valid],
            )

    return values


def _hourly_coverage(
    stream: Stream,
    *,
    day_start: UTCDateTime,
) -> pd.Series:
    """Fraction of each target-day hour covered by valid waveform segments."""
    index = pd.date_range(
        pd.Timestamp(day_start.datetime, tz="UTC"),
        periods=24,
        freq="1h",
    )
    coverage = pd.Series(0.0, index=index, name="waveform_coverage")

    for trace in stream:
        sample_interval = 1.0 / float(trace.stats.sampling_rate)
        segment_start = pd.Timestamp(trace.stats.starttime.datetime, tz="UTC")
        segment_end = pd.Timestamp(
            (trace.stats.endtime + sample_interval).datetime,
            tz="UTC",
        )

        for hour_start in index:
            hour_end = hour_start + pd.Timedelta(hours=1)
            overlap_start = max(segment_start, hour_start)
            overlap_end = min(segment_end, hour_end)
            overlap_seconds = max(
                0.0,
                (overlap_end - overlap_start).total_seconds(),
            )
            coverage.loc[hour_start] += overlap_seconds / 3600.0

    return coverage.clip(upper=1.0)


def get_day_stream(
    client,
    day_start,
    cfg: dict,
) -> tuple[Stream, pd.Series]:
    """
    Retrieve and response-correct one padded day.

    Tiny bounded gaps are interpolated. Longer gaps remain masked, the stream is
    split into valid continuous segments, and only affected hourly outputs are
    marked missing. This avoids discarding an otherwise usable full day.
    """
    day_start = UTCDateTime(day_start)
    pad_seconds = float(cfg.get("pad_sec", 0))
    request_start = day_start - pad_seconds
    request_end = day_start + 24 * 3600 + pad_seconds

    stream = client.get_waveforms(
        cfg["network"],
        cfg["station"],
        cfg["location"],
        cfg["channel"],
        request_start,
        request_end,
        attach_response=True,
    ).copy()
    stream.sort()

    if len(stream) == 0:
        raise ValueError(f"Empty waveform stream for {day_start.date}")

    stream.merge(method=0, fill_value=None)
    if len(stream) != 1:
        raise ValueError(
            f"Expected one merged trace for {day_start.date}, got {len(stream)}"
        )

    merged = stream[0]
    masked = np.ma.asarray(merged.data, dtype=float)
    values = masked.filled(np.nan)

    sampling_rate = float(merged.stats.sampling_rate)
    max_gap_samples = int(
        float(cfg.get("max_interp_gap_sec", 2.0)) * sampling_rate
    )
    values = _interpolate_short_gaps(
        values,
        max_gap_samples=max_gap_samples,
    )

    merged.data = np.ma.masked_invalid(values)
    valid_stream = Stream(traces=[merged]).split()

    if len(valid_stream) == 0:
        raise ValueError(f"No valid waveform segments for {day_start.date}")

    coverage = _hourly_coverage(valid_stream, day_start=day_start)

    processed = Stream()
    minimum_segment_seconds = max(
        60.0,
        4.0 / float(cfg.get("pre_filt", (0.5, 1.0, 20.0, 25.0))[1]),
    )

    for segment in valid_stream:
        duration = float(segment.stats.endtime - segment.stats.starttime)
        if duration < minimum_segment_seconds:
            continue

        trace = segment.copy()
        trace.data = np.asarray(trace.data, dtype=np.float64)
        trace.detrend("linear")
        trace.detrend("demean")
        trace.taper(max_percentage=0.02)
        trace.remove_response(
            output=cfg.get("response_output", "VEL"),
            pre_filt=cfg.get("pre_filt", (0.5, 1.0, 20.0, 25.0)),
        )
        processed += trace

    if len(processed) == 0:
        raise ValueError(
            f"No waveform segment was long enough to process for {day_start.date}"
        )

    return processed, coverage


def _band_rms_series(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    window_seconds: int,
) -> pd.Series:
    """Non-overlapping RMS windows computed separately on valid segments."""
    parts: list[pd.Series] = []

    for trace in stream:
        filtered = trace.copy()
        filtered.filter(
            "bandpass",
            freqmin=fmin,
            freqmax=fmax,
            corners=4,
            zerophase=True,
        )

        sampling_rate = float(filtered.stats.sampling_rate)
        window_samples = int(window_seconds * sampling_rate)
        if window_samples <= 0:
            raise ValueError("window_seconds is too small.")

        values = np.asarray(filtered.data, dtype=float)
        times: list[UTCDateTime] = []
        rms_values: list[float] = []

        for start in range(0, len(values) - window_samples + 1, window_samples):
            segment = values[start : start + window_samples]
            times.append(filtered.stats.starttime + start / sampling_rate)
            rms_values.append(float(np.sqrt(np.mean(segment**2))))

        if rms_values:
            parts.append(
                pd.Series(
                    rms_values,
                    index=pd.to_datetime(
                        [time.datetime for time in times],
                        utc=True,
                    ),
                )
            )

    if not parts:
        return pd.Series(dtype=float)

    return pd.concat(parts).sort_index()

def extract_features_for_day(
    client,
    day_start,
    cfg: dict,
) -> pd.DataFrame:
    """Extract the three retained hourly waveform variables for one UTC day."""
    day_start = UTCDateTime(day_start)
    stream, coverage = get_day_stream(client, day_start, cfg)

    window_seconds = int(cfg.get("rms_window_sec", 600))
    output_frequency = str(cfg.get("master_freq", "1h"))
    output_index = pd.date_range(
        pd.Timestamp(day_start.datetime, tz="UTC"),
        periods=24,
        freq=output_frequency,
    )

    hydro = _band_rms_series(
        stream,
        fmin=2.0,
        fmax=5.0,
        window_seconds=window_seconds,
    )
    low = _band_rms_series(
        stream,
        fmin=4.5,
        fmax=8.0,
        window_seconds=window_seconds,
    )
    high = _band_rms_series(
        stream,
        fmin=8.0,
        fmax=16.0,
        window_seconds=window_seconds,
    )
    effect = _band_rms_series(
        stream,
        fmin=5.0,
        fmax=15.0,
        window_seconds=window_seconds,
    )

    spectral_ratio = (
        np.log(low.clip(lower=0) + 1e-30)
        - np.log(high.clip(lower=0) + 1e-30)
    )

    frame = pd.DataFrame(index=output_index)

    frame["hydro_2_5"] = hydro.resample(
        output_frequency
    ).mean()

    frame["spectral_log_ratio_4p5_8_over_8_16"] = (
        spectral_ratio.resample(output_frequency).mean()
    )

    frame["effect_tremor_5_15"] = effect.resample(
        output_frequency
    ).quantile(
        float(cfg.get("effect_hourly_quantile", 0.90))
    )

    minimum_coverage = float(
        cfg.get("minimum_hourly_coverage", 0.999)
    )
    incomplete = (
        coverage.reindex(output_index).fillna(0.0)
        < minimum_coverage
    )
    frame.loc[incomplete, :] = np.nan
    frame.index.name = "time"

    return frame


def build_waveform_dataset(
    client,
    start,
    end,
    cfg: dict,
    *,
    save_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, list[tuple[object, str]]]:
    """Build or load the cached hourly Whakaari waveform feature dataframe."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists() and not overwrite:
            cached = pd.read_pickle(save_path)

            if isinstance(cached, dict):
                return (
                    cached["waveform_df"],
                    list(cached.get("failures", [])),
                )

            return cached, []

    days = pd.date_range(start=start, end=end, freq="D")
    frames: list[pd.DataFrame] = []
    failures: list[tuple[object, str]] = []

    for day in days:
        try:
            frames.append(extract_features_for_day(client, day, cfg))
            print(f"done: {day.date()}")
        except Exception as exc:
            failures.append((day.date(), str(exc)))
            print(f"failed: {day.date()} — {exc}")

    if not frames:
        raise RuntimeError("No waveform days were successfully processed.")

    waveform_df = pd.concat(frames).sort_index()
    waveform_df = waveform_df[~waveform_df.index.duplicated(keep="first")]
    waveform_df.index = pd.to_datetime(waveform_df.index, utc=True)
    waveform_df.index.name = "time"

    if save_path is not None:
        pd.to_pickle(
            {
                "waveform_df": waveform_df,
                "failures": failures,
                "start": start,
                "end": end,
                "config": dict(cfg),
            },
            save_path,
        )

    return waveform_df, failures