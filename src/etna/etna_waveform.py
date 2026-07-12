from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime



def _nan_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index ranges for contiguous True values."""
    mask = np.asarray(mask, dtype=bool)
    groups: list[tuple[int, int]] = []
    start: int | None = None

    for index, is_true in enumerate(mask):
        if is_true and start is None:
            start = index
        elif not is_true and start is not None:
            groups.append((start, index - 1))
            start = None

    if start is not None:
        groups.append((start, len(mask) - 1))

    return groups


def _interpolate_short_internal_gaps(
    values: np.ndarray,
    *,
    max_gap_samples: int,
) -> np.ndarray:
    """Interpolate only bounded internal gaps no longer than the limit."""
    values = np.asarray(values, dtype=float).copy()
    missing = np.isnan(values)

    if not missing.any():
        return values

    sample_index = np.arange(len(values))
    valid = ~missing

    if not valid.any():
        return values

    for start, end in _nan_groups(missing):
        gap_length = end - start + 1
        bounded = (
            start > 0
            and end < len(values) - 1
            and np.isfinite(values[start - 1])
            and np.isfinite(values[end + 1])
        )

        if gap_length <= max_gap_samples and bounded:
            values[start:end + 1] = np.interp(
                sample_index[start:end + 1],
                sample_index[valid],
                values[valid],
            )

    return values


def _minimum_processing_segment_seconds(config: dict) -> float:
    """Minimum continuous segment needed for stable low-frequency processing."""
    window_seconds = float(config["windows_sec"]["teleseismic"])
    fmin = float(config["bands"]["teleseismic"][0])

    default_seconds = max(
        2.0 * window_seconds,
        10.0 / fmin,
    )
    return float(config.get("min_response_segment_sec", default_seconds))


def _continuous_trace_segment(trace, values, start: int, end: int):
    """Create a trace copy for one inclusive finite-data segment."""
    sampling_rate = float(trace.stats.sampling_rate)
    segment = trace.copy()
    segment.data = np.asarray(values[start:end + 1], dtype=np.float64).copy()
    segment.stats.starttime = trace.stats.starttime + start / sampling_rate
    return segment


def preprocess_stream_safely(stream: Stream, config: dict):
    """
    Merge and response-correct all usable continuous waveform segments.

    Original ObsPy masks are converted to NaN. Bounded gaps no longer than
    ``max_interp_gap_sec`` are interpolated. Longer gaps remain NaN in the
    returned trace and are processed as boundaries between independent valid
    segments. Consequently, a few real miniSEED gaps do not invalidate an
    otherwise usable daily chunk.

    The return value remains one Trace for compatibility with the plotting and
    feature-extraction code; its data array contains NaN only at unresolved
    gaps and at continuous segments too short to process safely.
    """
    stream = stream.copy()
    stream.sort()

    if len(stream) == 0:
        raise ValueError("Empty waveform stream.")

    stream.merge(method=0, fill_value=None)

    if len(stream) != 1:
        raise ValueError(f"Expected one merged trace, got {len(stream)}.")

    trace = stream[0]

    # Preserve the original ObsPy mask. np.asarray(trace.data) alone can drop
    # the mask and expose arbitrary fill values as if they were observations.
    masked = np.ma.asarray(trace.data, dtype=float)
    values = masked.filled(np.nan)

    sampling_rate = float(trace.stats.sampling_rate)
    max_gap_seconds = float(config.get("max_interp_gap_sec", 2.0))
    max_gap_samples = max(0, int(round(max_gap_seconds * sampling_rate)))

    gaps_before = _nan_groups(np.isnan(values))
    values = _interpolate_short_internal_gaps(
        values,
        max_gap_samples=max_gap_samples,
    )
    unresolved_gaps = _nan_groups(np.isnan(values))

    corrected = np.full(values.shape, np.nan, dtype=np.float64)
    valid_segments = _nan_groups(np.isfinite(values))

    minimum_segment_samples = max(
        1,
        int(round(_minimum_processing_segment_seconds(config) * sampling_rate)),
    )
    taper_max_seconds = float(config.get("taper_max_sec", 60.0))

    processed_segments = 0
    skipped_segments: list[tuple[int, int, str]] = []

    for start, end in valid_segments:
        segment_samples = end - start + 1

        if segment_samples < minimum_segment_samples:
            skipped_segments.append((start, end, "too short"))
            continue

        segment = _continuous_trace_segment(trace, values, start, end)

        try:
            segment.detrend("linear")
            segment.detrend("demean")
            segment.taper(
                max_percentage=0.02,
                max_length=taper_max_seconds,
            )
            segment.remove_response(
                output=config["response_output"],
                pre_filt=config["pre_filt"],
            )
        except Exception as exc:
            skipped_segments.append((start, end, str(exc)))
            continue

        segment_values = np.asarray(segment.data, dtype=float)
        if len(segment_values) != segment_samples:
            skipped_segments.append(
                (start, end, "response correction changed segment length")
            )
            continue

        finite = np.isfinite(segment_values)
        corrected[start:end + 1][finite] = segment_values[finite]
        processed_segments += 1

    if processed_segments == 0:
        details = "; ".join(reason for _, _, reason in skipped_segments[:3])
        raise ValueError(
            "No continuous waveform segment was long enough or valid enough "
            f"for response correction. {details}"
        )

    output = trace.copy()
    output.data = corrected

    if not hasattr(output.stats, "processing") or output.stats.processing is None:
        output.stats.processing = []

    unresolved_samples = int(np.isnan(values).sum())
    unusable_after_processing = int(np.isnan(corrected).sum())
    output.stats.processing.append(
        "gap_policy: "
        f"short_gap_limit={max_gap_seconds}s, "
        f"gaps_before={len(gaps_before)}, "
        f"unresolved_gaps={len(unresolved_gaps)}, "
        f"unresolved_samples={unresolved_samples}, "
        f"processed_segments={processed_segments}, "
        f"skipped_segments={len(skipped_segments)}, "
        f"unusable_samples_after_processing={unusable_after_processing}"
    )

    return output


def band_rms_windows(
    trace,
    *,
    fmin: float,
    fmax: float,
    window_sec: int,
    filter_edge_guard_sec: float | None = None,
) -> pd.Series:
    """
    Return globally aligned, non-overlapping RMS windows.

    Each finite continuous section is bandpass-filtered independently, so NaN
    gaps cannot contaminate the full trace. A short guard is removed at each
    section boundary to avoid filter-edge artefacts. RMS windows remain aligned
    to the start of the original padded trace, matching the previous feature
    definition.
    """
    sampling_rate = float(trace.stats.sampling_rate)
    samples_per_window = int(round(float(window_sec) * sampling_rate))

    if samples_per_window <= 0:
        raise ValueError("window_sec must produce at least one sample.")

    values = np.asarray(trace.data, dtype=float)
    filtered_values = np.full(values.shape, np.nan, dtype=float)

    if filter_edge_guard_sec is None:
        filter_edge_guard_sec = max(float(window_sec), 3.0 / float(fmin))
    guard_samples = max(0, int(round(filter_edge_guard_sec * sampling_rate)))

    minimum_filter_samples = max(
        samples_per_window,
        2 * guard_samples + samples_per_window,
    )

    for start, end in _nan_groups(np.isfinite(values)):
        segment_samples = end - start + 1
        if segment_samples < minimum_filter_samples:
            continue

        segment = _continuous_trace_segment(trace, values, start, end)

        try:
            segment.filter(
                "bandpass",
                freqmin=float(fmin),
                freqmax=float(fmax),
                corners=4,
                zerophase=True,
            )
        except Exception:
            continue

        segment_values = np.asarray(segment.data, dtype=float)
        filtered_values[start:end + 1] = segment_values

        # The daily download is padded, so removing the filter transient at the
        # outer edges does not affect the retained day. Around real internal
        # gaps this guard prevents artificial peaks from entering hourly maxima.
        if guard_samples > 0:
            left_end = min(end + 1, start + guard_samples)
            right_start = max(start, end - guard_samples + 1)
            filtered_values[start:left_end] = np.nan
            filtered_values[right_start:end + 1] = np.nan

    times = []
    rms_values = []

    for start in range(
        0,
        len(filtered_values) - samples_per_window + 1,
        samples_per_window,
    ):
        window = filtered_values[start:start + samples_per_window]

        if np.isfinite(window).all():
            rms = float(np.sqrt(np.mean(window ** 2)))
        else:
            rms = np.nan

        times.append(trace.stats.starttime + start / sampling_rate)
        rms_values.append(rms)

    if not rms_values:
        return pd.Series(dtype=float, name="teleseismic_rms")

    return pd.Series(
        rms_values,
        index=pd.to_datetime([time.datetime for time in times], utc=True),
        name="teleseismic_rms",
        dtype=float,
    )


def teleseismic_rms_windows(trace, config: dict) -> pd.Series:
    """Return the exact RMS windows used to construct the teleseismic proxy."""
    fmin, fmax = config["bands"]["teleseismic"]
    return band_rms_windows(
        trace,
        fmin=float(fmin),
        fmax=float(fmax),
        window_sec=int(config["windows_sec"]["teleseismic"]),
        filter_edge_guard_sec=config.get("filter_edge_guard_sec"),
    )

def hourly_teleseismic_feature(trace, config: dict) -> pd.Series:
    """
    Aggregate valid 120-second RMS windows to hourly maxima.

    An hour is retained when at least 20 of the expected 30 RMS
    windows are valid, corresponding to at least 40 minutes of
    usable waveform data.
    """
    windows = teleseismic_rms_windows(trace, config)
    frequency = str(config["base_freq"])

    grouped = windows.resample(frequency)
    feature = grouped.max().rename("teleseismic")
    valid_counts = grouped.count()

    hour_seconds = pd.Timedelta(frequency).total_seconds()
    window_seconds = float(config["windows_sec"]["teleseismic"])
    expected_windows = max(
        1,
        int(round(hour_seconds / window_seconds)),
    )

    minimum_valid_windows = int(
        config.get("min_valid_rms_windows_per_hour", 20)
    )

    if not 1 <= minimum_valid_windows <= expected_windows:
        raise ValueError(
            "min_valid_rms_windows_per_hour must be between 1 and "
            f"{expected_windows}."
        )

    return feature.where(valid_counts >= minimum_valid_windows)


def extract_etna_features_for_chunk(
    client,
    station: str,
    chunk_start: UTCDateTime,
    config: dict,
) -> pd.DataFrame:
    """Extract the hourly teleseismic feature for one padded daily chunk."""
    padding = int(config["pad_sec"])
    chunk_seconds = int(config["chunk_sec"])

    stream = client.get_waveforms(
        network=config["network"],
        station=station,
        location=config["location"],
        channel=config["channel"],
        starttime=chunk_start - padding,
        endtime=chunk_start + chunk_seconds + padding,
        attach_response=True,
    )

    trace = preprocess_stream_safely(stream, config)
    feature = hourly_teleseismic_feature(trace, config).to_frame()

    left = pd.Timestamp(chunk_start.datetime, tz="UTC")
    right = left + pd.Timedelta(seconds=chunk_seconds)
    expected_index = pd.date_range(
        start=left,
        end=right,
        freq=config["base_freq"],
        inclusive="left",
        tz="UTC",
    )

    return feature.reindex(expected_index).rename_axis("time")[["teleseismic"]]


def build_station_waveform_dataset(
    client,
    station: str,
    config: dict,
    *,
    cache_path: str | Path | None = None,
    redownload: bool = False,
) -> tuple[pd.DataFrame, list[tuple[object, str]]]:
    """Build or load the complete hourly waveform-feature dataset."""
    cache = Path(cache_path) if cache_path is not None else None

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)

        if cache.exists() and not redownload:
            cached = pd.read_pickle(cache)
            print(f"{station}: loaded cached waveform features from {cache}")

            if isinstance(cached, dict):
                return (
                    cached["waveform_df"],
                    list(cached.get("failures", [])),
                )

            return cached, []

    chunk_starts = []
    current = UTCDateTime(config["start"])

    while current < config["end"]:
        chunk_starts.append(current)
        current += int(config["chunk_sec"])

    chunks: list[pd.DataFrame] = []
    failures: list[tuple[object, str]] = []

    for chunk_start in chunk_starts:
        try:
            chunk = extract_etna_features_for_chunk(
                client,
                station,
                chunk_start,
                config,
            )
            chunks.append(chunk)

            missing_hours = int(chunk["teleseismic"].isna().sum())
            suffix = f" | missing hourly values={missing_hours}" if missing_hours else ""
            print(f"{station} OK   {chunk_start.date}{suffix}")
        except Exception as exc:
            failures.append((chunk_start.date, str(exc)))
            print(f"{station} FAIL {chunk_start.date}: {exc}")

    if not chunks:
        raise RuntimeError(f"No waveform chunks were extracted for station {station}.")

    waveform_df = pd.concat(chunks).sort_index()
    waveform_df = waveform_df[
        ~waveform_df.index.duplicated(keep="first")
    ]
    waveform_df.index = pd.to_datetime(waveform_df.index, utc=True)

    expected_index = pd.date_range(
        start=pd.Timestamp(config["start"].datetime, tz="UTC"),
        end=pd.Timestamp(config["end"].datetime, tz="UTC"),
        freq=config["base_freq"],
        inclusive="left",
        tz="UTC",
    )
    waveform_df = waveform_df.reindex(expected_index)
    waveform_df.index.name = "time"

    if cache is not None:
        pd.to_pickle(
            {
                "waveform_df": waveform_df,
                "failures": failures,
                "station": station,
                "start": config["start"],
                "end": config["end"],
                "config": config,
                "missing_hourly_values": int(
                    waveform_df["teleseismic"].isna().sum()
                ),
            },
            cache,
        )
        print(f"{station}: saved waveform features to {cache}")

    return waveform_df, failures
