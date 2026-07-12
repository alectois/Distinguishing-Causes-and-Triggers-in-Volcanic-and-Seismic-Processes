"""
Etna station availability screening.It discovers vertical streams, filters them by distance from the
Etna summit, probes a short waveform segment each day, summarizes daily probe availability, and optionally exports readable CSV/TXT reports.
"""

from __future__ import annotations

from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
from obspy import UTCDateTime


DEFAULT_ETNA_SUMMIT_LAT = 37.748
DEFAULT_ETNA_SUMMIT_LON = 14.999

DEFAULT_CHANNEL_RANK = {
    "HHZ": 1,
    "BHZ": 2,
    "EHZ": 3,
    "SHZ": 4,
    "LHZ": 5,
    "VHZ": 6,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two latitude/longitude points."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def utc_day_starts(start: UTCDateTime, end: UTCDateTime) -> list[UTCDateTime]:
    """Return UTC day-start timestamps from start inclusive to end exclusive."""
    t = UTCDateTime(start)
    out: list[UTCDateTime] = []

    while t < end:
        out.append(t)
        t += 24 * 3600

    return out


def _format_utcdatetime_date(value: Any, missing: str = "open/unknown") -> str:
    """Format ObsPy UTCDateTime/pandas timestamps safely as YYYY-MM-DD."""
    if value is None or pd.isna(value):
        return missing

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def discover_vertical_streams(
    client,
    *,
    network: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    channel_pattern: str = "*HZ",
    summit_lat: float = DEFAULT_ETNA_SUMMIT_LAT,
    summit_lon: float = DEFAULT_ETNA_SUMMIT_LON,
) -> pd.DataFrame:
    """
    Query station metadata for all vertical streams available at any point
    in the requested analysis window.
    """
    inv = client.get_stations(
        network=network,
        station="*",
        location="*",
        channel=channel_pattern,
        starttime=starttime,
        endtime=endtime,
        level="channel",
    )

    rows: list[dict[str, Any]] = []

    for net in inv:
        for sta in net:
            for ch in sta.channels:
                rows.append({
                    "network": net.code,
                    "station": sta.code,
                    "location": ch.location_code,
                    "channel": ch.code,
                    "metadata_start": ch.start_date,
                    "metadata_end": ch.end_date,
                    "metadata_start_date": _format_utcdatetime_date(ch.start_date, missing="unknown"),
                    "metadata_end_date": _format_utcdatetime_date(ch.end_date, missing="open/unknown"),
                    "lat": ch.latitude,
                    "lon": ch.longitude,
                    "elevation_m": ch.elevation,
                    "distance_km": haversine_km(
                        ch.latitude,
                        ch.longitude,
                        summit_lat,
                        summit_lon,
                    ),
                })

    streams = pd.DataFrame(rows)

    if streams.empty:
        raise RuntimeError("No vertical streams found from metadata query.")

    streams = (
        streams
        .sort_values(["distance_km", "station", "channel", "location"])
        .drop_duplicates(["network", "station", "location", "channel"], keep="first")
        .reset_index(drop=True)
    )

    return streams


def probe_stream_daily(
    client,
    *,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    probe_offset_hours: int = 12,
    probe_duration_minutes: int = 10,
    min_coverage_fraction: float = 0.80,
) -> pd.DataFrame:
    """
    For one stream, download a short probe segment each day.
    This checks day-by-day waveform existence without downloading full days.
    """
    rows: list[dict[str, Any]] = []

    for day_start in utc_day_starts(starttime, endtime):
        t1 = day_start + probe_offset_hours * 3600
        t2 = t1 + probe_duration_minutes * 60

        if t2 > endtime:
            t2 = endtime

        try:
            st = client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=t1,
                endtime=t2,
                attach_response=False,
            )

            duration = sum(
                max(0, tr.stats.endtime - tr.stats.starttime)
                for tr in st
            )

            expected = t2 - t1
            coverage = duration / expected if expected > 0 else np.nan
            ok = len(st) > 0 and coverage >= min_coverage_fraction

            rows.append({
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,
                "date": day_start.date.isoformat(),
                "ok": ok,
                "n_traces": len(st),
                "coverage_fraction": coverage,
                "sample_rates": ",".join(
                    sorted({str(tr.stats.sampling_rate) for tr in st})
                ),
                "error": "",
            })

        except Exception as exc:
            rows.append({
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,
                "date": day_start.date.isoformat(),
                "ok": False,
                "n_traces": 0,
                "coverage_fraction": 0.0,
                "sample_rates": "",
                "error": str(exc).split("\n")[0][:180],
            })

    return pd.DataFrame(rows)


def _date_ranges_from_ok_days(dates):
    """
    Convert successful daily-probe dates into compact continuous ranges.
    """
    dates = pd.to_datetime(
        pd.Series(dates)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) == 0:
        return ""

    ranges: list[str] = []
    start = dates[0]
    prev = dates[0]

    for d in dates[1:]:
        if d == prev + pd.Timedelta(days=1):
            prev = d
        else:
            if start == prev:
                ranges.append(start.strftime("%Y-%m-%d"))
            else:
                ranges.append(f"{start.strftime('%Y-%m-%d')} to {prev.strftime('%Y-%m-%d')}")
            start = d
            prev = d

    if start == prev:
        ranges.append(start.strftime("%Y-%m-%d"))
    else:
        ranges.append(f"{start.strftime('%Y-%m-%d')} to {prev.strftime('%Y-%m-%d')}")

    return "; ".join(ranges)


def _summarize_probe_group(group: pd.DataFrame) -> pd.Series:
    ok_dates = pd.to_datetime(group.loc[group["ok"], "date"])

    actual_start = pd.NaT if ok_dates.empty else ok_dates.min()
    actual_end = pd.NaT if ok_dates.empty else ok_dates.max()

    sample_rates = ",".join(
        sorted(set(",".join(group["sample_rates"].astype(str)).split(",")) - {"", "nan"})
    )

    return pd.Series({
        "n_days": len(group),
        "ok_days": int(group["ok"].sum()),
        "ok_fraction": float(group["ok"].mean()),
        "successful_probe_start": actual_start,
        "successful_probe_end": actual_end,
        "successful_probe_start_date": (
            pd.to_datetime(actual_start).strftime("%Y-%m-%d")
            if pd.notna(actual_start) else "none/open"
        ),
        "successful_probe_end_date": (
            pd.to_datetime(actual_end).strftime("%Y-%m-%d")
            if pd.notna(actual_end) else "none/open"
        ),
        "successful_probe_ranges": _date_ranges_from_ok_days(
            group.loc[group["ok"], "date"].astype(str).tolist()
        ),
        "missing_dates": ", ".join(group.loc[~group["ok"], "date"].astype(str).tolist()),
        "mean_probe_coverage": group["coverage_fraction"].mean(),
        "min_probe_coverage": group["coverage_fraction"].min(),
        "n_error_days": int((group["error"].astype(str) != "").sum()),
        "sample_rates": sample_rates,
    })


def summarize_daily_probe(
    daily_probe: pd.DataFrame,
    *,
    channel_rank: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Summarize daily probe success for each station/location/channel."""
    if channel_rank is None:
        channel_rank = DEFAULT_CHANNEL_RANK

    summary = (
        daily_probe
        .groupby(
            [
                "network",
                "station",
                "location",
                "channel",
                "lat",
                "lon",
                "distance_km",
                "metadata_start_date",
                "metadata_end_date",
            ],
            dropna=False,
        )
        .apply(_summarize_probe_group, include_groups=False) 
        .reset_index()
    )

    summary["channel_rank"] = summary["channel"].map(channel_rank).fillna(99)

    summary = summary.sort_values(
        ["ok_fraction", "distance_km", "channel_rank", "mean_probe_coverage"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    return summary


def run_station_screening(
    client,
    waveform_cfg: dict,
    *,
    summit_lat: float = DEFAULT_ETNA_SUMMIT_LAT,
    summit_lon: float = DEFAULT_ETNA_SUMMIT_LON,
    max_distance_km: float = 80.0,
    channel_pattern: str = "*HZ",
    probe_offset_hours: int = 12,
    probe_duration_minutes: int = 10,
    min_coverage_fraction: float = 0.80,
    verbose: bool = True,
    display_func: Callable[[pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reproducible station screen:
    metadata discovery -> distance filtering -> daily probe -> ranked summary.
    """
    network = waveform_cfg["network"]
    starttime = waveform_cfg["start"]
    endtime = waveform_cfg["end"]

    streams_all = discover_vertical_streams(
        client,
        network=network,
        starttime=starttime,
        endtime=endtime,
        channel_pattern=channel_pattern,
        summit_lat=summit_lat,
        summit_lon=summit_lon,
    )

    streams = (
        streams_all
        .loc[streams_all["distance_km"] <= max_distance_km]
        .copy()
        .sort_values(["distance_km", "station", "channel", "location"])
        .reset_index(drop=True)
    )

    if streams.empty:
        raise RuntimeError(
            f"No vertical streams found within {max_distance_km} km of Etna summit."
        )

    if verbose:
        print(f"Discovered {len(streams_all)} vertical stream(s) before distance filtering.")
        print(
            f"Keeping {len(streams)} vertical stream(s) within "
            f"{max_distance_km:.0f} km of Etna summit."
        )
        if display_func is not None:
            display_func(streams)

    all_daily: list[pd.DataFrame] = []

    for i, row in streams.iterrows():
        if verbose:
            print(
                f"[{i + 1}/{len(streams)}] "
                f"{row.network}.{row.station}.{row.location}.{row.channel} "
                f"| distance={row.distance_km:.1f} km "
                f"| metadata period={row.metadata_start_date} to {row.metadata_end_date}"
            )

        daily = probe_stream_daily(
            client,
            network=row.network,
            station=row.station,
            location=row.location,
            channel=row.channel,
            starttime=starttime,
            endtime=endtime,
            probe_offset_hours=probe_offset_hours,
            probe_duration_minutes=probe_duration_minutes,
            min_coverage_fraction=min_coverage_fraction,
        )

        daily["distance_km"] = row.distance_km
        daily["lat"] = row.lat
        daily["lon"] = row.lon
        daily["metadata_start_date"] = row.metadata_start_date
        daily["metadata_end_date"] = row.metadata_end_date
        all_daily.append(daily)

    daily_probe = pd.concat(all_daily, ignore_index=True)
    summary = summarize_daily_probe(daily_probe)

    return streams, daily_probe, summary


def export_station_screening_results(
    *,
    station_summary: pd.DataFrame,
    daily_probe: pd.DataFrame,
    out_dir: str | Path,
    waveform_cfg: dict,
    summit_lat: float = DEFAULT_ETNA_SUMMIT_LAT,
    summit_lon: float = DEFAULT_ETNA_SUMMIT_LON,
    probe_offset_hours: int = 12,
    probe_duration_minutes: int = 10,
) -> dict[str, Path]:
    """Export station-screening results to CSV and a readable text report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_cols = [
        "station",
        "channel",
        "location",
        "metadata_start_date",
        "metadata_end_date",
        "successful_probe_start_date",
        "successful_probe_end_date",
        "successful_probe_ranges",
        "missing_dates",
        "lat",
        "lon",
        "distance_km",
        "ok_days",
        "n_days",
        "ok_fraction",
        "mean_probe_coverage",
        "min_probe_coverage",
        "n_error_days",
        "sample_rates",
        "channel_rank",
    ]

    summary_cols = [c for c in summary_cols if c in station_summary.columns]

    summary_export = (
        station_summary[summary_cols]
        .sort_values(["ok_fraction", "distance_km", "channel_rank"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    best_candidates = (
        summary_export
        .query("ok_fraction == 1.0")
        .sort_values(["distance_km", "channel_rank", "mean_probe_coverage"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    summary_csv = out_dir / "etna_station_screening_summary.csv"
    daily_csv = out_dir / "etna_station_screening_daily_probe.csv"
    best_csv = out_dir / "etna_station_screening_best_complete_candidates.csv"
    report_txt = out_dir / "etna_station_screening_report.txt"

    summary_export.to_csv(summary_csv, index=False)
    daily_probe.to_csv(daily_csv, index=False)
    best_candidates.to_csv(best_csv, index=False)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Etna station availability screening\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Network: {waveform_cfg['network']}\n")
        f.write(f"Screen window: {waveform_cfg['start']} to {waveform_cfg['end']}\n")
        f.write(
            f"Probe: {probe_duration_minutes} minutes per day "
            f"at +{probe_offset_hours} h from day start\n"
        )
        f.write(f"Etna summit reference: lat={summit_lat}, lon={summit_lon}\n\n")

        f.write("Ranking rule:\n")
        f.write("1. Highest ok_fraction\n")
        f.write("2. Shortest distance to Etna summit\n")
        f.write("3. Best channel rank: HHZ, BHZ, EHZ, SHZ, LHZ, VHZ\n\n")

        f.write("Best complete candidates\n")
        f.write("-" * 40 + "\n")

        if best_candidates.empty:
            f.write("No station-channel stream had ok_fraction == 1.0.\n\n")
        else:
            for _, row in best_candidates.head(20).iterrows():
                f.write(
                    f"{row['station']}.{row['location']}.{row['channel']} | "
                    f"distance={row['distance_km']:.2f} km | "
                    f"successful_probes={row.get('successful_probe_start_date', 'none')} "
                    f"to {row.get('successful_probe_end_date', 'none')} | "
                    f"ok={int(row['ok_days'])}/{int(row['n_days'])} days | "
                    f"mean_probe_coverage={row['mean_probe_coverage']:.3f} | "
                    f"metadata={row.get('metadata_start_date', 'unknown')} "
                    f"to {row.get('metadata_end_date', 'open/unknown')} | "
                    f"sample_rates={row.get('sample_rates', '')}\n"
                )

        f.write("\n\nFull ranked summary\n")
        f.write("-" * 40 + "\n")
        f.write(
            summary_export.to_string(
                index=False,
                max_rows=300,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    return {
        "summary_csv": summary_csv,
        "daily_csv": daily_csv,
        "best_csv": best_csv,
        "report_txt": report_txt,
    }
