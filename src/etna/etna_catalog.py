from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from obspy import read_events


def load_ingv_oe_events(
    start,
    end,
    latitude,
    longitude,
    radius_km=30.0,
    min_magnitude=None,
    max_depth_km=None,
    base_url="https://sismoweb.ct.ingv.it/fdsnws/event/1/query",
    timeout=120,
):
    """
    Load local earthquake events from the INGV-OE FDSN event service.

    Parameters
    ----------
    start, end : str or datetime-like
        Time window for catalogue query.
    latitude, longitude : float
        Centre of radius search.
    radius_km : float
        Search radius around centre point.
    min_magnitude : float or None
        Optional minimum magnitude.
    max_depth_km : float or None
        Optional maximum depth in km.
    base_url : str
        INGV-OE FDSN event query URL.

    Returns
    -------
    events_df : pd.DataFrame
        Event table with timestamp, latitude, longitude, depth_km, magnitude.
    """
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    # FDSN maxradius is in degrees.
    maxradius_deg = radius_km / 111.2

    params = {
        "starttime": start_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": latitude,
        "longitude": longitude,
        "maxradius": maxradius_deg,
        "format": "xml",
    }

    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude

    if max_depth_km is not None:
        params["maxdepth"] = max_depth_km

    response = requests.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()

    if len(response.content) == 0:
        return pd.DataFrame(
            columns=["timestamp", "latitude", "longitude", "depth_km", "magnitude"]
        )

    catalog = read_events(BytesIO(response.content), format="QUAKEML")

    rows = []

    for event in catalog:
        if len(event.origins) == 0:
            continue

        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude()

        if magnitude is None and len(event.magnitudes) > 0:
            magnitude = event.magnitudes[0]

        rows.append({
            "timestamp": pd.Timestamp(origin.time.datetime, tz="UTC"),
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "depth_km": origin.depth / 1000 if origin.depth is not None else np.nan,
            "magnitude": magnitude.mag if magnitude is not None else np.nan,
        })

    events_df = pd.DataFrame(rows)

    if events_df.empty:
        return events_df

    return events_df.sort_values("timestamp").reset_index(drop=True)


def build_local_eq_features(
    events_df,
    start,
    end,
    freq="1h",
):
    """
    Convert local Etna earthquake catalogue into one hourly feature:

        local_eq_count_24h

    This is the rolling 24-hour count of local earthquakes and is used as
    a compact proxy for short-term volcanic seismic unrest.
    """
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    hourly_index = pd.date_range(
        start=start_ts,
        end=end_ts - pd.Timedelta(hours=1),
        freq=freq,
        tz="UTC",
    )

    features = pd.DataFrame(index=hourly_index)
    features.index.name = "time"

    if events_df.empty:
        features["local_eq_count_24h"] = 0.0
        return features

    events = events_df.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events = events.set_index("timestamp").sort_index()

    local_eq_count_1h = (
        pd.Series(1, index=events.index)
        .resample(freq)
        .sum()
        .reindex(hourly_index)
        .fillna(0)
    )

    features["local_eq_count_24h"] = (
        local_eq_count_1h
        .rolling(24, min_periods=1)
        .sum()
    )

    return features


def load_local_eq_features(
    start,
    end,
    latitude,
    longitude,
    radius_km=30.0,
    min_magnitude=None,
    max_depth_km=None,
    base_url="https://sismoweb.ct.ingv.it/fdsnws/event/1/query",
    freq="1h",
):
    """
    Load INGV-OE events and derive local_eq_count_24h.
    """
    events_df = load_ingv_oe_events(
        start=start,
        end=end,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        min_magnitude=min_magnitude,
        max_depth_km=max_depth_km,
        base_url=base_url,
    )

    features = build_local_eq_features(
        events_df=events_df,
        start=start,
        end=end,
        freq=freq,
    )

    return features, events_df