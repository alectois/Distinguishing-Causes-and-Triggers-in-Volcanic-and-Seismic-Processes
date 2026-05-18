import requests
import pandas as pd
import numpy as np
from io import StringIO
from obspy import UTCDateTime

def get_csv_df(url, timeout=120):
    r = requests.get(url, headers={"Accept": "text/csv"}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()

def tilde_data_url(base_url, domain, station, name, sensor, method, aspect, start, end):
    return f"{base_url}{domain}/{station}/{name}/{sensor}/{method}/{aspect}/{start}/{end}"

def load_so2_flux(tilde_data_base_url, start, end):
    url = tilde_data_url(
        base_url=tilde_data_base_url,
        domain="scandoas",
        station="WID01",
        name="gasflux",
        sensor="01",
        method="reviewed",
        aspect="SO2",
        start=start,
        end=end,
    )

    so2_wid01 = get_csv_df(url)

    so2 = so2_wid01[["value"]].rename(columns={"value": "SO2_flux"})
    so2.index = pd.to_datetime(so2.index, utc=True)
    return so2.sort_index()

def load_gnss_deformation(tilde_data_base_url, start, end):
    rgwc_url = tilde_data_url(
        base_url=tilde_data_base_url,
        domain="gnss",
        station="RGWC",
        name="displacement",
        sensor="nil",
        method="1d",
        aspect="up",
        start=start,
        end=end,
    )

    rgwi_url = tilde_data_url(
        base_url=tilde_data_base_url,
        domain="gnss",
        station="RGWI",
        name="displacement",
        sensor="nil",
        method="1d",
        aspect="up",
        start=start,
        end=end,
    )

    gnss_rgwc_up = get_csv_df(rgwc_url)
    gnss_rgwi_up = get_csv_df(rgwi_url)

    up_rgwc = gnss_rgwc_up["value"].rename("GNSS_up_RGWC")
    up_rgwi = gnss_rgwi_up["value"].rename("GNSS_up_RGWI")

    gnss_up = pd.concat([up_rgwc, up_rgwi], axis=1).sort_index()
    gnss_up["GNSS_deformation"] = (
        gnss_up["GNSS_up_RGWC"] - gnss_up["GNSS_up_RGWI"]
    )

    gnss = gnss_up[["GNSS_deformation"]].copy()
    gnss.index = pd.to_datetime(gnss.index, utc=True)

    return gnss.sort_index()

def load_weather_vars(lat, lon, start, end, api_alpha=0.05):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=precipitation,surface_pressure"
        "&timezone=UTC"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "rainfall_mm": data["hourly"]["precipitation"],
        "surface_pressure_hPa": data["hourly"]["surface_pressure"],
    }).set_index("timestamp").sort_index()

    weather.index = pd.to_datetime(weather.index, utc=True)

    weather["pressure_change"] = weather["surface_pressure_hPa"].diff()
    weather["pressure_drop"] = -weather["pressure_change"]
    weather["rain_24h_sum"] = weather["rainfall_mm"].rolling(24, min_periods=1).sum()
    weather["API"] = weather["rainfall_mm"].ewm(alpha=api_alpha).mean()

    weather = weather.dropna()

    return weather[["API", "pressure_drop"]].copy()


def load_local_earthquake_counts(
    client,
    start,
    end,
    latitude,
    longitude,
    radius_km=30.0,
    min_magnitude=None,
    master_freq="1h",
):
    """
    Load local earthquakes around Whakaari from the GeoNet/FDSN event service
    and convert them to hourly count variables.
    """
    start_utc = UTCDateTime(start)
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    end_utc = UTCDateTime(end_ts.to_pydatetime())

    # FDSN maxradius is in degrees. Approximate conversion:
    # 1 degree latitude ≈ 111.2 km.
    maxradius_deg = radius_km / 111.2

    kwargs = {
        "starttime": start_utc,
        "endtime": end_utc,
        "latitude": latitude,
        "longitude": longitude,
        "maxradius": maxradius_deg,
    }

    if min_magnitude is not None:
        kwargs["minmagnitude"] = min_magnitude

    catalog = client.get_events(**kwargs)

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

    hourly_index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
        freq=master_freq,
    )

    if events_df.empty:
        local_eq = pd.DataFrame(index=hourly_index)
        local_eq["local_eq_count_1h"] = 0.0
        #local_eq["local_eq_count_24h"] = 0.0
        local_eq.index.name = "timestamp"
        return local_eq, events_df

    events_df = events_df.sort_values("timestamp").set_index("timestamp")

    local_eq_count_1h = (
        pd.Series(1, index=events_df.index)
        .resample(master_freq)
        .sum()
        .reindex(hourly_index)
        .fillna(0)
    )

    local_eq = pd.DataFrame(index=hourly_index)
    local_eq["local_eq_count_1h"] = local_eq_count_1h
    #local_eq["local_eq_count_24h"] = (local_eq["local_eq_count_1h"].rolling(24, min_periods=1).sum())

    local_eq.index.name = "timestamp"

    return local_eq, events_df.reset_index()