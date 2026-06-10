import requests
import pandas as pd
import numpy as np
from io import StringIO

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
    weather["API"] = (
        pd.to_numeric(weather["rainfall_mm"], errors="coerce")
        .fillna(0)
        .ewm(alpha=api_alpha, adjust=False)
        .mean()
    )

    weather = weather.dropna()

    return weather[["API", "pressure_drop"]].copy()