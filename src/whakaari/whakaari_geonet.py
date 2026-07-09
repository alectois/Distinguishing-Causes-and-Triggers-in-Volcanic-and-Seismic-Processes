import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

def _requests_session_with_retries(
    total_retries=5,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
):
    session = requests.Session()

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET",),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def load_weather_vars(
    lat,
    lon,
    start,
    end,
    *,
    timeout=60,
    cache_path=None,
    redownload=False,
):
    """
    Load Open-Meteo weather variables for Whakaari.

    Returns:
        rain_12h_sum
        pressure_drop

    Uses a local cache when available so the notebook is reproducible and does
    not fail every time Open-Meteo is temporarily unreachable.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not redownload:
            cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
            return cached.set_index("timestamp").sort_index()

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation,surface_pressure",
        "timezone": "UTC",
    }

    session = _requests_session_with_retries()

    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()

        try:
            data = r.json()
        except requests.exceptions.JSONDecodeError as exc:
            if cache_path is not None and cache_path.exists():
                cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
                cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
                return cached.set_index("timestamp").sort_index()

            debug_path = None
            if cache_path is not None:
                debug_path = Path(cache_path).with_suffix(".openmeteo_response_debug.txt")
                debug_path.write_text(
                    f"URL: {r.url}\n"
                    f"Status code: {r.status_code}\n"
                    f"Content-Type: {r.headers.get('Content-Type')}\n\n"
                    f"{r.text[:5000]}",
                    encoding="utf-8",
                )

            raise RuntimeError(
                "Open-Meteo returned a non-JSON response and no usable cache exists. "
                f"Status={r.status_code}, "
                f"Content-Type={r.headers.get('Content-Type')!r}. "
                f"Debug response saved to: {debug_path}"
            ) from exc

    except requests.exceptions.RequestException as exc:
        if cache_path is not None and cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
            return cached.set_index("timestamp").sort_index()

        raise RuntimeError(
            "Could not download Whakaari Open-Meteo weather data, and no usable cache exists."
        ) from exc
    
    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "rainfall_mm": data["hourly"]["precipitation"],
        "surface_pressure_hPa": data["hourly"]["surface_pressure"],
    }).set_index("timestamp").sort_index()

    weather["rainfall_mm"] = pd.to_numeric(weather["rainfall_mm"], errors="coerce")
    weather["surface_pressure_hPa"] = pd.to_numeric(
        weather["surface_pressure_hPa"],
        errors="coerce",
    )

    weather["pressure_change"] = weather["surface_pressure_hPa"].diff()
    weather["pressure_drop"] = -weather["pressure_change"]

    # Keep first row instead of dropping it only because diff() creates one NaN.
    weather["pressure_drop"] = weather["pressure_drop"].fillna(0)

    weather["rain_12h_sum"] = (
        weather["rainfall_mm"]
        .fillna(0)
        .rolling(window=12, min_periods=1)
        .sum()
    )

    weather_out = weather[["rain_12h_sum", "pressure_drop"]].copy()

    if weather_out.isna().any().any():
        missing = weather_out.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(f"Weather variables contain NaNs after processing:\n{missing}")

    if cache_path is not None:
        weather_out.reset_index().to_csv(
            cache_path,
            index=False,
            float_format="%.17g",
        )

    return weather_out