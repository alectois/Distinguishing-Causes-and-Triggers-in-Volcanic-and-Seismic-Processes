from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _requests_session_with_retries(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
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


def _read_cached_indexed_csv(path: Path) -> pd.DataFrame:
    cached = pd.read_csv(path, parse_dates=["timestamp"])
    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="raise")
    return cached.set_index("timestamp").sort_index()


def get_csv_df(
    url: str,
    *,
    timeout: int = 120,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download a Tilde CSV response as a sorted UTC-indexed dataframe."""
    session = _requests_session_with_retries() if session is None else session
    response = session.get(url, headers={"Accept": "text/csv"}, timeout=timeout)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if "timestamp" not in df.columns:
        raise ValueError("Tilde response does not contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Tilde response contains invalid timestamps.")

    df = (
        df.drop_duplicates(subset="timestamp", keep="first")
        .set_index("timestamp")
        .sort_index()
    )
    return df


def tilde_data_url(
    base_url: str,
    domain: str,
    station: str,
    name: str,
    sensor: str,
    method: str,
    aspect: str,
    start: str,
    end: str,
) -> str:
    return f"{base_url}{domain}/{station}/{name}/{sensor}/{method}/{aspect}/{start}/{end}"


def load_gnss_deformation(
    tilde_data_base_url: str,
    start: str,
    end: str,
    *,
    cache_path: str | Path | None = None,
    redownload: bool = False,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Load daily vertical displacement for RGWC and RGWI and return their difference.

    The returned series is a deformation level. The past-only daily change is
    constructed later in ``build_master_dataframe``.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not redownload:
            return _read_cached_indexed_csv(cache_path)

    session = _requests_session_with_retries()

    def load_station(station: str) -> pd.Series:
        url = tilde_data_url(
            base_url=tilde_data_base_url,
            domain="gnss",
            station=station,
            name="displacement",
            sensor="nil",
            method="1d",
            aspect="up",
            start=start,
            end=end,
        )
        frame = get_csv_df(url, timeout=timeout, session=session)
        if "value" not in frame.columns:
            raise ValueError(f"Tilde GNSS response for {station} has no 'value' column.")
        values = pd.to_numeric(frame["value"], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Tilde GNSS response for {station} contains invalid values.")
        return values.rename(f"GNSS_up_{station}")

    up_rgwc = load_station("RGWC")
    up_rgwi = load_station("RGWI")

    gnss_up = pd.concat([up_rgwc, up_rgwi], axis=1, join="inner").sort_index()
    if gnss_up.empty:
        raise ValueError("RGWC and RGWI have no overlapping GNSS observations.")

    gnss = pd.DataFrame(
        {
            "GNSS_deformation": (
                gnss_up["GNSS_up_RGWC"] - gnss_up["GNSS_up_RGWI"]
            )
        },
        index=gnss_up.index,
    )
    gnss.index = pd.to_datetime(gnss.index, utc=True)
    gnss.index.name = "timestamp"

    if cache_path is not None:
        gnss.reset_index().to_csv(cache_path, index=False, float_format="%.17g")

    return gnss.sort_index()


def load_weather_vars(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    timeout: int = 60,
    cache_path: str | Path | None = None,
    redownload: bool = False,
) -> pd.DataFrame:
    """
    Load hourly precipitation and atmospheric-pressure drop.

    A buffer interval should be requested before the analysis start so the first
    retained analysis hour has a real pressure difference. The first downloaded
    row is dropped because its pressure change is undefined.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not redownload:
            return _read_cached_indexed_csv(cache_path)

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
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        if cache_path is not None and cache_path.exists():
            return _read_cached_indexed_csv(cache_path)
        raise RuntimeError(
            "Could not retrieve Whakaari Open-Meteo data and no usable cache exists."
        ) from exc

    try:
        hourly = data["hourly"]
        weather = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"], utc=True),
                "rainfall_mm": hourly["precipitation"],
                "surface_pressure_hPa": hourly["surface_pressure"],
            }
        ).set_index("timestamp").sort_index()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Open-Meteo response has an unexpected structure.") from exc

    weather["rainfall_mm"] = pd.to_numeric(weather["rainfall_mm"], errors="coerce")
    pressure = pd.to_numeric(weather["surface_pressure_hPa"], errors="coerce")
    weather["pressure_drop"] = -pressure.diff()

    weather_out = weather[["rainfall_mm", "pressure_drop"]].iloc[1:].copy()

    if weather_out.isna().any().any():
        missing = weather_out.isna().sum()
        raise ValueError(
            "Whakaari weather variables contain missing API values:\n"
            f"{missing[missing > 0]}"
        )

    if weather_out.index.has_duplicates:
        raise ValueError("Whakaari weather data contain duplicate timestamps.")

    deltas = weather_out.index.to_series().diff().dropna()
    if not deltas.eq(pd.Timedelta("1h")).all():
        raise ValueError("Whakaari weather data are not a complete hourly grid.")

    weather_out.index.name = "timestamp"

    if cache_path is not None:
        weather_out.reset_index().to_csv(
            cache_path,
            index=False,
            float_format="%.17g",
        )

    return weather_out
