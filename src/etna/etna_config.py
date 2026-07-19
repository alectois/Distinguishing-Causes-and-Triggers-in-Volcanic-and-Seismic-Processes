from obspy import UTCDateTime
import pandas as pd


ETNA_WAVEFORM_CONFIG = {
    "network": "IV",
    "station": "ESLN",
    "location": "",
    "channel": "HHZ",
    "start": UTCDateTime("2008-04-10T09:00:00"),
    "end": UTCDateTime("2008-05-19T07:00:00"),
    "chunk_sec": 24 * 3600,
    "pad_sec": 3600,
    "base_freq": "1h",
    "bands": {
        "teleseismic": (0.03, 0.30),
    },
    "windows_sec": {
        "teleseismic": 120,
    },
    "response_output": "VEL",
    "pre_filt": (0.02, 0.03, 30.0, 40.0),
    "max_interp_gap_sec": 2.0,
    "min_valid_rms_windows_per_hour": 20,
}

ETNA_GAS_METEO_COLS = [
    "CO2_3",
    "pressure_drop",
]

ETNA_WEATHER_COLS = ["rainfall_mm"]
EVENT_TIME = UTCDateTime("2008-05-12T06:28:00")


def etna_observable_metadata() -> pd.DataFrame:
    """Return source metadata for the variables retained in the Etna model."""
    rows = [
        {
            "case": "Etna",
            "source_id": "ESLN",
            "source_label": "ESLN seismic station (HHZ)",
            "observable": "teleseismic",
            "observable_label": "Teleseismic-band RMS velocity, 0.03–0.30 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": 37.6934,
            "lon": 14.9744,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Etna",
            "source_id": "EtnaSC_2000_2010",
            "source_label": "INGV-OE Mt. Etna seismic event catalogue, 2000–2010",
            "observable": "local_event_rate_state",
            "observable_label": "Past 48-hour local event-rate state",
            "family": "catalogue_seismicity",
            "spatial_type": "event_catalogue",
            "lat": float("nan"),
            "lon": float("nan"),
            "plot_role": "candidate_state",
        },
        {
            "case": "Etna",
            "source_id": "EtnaSC_2000_2010",
            "source_label": "INGV-OE Mt. Etna seismic event catalogue, 2000–2010",
            "observable": "local_event_rate_response",
            "observable_label": "Positive local event-rate response",
            "family": "catalogue_seismicity",
            "spatial_type": "event_catalogue",
            "lat": float("nan"),
            "lon": float("nan"),
            "plot_role": "effect",
        },
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3C",
            "observable": "CO2_3",
            "observable_label": "Soil CO₂ concentration",
            "family": "gas",
            "spatial_type": "point_or_network_station",
            "lat": 37.6086,
            "lon": 15.0822,
            "plot_role": "direct_or_network_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3C",
            "observable": "pressure_drop",
            "observable_label": "Atmospheric pressure drop",
            "family": "meteorology",
            "spatial_type": "point_or_network_station",
            "lat": 37.6086,
            "lon": 15.0822,
            "plot_role": "direct_or_network_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ETNA_OPENMETEO_PROXY",
            "source_label": "Open-Meteo Etna proxy point",
            "observable": "rainfall_mm",
            "observable_label": "Hourly precipitation",
            "family": "weather_proxy",
            "spatial_type": "proxy_point",
            "lat": 37.75,
            "lon": 14.99,
            "plot_role": "proxy",
        },
    ]
    return pd.DataFrame(rows)
