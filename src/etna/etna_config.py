from obspy import UTCDateTime
import pandas as pd

# configure the processing parameters for the study
ETNA_WAVEFORM_CONFIG = {
    "network": "IV",
    "location": "",
    "channel": "HHZ",
    "stations": ["ESLN"],

    "start": UTCDateTime("2008-04-12T00:00:00"),
    "end": UTCDateTime("2008-05-14T00:00:00"),

    # process in daily chunks, but pad each day for filtering stability
    "chunk_sec": 24 * 3600,
    "pad_sec": 3600,
    "base_freq": "1h",

    # physical bands
    "bands": {
        "teleseismic": (0.03, 0.30),          # teleseismic / Wenchuan wavefield
        "background_seismic": (0.80, 2.30),   # Etna tremor/state band
        "effect_seismic": (4.00, 8.00),       # HF local response/effect
    },

    # metric windows in seconds
    "windows_sec": {
        "teleseismic": 120,
        "background_seismic": 600,   # slower state
        "effect_seismic": 60,    # response
    },

    # remove response to velocity
    "response_output": "VEL",
    "pre_filt": (0.02, 0.03, 30.0, 40.0),
    # gap policy
    "max_interp_gap_sec": 2.0, # only interpolate tiny gaps
}

ETNA_GAS_METEO_COLS = [
    "CO2_3",
    "AirTemp_3",
    "pressure_drop",
    "WindSpeed",
]
ETNA_WEATHER_COLS = ["API"]
ETNA_EVENT_TIME = UTCDateTime("2008-05-12T06:28:00")

# etna geo-metadata:
def etna_observable_metadata():
    rows = [
        # esln waveform-derived variables
        {
            "case": "Etna",
            "source_id": "ESLN",
            "source_label": "ESLN seismic station (HHZ)",
            "observable": "teleseismic",
            "observable_label": "Teleseismic energy, 0.03–0.30 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": 37.69,
            "lon": 14.97,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ESLN",
            "source_label": "ESLN seismic station (HHZ)",
            "observable": "background_seismic",
            "observable_label": "Etna tremor/state energy, 0.80–2.30 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": 37.69,
            "lon": 14.97,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ESLN",
            "source_label": "ESLN seismic station (HHZ)",
            "observable": "effect_seismic",
            "observable_label": "High-frequency seismic response, 4–8 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": 37.69,
            "lon": 14.97,
            "plot_role": "effect",
        },

        # Etna gas/meteo variables
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3c",
            "observable": "CO2_3",
            "observable_label": "Soil CO₂ concentration",
            "family": "gas",
            "spatial_type": "point_or_network_station",
            "lat": 37.61,
            "lon": 15.08,
            "plot_role": "direct_or_network_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3c",
            "observable": "AirTemp_3",
            "observable_label": "Air temperature",
            "family": "meteorology",
            "spatial_type": "point_or_network_station",
            "lat": 37.61,
            "lon": 15.08,
            "plot_role": "direct_or_network_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3c",
            "observable": "pressure_drop",
            "observable_label": "Atmospheric pressure drop",
            "family": "meteorology",
            "spatial_type": "point_or_network_station",
            "lat": 37.61,
            "lon": 15.08,
            "plot_role": "direct_or_network_measurement",
        },
        {
            "case": "Etna",
            "source_id": "ETNAGAS_3",
            "source_label": "ETNAGAS network, 3c",
            "observable": "WindSpeed",
            "observable_label": "Wind speed",
            "family": "meteorology",
            "spatial_type": "point_or_network_station",
            "lat": 37.61,
            "lon": 15.08,
            "plot_role": "direct_or_network_measurement",
        },

        # Etna proxy/source observables
        {
            "case": "Etna",
            "source_id": "ETNA_OPENMETEO_PROXY",
            "source_label": "Open-Meteo Etna proxy point",
            "observable": "API",
            "observable_label": "Antecedent precipitation index",
            "family": "weather_proxy",
            "spatial_type": "proxy_point",
            "lat": 37.75,
            "lon": 14.99,
            "plot_role": "proxy",
        },
        {
            "case": "Etna",
            "source_id": "ETNA_SUMMIT_PLUME",
            "source_label": "INGV-PA network, Multi-GAS",
            "observable": "CO2_SO2",
            "observable_label": "Plume CO₂/SO₂ ratio",
            "family": "gas_plume",
            "spatial_type": "source_region_proxy",
            "lat": 37.75,
            "lon": 14.99,
            "plot_role": "source_proxy",
        },
    ]

    return pd.DataFrame(rows)