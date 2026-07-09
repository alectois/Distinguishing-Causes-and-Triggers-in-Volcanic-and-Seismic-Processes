from obspy import UTCDateTime
import pandas as pd

# configure the processing parameters for the study
ETNA_WAVEFORM_CONFIG = {
    "network": "IV",
    "location": "",
    "channel": "HHZ",
    "stations": ["ESLN"],

    "start": UTCDateTime("2008-04-12T00:00:00"),
    "end": UTCDateTime("2008-05-16T00:00:00"),

    # process in daily chunks, but pad each day for filtering stability
    "chunk_sec": 24 * 3600,
    "pad_sec": 3600,
    "base_freq": "1h",

    # physical bands
    "bands": {
        "teleseismic": (0.03, 0.30),          # teleseismic / Wenchuan wavefield
        "background_seismic": (0.80, 2.30),   # Etna tremor/state band
    },

    # metric windows in seconds
    "windows_sec": {
        "teleseismic": 120,
        "background_seismic": 600,   # slower state
    },

    # remove response to velocity
    "response_output": "VEL",
    "pre_filt": (0.02, 0.03, 30.0, 40.0),
    # gap policy
    "max_interp_gap_sec": 2.0, # only interpolate tiny gaps
}

ETNA_GAS_METEO_COLS = [
    "CO2_3",
    "pressure_drop",
    "WindSpeed",
]
ETNA_WEATHER_COLS = ["rain_6h_sum"]
ETNA_EVENT_TIME = UTCDateTime("2008-05-12T06:28:00")
ETNA_CATALOGUE_FILENAME = "Etna catalogue_2000-2010.xls"

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
            "lat": 37.75,
            "lon": 14.99,
            "plot_role": "candidate_state",
        },
        {
            "case": "Etna",
            "source_id": "EtnaSC_2000_2010",
            "source_label": "INGV-OE Mt. Etna seismic event catalogue, 2000–2010",
            "observable": "local_event_rate_anomaly",
            "observable_label": "Positive local event-rate anomaly",
            "family": "catalogue_seismicity",
            "spatial_type": "event_catalogue",
            "lat": 37.75,
            "lon": 14.99,
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
            "lat": 37.6086,
            "lon": 15.0822,
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
            "lat": 37.6086,
            "lon": 15.0822,
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
            "lat": 37.6086,
            "lon": 15.0822,
            "plot_role": "direct_or_network_measurement",
        },

        # Etna proxy/source observables
        {
            "case": "Etna",
            "source_id": "ETNA_OPENMETEO_PROXY",
            "source_label": "Open-Meteo Etna proxy point",
            "observable": "rain_6h_sum",
            "observable_label": "6-hour rainfall sum",
            "family": "weather_proxy",
            "spatial_type": "proxy_point",
            "lat": 37.75,
            "lon": 14.99,
            "plot_role": "proxy",
        },
    ]

    return pd.DataFrame(rows)