import pandas as pd

WHAKAARI_START = "2019-10-01"
WHAKAARI_END = "2019-12-12"

WHAKAARI_ERUPTION_TIME = pd.Timestamp("2019-12-09 01:11", tz="UTC")

TILDE_SUMMARY_URL = "https://tilde.geonet.org.nz/v4/dataSummary/"
TILDE_DATA_URL = "https://tilde.geonet.org.nz/v4/data/"

WHAKAARI_LAT = -37.52
WHAKAARI_LON = 177.18

WHAKAARI_WAVEFORM_CONFIG = {
    "network": "NZ",
    "station": "WSRZ",
    "location": "10",
    "channel": "HHZ",
    "master_freq": "1h",
    "rms_window_sec": 600,
    "response_output": "VEL",
    "pre_filt": (0.5, 1.0, 20.0, 25.0),
    "max_interp_gap_sec": 2.0,
    "pad_sec": 3600,
}

MASTER_FREQ = "1h"

#whakaari geo-metadata
def whakaari_observable_metadata():
    rows = [
        # WSRZ waveform-derived observables
        {
            "case": "Whakaari",
            "source_id": "WSRZ",
            "source_label": "WSRZ seismic station",
            "observable": "hydro_2_5",
            "observable_label": "Hydrothermal tremor RMS, 2–5 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": -37.5181,
            "lon": 177.1778,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Whakaari",
            "source_id": "WSRZ",
            "source_label": "WSRZ seismic station",
            "observable": "ratio_4p5_8_over_8_16",
            "observable_label": "Past-smoothed spectral ratio, 4.5–8 / 8–16 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": -37.5181,
            "lon": 177.1778,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Whakaari",
            "source_id": "WSRZ",
            "source_label": "WSRZ seismic station",
            "observable": "event_rate_2_5",
            "observable_label": "High-frequency event rate, 2–5 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": -37.5181,
            "lon": 177.1778,
            "plot_role": "direct_measurement",
        },
        {
            "case": "Whakaari",
            "source_id": "WSRZ",
            "source_label": "WSRZ seismic station",
            "observable": "effect_tremor_5_15",
            "observable_label": "Positive tremor-response anomaly, 5–15 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": -37.5181,
            "lon": 177.1778,
            "plot_role": "effect",
        },

        # GNSS pair
        {
            "case": "Whakaari",
            "source_id": "RGWC_RGWI",
            "source_label": "RGWC–RGWI GNSS vertical displacement difference",
            "observable": "GNSS_deformation",
            "observable_label": "GNSS deformation proxy",
            "family": "deformation",
            "spatial_type": "derived_station_pair_midpoint",

            # Display the derived deformation proxy at the midpoint of the pair.
            "lat": (-37.5243 + -37.5181) / 2,
            "lon": (177.1925 + 177.1778) / 2,

            # Keep endpoints for documentation if needed, but the main map will not draw the line.
            "lat1": -37.5243,
            "lon1": 177.1925,
            "lat2": -37.5181,
            "lon2": 177.1778,

            "plot_role": "derived_station_pair",
        },

        # Weather proxy
        {
            "case": "Whakaari",
            "source_id": "WHAKAARI_OPENMETEO_PROXY",
            "source_label": "Whakaari Open-Meteo proxy point",
            "observable": "rain_12h_sum",
            "observable_label": "12-hour rainfall sum",
            "family": "weather_proxy",
            "spatial_type": "proxy_point",
            "lat": WHAKAARI_LAT,
            "lon": WHAKAARI_LON,
            "plot_role": "proxy",
        },
        {
            "case": "Whakaari",
            "source_id": "WHAKAARI_OPENMETEO_PROXY",
            "source_label": "Whakaari Open-Meteo proxy point",
            "observable": "pressure_drop",
            "observable_label": "Atmospheric pressure drop",
            "family": "meteorology",
            "spatial_type": "proxy_point",
            "lat": WHAKAARI_LAT,
            "lon": WHAKAARI_LON,
            "plot_role": "proxy",
        },
    ]

    return pd.DataFrame(rows)