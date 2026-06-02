import pandas as pd

from whakaari_config import (
    WHAKAARI_LAT,
    WHAKAARI_LON,
    WHAKAARI_EQ_RADIUS_KM,
)

def whakaari_observable_metadata():
    rows = [
        # WSRZ waveform-derived observables
        {
            "case": "Whakaari",
            "source_id": "WSRZ",
            "source_label": "WSRZ seismic station",
            "observable": "hydro_rms_2_5",
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
            "observable_label": "Spectral ratio, 4.5–8 / 8–16 Hz",
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
            "observable": "effect_tremor_rms_5_15",
            "observable_label": "Tremor response RMS, 5–15 Hz",
            "family": "seismic",
            "spatial_type": "point_station",
            "lat": -37.5181,
            "lon": 177.1778,
            "plot_role": "effect",
        },

        # SO2
        {
            "case": "Whakaari",
            "source_id": "WID01",
            "source_label": "WID01 scanning DOAS station",
            "observable": "SO2_flux",
            "observable_label": "SO₂ flux",
            "family": "gas",
            "spatial_type": "point_or_path_measurement",
            "lat": -37.517,
            "lon": 177.193,
            "plot_role": "direct_or_path_measurement",
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
            "observable": "API",
            "observable_label": "Antecedent precipitation index",
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

        # Local earthquake catalogue search area
        {
            "case": "Whakaari",
            "source_id": "WHAKAARI_EQ_RADIUS",
            "source_label": "Local earthquake search radius",
            "observable": "local_eq_count_1h",
            "observable_label": "Local earthquake count, 1h",
            "family": "earthquake_catalogue",
            "spatial_type": "search_radius",
            "lat": WHAKAARI_LAT,
            "lon": WHAKAARI_LON,
            "radius_km": WHAKAARI_EQ_RADIUS_KM,
            "plot_role": "catalogue_radius",
        },
    ]

    return pd.DataFrame(rows)