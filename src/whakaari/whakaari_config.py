import pandas as pd

WHAKAARI_START = "2019-10-01"
WHAKAARI_END = "2019-12-10"

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
}

MASTER_FREQ = "1h"

# Local earthquake catalogue settings
WHAKAARI_EQ_RADIUS_KM = 30.0
WHAKAARI_EQ_MIN_MAGNITUDE = None