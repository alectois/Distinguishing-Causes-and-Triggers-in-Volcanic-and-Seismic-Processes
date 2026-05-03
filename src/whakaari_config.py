import pandas as pd

WHAKAARI_START = "2019-10-01"
WHAKAARI_END = "2019-12-10"

WHAKAARI_ERUPTION_TIME = pd.Timestamp("2019-12-09 01:11", tz="UTC")

TILDE_SUMMARY_URL = "https://tilde.geonet.org.nz/v4/dataSummary/"
TILDE_DATA_URL = "https://tilde.geonet.org.nz/v4/data/"

WHAKAARI_LAT = -37.5167
WHAKAARI_LON = 177.193

WHAKAARI_WAVEFORM_CONFIG = {
    "network": "NZ",
    "station": "WSRZ",
    "location": "10",
    "channel": "HHZ",
    "master_freq": "1h",
    "rms_window_sec": 600,
}

MASTER_FREQ = "1h"