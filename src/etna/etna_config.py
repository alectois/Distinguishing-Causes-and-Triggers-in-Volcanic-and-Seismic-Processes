from obspy import UTCDateTime

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