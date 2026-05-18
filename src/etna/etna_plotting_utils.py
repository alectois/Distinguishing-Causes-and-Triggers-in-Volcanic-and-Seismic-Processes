import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime


# ---------------------------------------------------------------------
# Thesis plotting style
# ---------------------------------------------------------------------

THESIS_COLORS = {
    "series": "#0072B2",      
    "event": "#D55E00",       
    "secondary": "#009E73",   
    "grid": "0.85",
}

ETNA_AXIS_LABELS = {
    "teleseismic_band": ("Teleseismic-band energy, 0.05–0.5 Hz", "RMS"),
    "background_seismic": ("Background seismic energy, 0.5–4 Hz", "RMS"),
    "effect_seismic": ("High-frequency seismic response (Effect), 4–12 Hz", "RMS"),
    "CO2_3": ("Soil CO₂ concentration", "%"),
    "AirTemp_3": ("Air temperature", "°C"),
    "API": ("Antecedent precipitation index", "mm"),
    "pressure_drop": ("Atmospheric pressure drop", "hPa"),
    "WindSpeed": ("Wind speed", "m s⁻¹"),
    "CO2_SO2": ("Plume CO₂/SO₂ ratio", "Molar ratio"),
    
    "teleseismic_band_scaled": ("Teleseismic-band energy, 0.05–0.5 Hz", "RMS, Robust-scaled"),
    "background_seismic_scaled": ("Background seismic energy, 0.5–4 Hz", "RMS, Robust-scaled"),
    "effect_seismic_scaled": ("High-frequency seismic response (Effect), 4–12 Hz", "RMS, Robust-scaled"),
    "CO2_3_scaled": ("Soil CO₂ concentration", "%, Robust-scaled"),
    "AirTemp_3_scaled": ("Air temperature", "°C, Robust-scaled"),
    "API_scaled": ("Antecedent precipitation index", "Robust-scaled"),
    "pressure_drop_scaled": ("Atmospheric pressure drop", "hPa, Robust-scaled"),
    "WindSpeed_scaled": ("Wind speed", "m s⁻¹, Robust-scaled"),
    "CO2_SO2_scaled": ("Plume CO₂/SO₂ ratio", "Molar ratio, Robust-scaled"),
    
}

ETNA_RAW_ORDER = [
    "teleseismic_band",
    "background_seismic",
    "effect_seismic",
    "CO2_3",
    "AirTemp_3",
    "API",
    "pressure_drop",
    "WindSpeed",
    "CO2_SO2",
]

ETNA_SCALED_ORDER = [
    "teleseismic_band_scaled",
    "background_seismic_scaled",
    "effect_seismic_scaled",
    "CO2_3_scaled",
    "AirTemp_3_scaled",
    "API_scaled",
    "pressure_drop_scaled",
    "WindSpeed_scaled",
    "CO2_SO2_scaled",
]


def set_thesis_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
    })


def dataset_health_report(df, name):
    print(f"\n{name}")
    print("shape:", df.shape)

    if "time" in df.columns:
        print("duplicate timestamps:", df["time"].duplicated().sum())
        print("time sorted:", df["time"].is_monotonic_increasing)
    else:
        print("No 'time' column found.")

    print("\nMissing fraction:")
    print(df.isna().mean().sort_values())


def plot_scaled_dataset(df, station_name):
    cols = [c for c in df.columns if c.endswith("_scaled")]

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.5 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df["time"], df[col], linewidth=0.8)
        ax.set_title(f"{station_name} — {col}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_variable_pdfs(df, name, cols=None, bins=40):
    """
    Plot empirical probability density functions for numeric variables.
    Uses histograms normalized to density.
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["time", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    n = len(cols)
    if n == 0:
        print(f"No numeric columns to plot for {name}.")
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna()

        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.axvline(x.mean(), linestyle="--", linewidth=1.5, label="mean")
        ax.axvline(x.median(), linestyle=":", linewidth=1.5, label="median")

        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle(f"Probability density functions — {name}", y=1.02)
    plt.tight_layout()
    plt.show()


def distribution_summary(df, name=None):
    cols = [
        c for c in df.columns
        if c not in ["time", "station"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = df[cols].agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
        "skew",
    ]).T

    summary["n_unique"] = df[cols].nunique()
    summary["missing_fraction"] = df[cols].isna().mean()

    if name is not None:
        print(name)

    return summary.sort_index()

def run_teleseismic_checks(
    client,
    station,
    cfg,
    event_time=UTCDateTime("2008-05-12T06:28:00"),
    pre_sec=3600,
    post_sec=3600,
    save_dir=None,
    spectrogram_cmap="cividis",
):
    """
    Produces three figures:
    1. instrument-response-corrected waveform,
    2. 1-min RMS and hourly maximum RMS comparison,
    3. spectrogram with the frequency bands used in the Etna analysis.
    """
    set_thesis_style()

    if not isinstance(event_time, UTCDateTime):
        event_time = UTCDateTime(event_time)

    t1 = event_time - pre_sec
    t2 = event_time + post_sec

    st = client.get_waveforms(
        network=cfg["network"],
        station=station,
        location=cfg["location"],
        channel=cfg["channel"],
        starttime=t1,
        endtime=t2,
        attach_response=True,
    )

    st.remove_response(
        output=cfg.get("response_output", "VEL"),
        pre_filt=cfg.get("pre_filt", None),
    )
    tr = st[0]

    response_output = cfg.get("response_output", "VEL").upper()

    motion_quantity = {
        "VEL": "ground velocity",
        "DISP": "ground displacement",
        "ACC": "ground acceleration",
    }.get(response_output, "ground-motion amplitude")

    motion_units = {
        "VEL": "m s⁻¹",
        "DISP": "m",
        "ACC": "m s⁻²",
    }.get(response_output, "")

    amplitude_label = (
        f"Response-corrected {motion_quantity} ({motion_units})"
        if motion_units
        else f"Response-corrected {motion_quantity}"
    )

    rms_label = (
        f"60 s RMS {motion_quantity} ({motion_units})"
        if motion_units
        else f"60 s RMS {motion_quantity}"
    )

    event_dt = event_time.datetime
    event_mpl = mdates.date2num(event_dt)
    t1_mpl = mdates.date2num(t1.datetime)
    t2_mpl = mdates.date2num(t2.datetime)
    event_label = event_time.strftime("%Y-%m-%d %H:%M UTC")

    def _format_timestamp_axis(ax, xlabel=True):
        """Apply compact UTC timestamp formatting to a Matplotlib date axis."""
        ax.axvline(
            event_mpl,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=1.15,
            alpha=0.95,
            label="Wenchuan earthquake",
            zorder=5,
        )
        ax.set_xlim(t1_mpl, t2_mpl)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        if xlabel:
            ax.set_xlabel("Time (UTC)")
        ax.grid(True, alpha=0.22, linewidth=0.6)

    def _maybe_save(fig, stem):
        if save_dir is not None:
            from pathlib import Path
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{station}_{stem}.png", bbox_inches="tight", dpi=300)

    # -----------------------------
    # Corrected waveform
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.plot(
        tr.times("matplotlib"),
        tr.data,
        color=THESIS_COLORS["series"],
        linewidth=0.65,
        rasterized=True,
    )
    _format_timestamp_axis(ax)
    ax.grid(False)
    ax.set_ylabel(amplitude_label)
    ax.set_title(
        f"{station}: response-corrected waveform around Wenchuan earthquake",
        loc="left",
        pad=6,
    )
    ax.text(
        0.99,
        0.96,
        event_label,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
    )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _maybe_save(fig, "waveform_teleseismic")
    plt.show()

    # -----------------------------
    # 1-min RMS and hourly aggregation
    # -----------------------------
    df = pd.DataFrame({
        "time": pd.to_datetime(tr.times("timestamp"), unit="s", utc=True),
        "amp": tr.data,
    }).set_index("time")

    rms_1min = (
        df["amp"]
        .pow(2)
        .rolling("60s")
        .mean()
        .pow(0.5)
        .resample("1min")
        .mean()
    )

    rms_hourly = rms_1min.resample("1h").max()

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.plot(
        rms_1min.index,
        rms_1min,
        color=THESIS_COLORS["series"],
        linewidth=0.9,
        alpha=0.72,
        label="1-min RMS",
    )
    ax.plot(
        rms_hourly.index,
        rms_hourly,
        color=THESIS_COLORS["secondary"],
        marker="o",
        markersize=4.2,
        linewidth=1.25,
        label="Hourly maximum of 1-min RMS",
    )
    _format_timestamp_axis(ax)
    ax.set_ylabel(rms_label)
    ax.set_title(
        f"{station}: hourly aggregation of response-corrected ground motion",
        loc="left",
        pad=6,
    )
    ax.legend(loc="upper left", frameon=False, ncols=3)
    fig.tight_layout()
    _maybe_save(fig, "rms_hourly_aggregation")
    plt.show()

    t_peak_1m = rms_1min.idxmax()
    t_peak_hourly = rms_hourly.idxmax()

    print(f"\n{station}")
    print("1-min peak:", t_peak_1m)
    print("hourly peak bin:", t_peak_hourly)
    print("difference:", t_peak_1m - t_peak_hourly)

    # -----------------------------
    # Spectrogram
    # -----------------------------
    tr_spec = tr.copy()
    fs = tr_spec.stats.sampling_rate

    from matplotlib import mlab
    Pxx, freqs, bins = mlab.specgram(
        tr_spec.data,
        NFFT=2048,
        Fs=fs,
        noverlap=1536,
    )

    start = tr_spec.stats.starttime.datetime
    bin_datetimes = np.array([start + pd.Timedelta(seconds=b) for b in bins])
    bin_times = mdates.date2num(bin_datetimes)

    Pxx_db = 10 * np.log10(Pxx + np.finfo(float).eps)

    vmin, vmax = np.nanpercentile(Pxx_db, [2, 98])

    fig, ax = plt.subplots(figsize=(12.0, 5.1))
    mesh = ax.pcolormesh(
        bin_times,
        freqs,
        Pxx_db,
        shading="auto",
        cmap=spectrogram_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.018)
    cbar.set_label("Power spectral density (dB)")

    _format_timestamp_axis(ax)

    band_guides = [
        (0.5, "0.5 Hz"),
        (4.0, "4 Hz"),
        (12.0, "12 Hz"),
    ]

    for freq, label in band_guides:
        ax.axhline(
            freq,
            color="white",
            linestyle="-",
            linewidth=2.8,
            alpha=0.95,
            zorder=8,
        )
        ax.axhline(
            freq,
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=0.95,
            zorder=9,
        )

        ax.text(
            0.985,
            freq,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.8,
            },
            zorder=10,
        )

    ax.set_ylim(0, 15)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"{station}: spectrogram around the Wenchuan earthquake",
        loc="left",
        pad=6,
    )
    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=1.15,
            label="Wenchuan earthquake",
        ),
        plt.Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.15,
            label="Frequency-band boundaries",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.90,
        fontsize=8,
    )
    fig.tight_layout()
    _maybe_save(fig, "spectrogram_teleseismic")
    plt.show()

    return {
        "station": station,
        "trace": tr,
        "rms_1min": rms_1min,
        "rms_hourly": rms_hourly,
        "t_peak_1m": t_peak_1m,
        "t_peak_hourly": t_peak_hourly,
        "peak_diff": t_peak_1m - t_peak_hourly,
    }

def plot_etna_data_with_event(
    csv_path,
    station,
    event_time,
    cols=None,
    title=None,
    tick_interval_days=4,
    figsize=None,
):
    set_thesis_style()

    df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")

    if isinstance(event_time, UTCDateTime):
        event_time = pd.Timestamp(event_time.datetime, tz="UTC")
    else:
        event_time = pd.to_datetime(event_time, utc=True)

    if "station" in df.columns:
        df = df.drop(columns=["station"])

    if cols is None:
        cols = [c for c in ETNA_RAW_ORDER if c in df.columns]
    else:
        cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        raise ValueError("No requested columns are present in the dataframe.")

    if figsize is None:
        figsize = (14, 2.05 * len(cols))

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=figsize,
        sharex=True,
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(
            df.index,
            df[col],
            color=THESIS_COLORS["series"],
            linewidth=0.75,
        )

        ax.axvline(
            event_time,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )

        panel_title, ylabel = ETNA_AXIS_LABELS.get(col, (col, "Value"))

        ax.set_title(panel_title, loc="left", fontsize=10, pad=3)
        ax.set_ylabel(ylabel, fontsize=9)

        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    axes[0].legend(
        handles=[
            plt.Line2D(
                [0], [0],
                color=THESIS_COLORS["event"],
                linestyle="--",
                linewidth=1.1,
                label="Wenchuan earthquake",
            )
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    locator = mdates.DayLocator(interval=tick_interval_days)
    formatter = mdates.DateFormatter("%b %d")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel("Time (UTC)", fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    plt.tight_layout()

    plt.show()