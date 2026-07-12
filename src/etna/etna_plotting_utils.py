import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from pathlib import Path

def _save_current_figure(fig, filename, save_dir="figures", formats=("pdf", "png"), dpi=450):
    """Save every figure as PDF and PNG under figures/ by default."""
    if save_dir is None or filename is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    if isinstance(formats, str):
        formats = (formats,)

    for fmt in formats:
        fig.savefig(
            out_dir / f"{stem}.{fmt}",
            bbox_inches="tight",
            pad_inches=0.055,
            dpi=dpi,
            facecolor="white",
        )


THESIS_COLORS = {
    "series": "#0072B2",      
    "event": "#D55E00",       
    "secondary": "#009E73",   
    "grid": "0.85",
}

ETNA_AXIS_LABELS = {
    "teleseismic": (
        "Teleseismic energy, 0.03–0.30 Hz",
        "RMS velocity"
    ),
    "local_event_rate_state": (
        "Past 48-hour local event-rate state",
        "log1p count"
    ),

    "local_event_rate_response": (
        "Catalogue local event-rate response (Effect)",
        "log1p-count excess"
    ),
    "CO2_3": ("Soil CO₂ concentration", "%"),
    "rainfall_mm": ("Hourly precipitation", "mm"),
    "pressure_drop": ("Atmospheric pressure drop", "hPa"),
    "WindSpeed": ("Wind speed", "m s⁻¹"),
}

ETNA_RAW_ORDER = [
    "teleseismic",
    "local_event_rate_state",
    "local_event_rate_response",
    "CO2_3",
    "rainfall_mm",
    "pressure_drop",
    "WindSpeed",
]

ETNA_EXTERNAL_COLS = [
    "CO2_3",
    "rainfall_mm",
    "pressure_drop",
    "WindSpeed",
]

ETNA_SEISMIC_COLS = [
    "teleseismic",
    "local_event_rate_state",
    "local_event_rate_response",
]


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


def plot_variable_pdfs(df, cols=None, bins=40, save_dir="figures", filename=None):
    """
    Plot empirical probability density functions for numeric variables.
    Uses histograms normalized to density.
    """
    set_thesis_style()
    
    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["time", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    n = len(cols)
    if n <= 4:
        ncols = n
    elif n <= 8:
        ncols = 4
    else:
        ncols = 3

    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 2.85 * nrows),
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

        panel_title, _ = ETNA_AXIS_LABELS.get(col, (col, "Value"))
        ax.set_title(panel_title, loc="left")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    _save_current_figure(fig, filename, save_dir)

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
    save_dir="figures",
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

    def _save_teleseismic_figure(fig, stem):
        station_lower = station.lower()
        _save_current_figure(
            fig,
            filename=f"etna_{station_lower}_{stem}",
            save_dir=save_dir,
        )

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
    _save_teleseismic_figure(fig, "waveform")
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
    _save_teleseismic_figure(fig, "rms_aggregation")
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

    band_low, band_high = cfg["bands"]["teleseismic"]

    # Lightly shade the final teleseismic proxy band instead of writing tiny
    # labels at the bottom of the spectrogram.
    ax.axhspan(
        band_low,
        band_high,
        color="white",
        alpha=0.16,
        zorder=7,
    )

    for freq in (band_low, band_high):
        ax.axhline(
            freq,
            color="black",
            linestyle="--",
            linewidth=1.05,
            alpha=0.95,
            zorder=9,
        )

    ax.text(
        0.015,
        0.055,
        f"Teleseismic proxy band: {band_low:.2f}–{band_high:.2f} Hz",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.35,
            "alpha": 0.88,
            "pad": 2.2,
        },
        zorder=20,
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
            label="Teleseismic proxy band",
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

    _save_teleseismic_figure(fig, "spectro")

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

def plot_etna_catalogue_counts(
    catalog_df,
    event_time,
    *,
    time_window=None,
    filename="etna_catalogue_hourly_counts",
    save_dir="figures",
    fig_width=8.0,
    fig_height=2.6,
):
    """
    Plot raw hourly catalogue event counts around the Etna case window.
    This is a diagnostic figure for the catalogue-derived effect target.
    """
    set_thesis_style()

    events = catalog_df.copy()

    if "timestamp" not in events.columns:
        if "time" in events.columns:
            events = events.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("catalog_df must contain 'timestamp' or 'time'.")

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

    if isinstance(event_time, UTCDateTime):
        event_time = pd.Timestamp(event_time.datetime, tz="UTC")
    else:
        event_time = pd.to_datetime(event_time, utc=True)

    if time_window is None:
        start = event_time - pd.Timedelta(days=3)
        end = event_time + pd.Timedelta(hours=36)
    else:
        start = pd.to_datetime(time_window[0], utc=True)
        end = pd.to_datetime(time_window[1], utc=True)

    hourly = (
        events
        .set_index("timestamp")
        .assign(count=1)
        ["count"]
        .resample("1h")
        .sum()
        .loc[start:end]
    )

    full_index = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    hourly = hourly.reindex(full_index, fill_value=0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.bar(
        hourly.index,
        hourly.values,
        width=0.035,
        color=THESIS_COLORS["series"],
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )

    ax.axvline(
        event_time,
        color=THESIS_COLORS["event"],
        linestyle="--",
        linewidth=0.85,
        alpha=0.90,
        label="Wenchuan earthquake",
    )

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator, tz=mdates.UTC)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title("Catalogue local Etna events", loc="left", pad=5)
    ax.set_ylabel("Events per hour")
    ax.set_xlabel("Time (UTC)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.14, linewidth=0.42)

    fig.tight_layout()

    _save_current_figure(
        fig,
        filename=filename,
        save_dir=save_dir,
        formats=("pdf", "png"),
    )

    plt.show()
    return fig, ax, hourly

# -----------------------------------------------------------------------------
# Clean thesis overview plots
# -----------------------------------------------------------------------------

def set_thesis_style():
    """Style for A4 thesis figures."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 450,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.titlesize": 10.7,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.60,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.42,
        "lines.linewidth": 0.80,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _prepare_etna_dataframe(csv_path, event_time, time_window=None):
    df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if isinstance(event_time, UTCDateTime):
        event_time = pd.Timestamp(event_time.datetime, tz="UTC")
    else:
        event_time = pd.to_datetime(event_time, utc=True)

    if time_window is not None:
        start = pd.to_datetime(time_window[0], utc=True)
        end = pd.to_datetime(time_window[1], utc=True)
        df = df.loc[start:end]

    if "station" in df.columns:
        df = df.drop(columns=["station"])

    return df, event_time


def _plot_etna_group(
    csv_path,
    event_time,
    cols,
    filename,
    save_dir="figures",
    title=None,
    fig_width=8.0,
    panel_height=1.30,
    tick_interval_days=4,
    line_width=0.70,
    formats=("pdf", "png"),
    time_window=None,
):
    """Internal helper used by plot_etna_thesis_figures()."""
    set_thesis_style()

    df, event_time = _prepare_etna_dataframe(
        csv_path,
        event_time,
        time_window=time_window,
    )

    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        raise ValueError("None of the requested columns are present in the dataframe.")

    fig_height = panel_height * len(cols) + (0.50 if title is None else 0.82)

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        y = pd.to_numeric(df[col], errors="coerce")
        panel_title, ylabel = ETNA_AXIS_LABELS.get(col, (col, "Value"))

        ax.plot(
            df.index,
            y,
            color=THESIS_COLORS["series"],
            linewidth=line_width,
            alpha=0.98,
            antialiased=True,
        )

        ax.axvline(
            event_time,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=0.75,
            alpha=0.82,
            zorder=5,
        )

        ax.set_title(panel_title, loc="left", pad=4.5)
        ax.set_ylabel(ylabel, labelpad=7, fontsize=8.8)
        ax.yaxis.set_label_coords(-0.060, 0.5)

        ax.grid(True, alpha=0.14, linewidth=0.42)
        ax.tick_params(axis="both", which="major", length=2.8, width=0.55, pad=3)
        ax.margins(x=0.01)

        y_values = y.to_numpy(dtype=float)
        if np.isfinite(y_values).any() and np.nanmin(y_values) <= 0 <= np.nanmax(y_values):
            ax.axhline(0, color="0.25", linewidth=0.35, alpha=0.12, zorder=0)

    if title is not None:
        fig.suptitle(title, fontsize=11.0, fontweight="bold", y=0.995)
        top = 0.925
    else:
        top = 0.985

    if time_window is None:
        locator = mdates.DayLocator(interval=tick_interval_days)
        formatter = mdates.DateFormatter("%b %d", tz=mdates.UTC)
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator, tz=mdates.UTC)

    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Time (UTC)", labelpad=6)

    fig.align_ylabels(axes)

    fig.subplots_adjust(
        left=0.112,
        right=0.992,
        bottom=0.095,
        top=top,
        hspace=0.64,
    )

    _save_current_figure(
        fig,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
    )

    plt.show()
    return fig, axes


def plot_etna_thesis_figures(
    csv_path,
    event_time,
    save_dir="figures",
    include_titles=False,
):
    """
    Create final thesis overview figures:
    1. trigger/state/effect variables
    2. external variables

    Both PDF and PNG are saved under figures/ by default.
    """

    etna_seismic_title = (
        "Teleseismic trigger and catalogue seismicity variables"
        if include_titles else None
    )
    external_title = "External variables" if include_titles else None

    etna_seismic = _plot_etna_group(
        csv_path=csv_path,
        event_time=event_time,
        cols=ETNA_SEISMIC_COLS,
        filename="etna_seismic_variables",
        save_dir=save_dir,
        title=etna_seismic_title,
        fig_width=8.0,
        panel_height=1.42,
        tick_interval_days=4,
        line_width=0.75,
    )

    external = _plot_etna_group(
        csv_path=csv_path,
        event_time=event_time,
        cols=ETNA_EXTERNAL_COLS,
        filename="etna_external_variables",
        save_dir=save_dir,
        title=external_title,
        fig_width=8.0,
        panel_height=1.18,
        tick_interval_days=4,
        line_width=0.70,
    )

    return etna_seismic, external

# -----------------------------------------------------------------------------
# Etna map / geographic plotting utilities
# -----------------------------------------------------------------------------

FAMILY_STYLE = {
    "seismic": {
        "marker": "^",
        "color": "#1f77b4",
        "label": "Teleseismic station",
    },
    "catalogue_seismicity": {
        "marker": "*",
        "color": "#d62728",
        "label": "Catalogue seismicity",
    },
    "gas_meteo": {
        "marker": "o",
        "color": "#2ca02c",
        "label": "ETNAGAS gas/meteo",
    },
    "weather_proxy": {
        "marker": "P",
        "color": "#bcbd22",
        "label": "Hourly precipitation",
    },
    "summit": {
        "marker": "X",
        "color": "black",
        "label": "Etna summit",
    },
}


SOURCE_LABELS = {
    "ESLN": "ESLN",
    "ETNAGAS_3": "ETNAGAS network",
    "ETNA_OPENMETEO_PROXY": "Open-Meteo",
    "Etna summit": "Etna summit",
    "EtnaSC_2000_2010": "EtnaSC catalogue",
}

DEFAULT_LABEL_OFFSETS = {
    "ESLN": (0, -18),
    "EtnaSC_2000_2010": (-8, 18),
    "ETNA_OPENMETEO_PROXY": (-12, -18),
    "Etna summit": (16, -12),
    "ETNAGAS_3": (0, 34),
}

SOURCE_DISPLAY_NUDGES_M = {
    # Visual-only nudges for the summit/proxy/catalogue cluster.
    # True coordinates remain in x/y; displayed marker positions use x_plot/y_plot.
    "EtnaSC_2000_2010": (-650, 430),
    "ETNA_OPENMETEO_PROXY": (-650, -430),
    "Etna summit": (520, -260),

    # Other sources remain fixed.
    "ESLN": (0, 0),
    "ETNAGAS_3": (0, 0),
}

def _project_to_web_mercator(df):
    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")

        out = pd.DataFrame(gdf.drop(columns="geometry"))
        out["x"] = gdf.geometry.x
        out["y"] = gdf.geometry.y
        out["is_projected"] = True
        return out

    except Exception as exc:
        warnings.warn(f"Using lon/lat fallback because projection failed: {exc}")

        out = df.copy()
        out["x"] = out["lon"]
        out["y"] = out["lat"]
        out["is_projected"] = False
        return out


def _add_basemap(ax, satellite=False):
    try:
        import contextily as cx

        source = (
            cx.providers.Esri.WorldImagery
            if satellite
            else cx.providers.Esri.WorldTopoMap
        )

        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=source,
            reset_extent=False,
            attribution_size=4,
        )

    except Exception as exc:
        warnings.warn(f"Could not add basemap: {exc}")


def _nudge_overlapping_sources(df, source_nudges_m=None):
    """
    Move displayed marker positions for sources that share nearly the same
    coordinates. This keeps the true coordinates unchanged in x/y, but shifts
    x_plot/y_plot so overlapping markers become visible.
    """
    if source_nudges_m is None:
        source_nudges_m = SOURCE_DISPLAY_NUDGES_M

    df = df.copy()
    is_projected = bool(df["is_projected"].iloc[0])

    for source_id, (dx_m, dy_m) in source_nudges_m.items():
        mask = df["source_id"] == source_id

        if not mask.any():
            continue

        if is_projected:
            dx = dx_m
            dy = dy_m
        else:
            dx = dx_m / 111_200
            dy = dy_m / 111_200

        df.loc[mask, "x_plot"] = df.loc[mask, "x_plot"] + dx
        df.loc[mask, "y_plot"] = df.loc[mask, "y_plot"] + dy

    return df

def _set_extent(ax, df, pad_fraction=0.065, min_pad_m=1200):
    xs = pd.concat([df["x"], df["x_plot"]])
    ys = pd.concat([df["y"], df["y_plot"]])

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    xpad = max((xmax - xmin) * pad_fraction, min_pad_m)
    ypad = max((ymax - ymin) * pad_fraction, min_pad_m)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def _clean_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    preferred_order = [
        "Teleseismic station",
        "Catalogue seismicity",
        "ETNAGAS gas/meteo",
        "Hourly precipitation",
        "Etna summit",
    ]

    ordered_labels = [l for l in preferred_order if l in unique]
    ordered_handles = [unique[l] for l in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=True,
        framealpha=0.97,
        fontsize=8.0,
        handletextpad=0.85,
        columnspacing=2.45,
        borderpad=0.85,
        labelspacing=0.85,
        markerscale=1.0,
    )


def _source_centres(df):
    """
    Return one row per measurement source.

    x/y are the true source coordinates.
    x_plot/y_plot are the mean displayed positions of that source's observables.
    Labels should point to x_plot/y_plot so they connect to the visible cluster.
    """
    rows = []

    for source_id, g in df[df["source_id"] != "Etna summit"].groupby("source_id"):
        rows.append({
            "source_id": source_id,
            "x": g["x"].iloc[0],
            "y": g["y"].iloc[0],
            "x_plot": g["x_plot"].mean(),
            "y_plot": g["y_plot"].mean(),
        })

    return pd.DataFrame(rows)

def _annotate_sources(ax, centres, summit_df, satellite=False):
    label_rows = pd.concat([centres, summit_df], ignore_index=True)

    line_color = "white" if satellite else "black"
    line_alpha = 0.95 if satellite else 0.80

    for _, row in label_rows.iterrows():
        source_id = row["source_id"]
        label = SOURCE_LABELS.get(source_id, source_id)
        dx, dy = DEFAULT_LABEL_OFFSETS.get(source_id, (12, 12))

        x_anchor = (
            row["x_plot"]
            if "x_plot" in row.index and pd.notna(row["x_plot"])
            else row["x"]
        )
        y_anchor = (
            row["y_plot"]
            if "y_plot" in row.index and pd.notna(row["y_plot"])
            else row["y"]
        )

        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            ha="center" if abs(dx) < 1 else ("left" if dx > 0 else "right"),
            va="bottom" if dy >= 0 else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.30,
                "alpha": 0.92,
                "pad": 1.20,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": line_color,
                "linewidth": 0.85,
                "alpha": line_alpha,
                "shrinkA": 2,
                "shrinkB": 4,
                "connectionstyle": "arc3,rad=0.0",
            },
            zorder=45,
        )


def _make_source_variable_table(df):
    rows = []

    for source_id, g in df[df["source_id"] != "Etna summit"].groupby("source_id"):
        observables = list(dict.fromkeys(g["observable"].astype(str).tolist()))

        rows.append({
            "Source": SOURCE_LABELS.get(source_id, source_id),
            "Source ID": source_id,
            "Variables": ", ".join(observables),
        })

    return pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)

def _collapse_to_source_markers(df):
    """
    Collapse observable-level metadata to one displayed marker per source.

    The variable table still lists all observables at the source, but the map
    shows source locations rather than one symbol per variable.
    """
    rows = []

    family_by_source = {
        "ESLN": "seismic",
        "EtnaSC_2000_2010": "catalogue_seismicity",
        "ETNAGAS_3": "gas_meteo",
        "ETNA_OPENMETEO_PROXY": "weather_proxy",
        "Etna summit": "summit",
    }

    for source_id, g in df.groupby("source_id", sort=False):
        row = g.iloc[0].copy()

        observables = list(dict.fromkeys(g["observable"].astype(str).tolist()))
        observable_labels = list(dict.fromkeys(g["observable_label"].astype(str).tolist()))

        row["observable"] = ", ".join(observables)
        row["observable_label"] = "; ".join(observable_labels)
        row["family"] = family_by_source.get(source_id, row["family"])

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)

def _sort_sources_for_map(df):
    """Order final Etna map sources without adding map IDs."""
    source_order = [
        "ESLN",
        "EtnaSC_2000_2010",
        "ETNAGAS_3",
        "ETNA_OPENMETEO_PROXY",
        "Etna summit",
    ]

    df = df.copy()
    df["source_order"] = df["source_id"].apply(
        lambda x: source_order.index(x) if x in source_order else 999
    )

    return (
        df.sort_values(["source_order", "source_id", "observable"])
        .drop(columns="source_order")
        .reset_index(drop=True)
    )

def plot_etna_all_variables_map(
    metadata,
    *,
    summit_lat=37.748,
    summit_lon=14.999,
    satellite=True,
    title=None,
    figsize=(9.4, 7.2),
    filename=None,
    save_dir="figures",
):
    df = metadata.copy()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    missing = df[df[["lat", "lon"]].isna().any(axis=1)]

    if not missing.empty:
        print("Rows omitted because lat/lon are missing:")
        print(
            missing[["source_id", "observable", "family", "lat", "lon"]]
            .to_string(index=False)
        )

    df = df.dropna(subset=["lat", "lon"]).copy()

    summit_row = pd.DataFrame([{
        "case": "Etna",
        "source_id": "Etna summit",
        "source_label": "Etna summit reference",
        "observable": "summit_reference",
        "observable_label": "Etna summit reference",
        "family": "summit",
        "spatial_type": "reference_point",
        "lat": summit_lat,
        "lon": summit_lon,
        "plot_role": "reference",
    }])

    df = pd.concat([df, summit_row], ignore_index=True)
    df = _collapse_to_source_markers(df)
    df = _sort_sources_for_map(df)
    df = _project_to_web_mercator(df)
    df["x_plot"] = df["x"]
    df["y_plot"] = df["y"]
    df = _nudge_overlapping_sources(df)

    
    fig, ax = plt.subplots(figsize=figsize)

    _set_extent(ax, df)

    if bool(df["is_projected"].iloc[0]):
        _add_basemap(ax, satellite=satellite)

    # Thin connector lines from offset observable marker to true source centre.
    for _, row in df.iterrows():
        if row["source_id"] == "Etna summit":
            continue

        if row["x_plot"] != row["x"] or row["y_plot"] != row["y"]:
            ax.plot(
                [row["x"], row["x_plot"]],
                [row["y"], row["y_plot"]],
                color="white" if satellite else "black",
                linewidth=0.35,
                alpha=0.30,
                zorder=6,
            )

    centres = _source_centres(df)

    summit_df = df[df["source_id"] == "Etna summit"].drop_duplicates("source_id")

    # Plot all observable markers.
    for _, row in df.iterrows():
        family = row["family"]
        style = FAMILY_STYLE.get(
            family,
            {"marker": "o", "color": "0.5", "label": family},
        )

        size_by_family = {
            "summit": 135,
            "catalogue_seismicity": 190,
            "weather_proxy": 135,
            "gas_meteo": 120,
            "seismic": 115,
        }

        size = size_by_family.get(family, 105)

        ax.scatter(
            row["x_plot"],
            row["y_plot"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.85,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

    # Source names only.
    _annotate_sources(
        ax,
        centres,
        summit_df,
        satellite=satellite,
    )

    _clean_legend(ax)

    if title is not None:
        ax.set_title(title, fontsize=14, pad=10)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fig.tight_layout(rect=[0, 0.085, 1, 1])

    variable_table = _make_source_variable_table(df)

    if filename is None:
        filename = "etna_all_variables_map"
    _save_current_figure(fig, filename, save_dir)

    return fig, ax, variable_table

def plot_loglog_distributions(
    df,
    cols=None,
    *,
    axis_labels=None,
    bins=35,
    filename=None,
    save_dir="figures",
    formats=("pdf", "png"),
    title=None,
    ncols=None,
    fig_width_per_col=3.9,
    panel_height=2.85,
    min_positive_count=5,
):
    """
    Plot log-binned empirical densities on log-log axes for all requested variables.

    Non-positive values cannot appear on true log axes, so they are excluded
    variable-by-variable and reported. Variables with too few positive values
    still get a panel explaining why they were skipped.
    """
    set_thesis_style()

    if axis_labels is None:
        axis_labels = {}

    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["time", "timestamp", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No requested numeric columns available for log-log plotting.")
        return None, None, pd.DataFrame()
    
    if ncols is None:
        if len(cols) <= 4:
            ncols = len(cols)
        elif len(cols) <= 8:
            ncols = 4
        else:
            ncols = 3

    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width_per_col * ncols, panel_height * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    report_rows = []

    for ax, col in zip(axes, cols):
        x_all = pd.to_numeric(df[col], errors="coerce").dropna()
        x_pos = x_all[x_all > 0]

        n_total = int(len(x_all))
        n_positive = int(len(x_pos))
        n_nonpositive = int((x_all <= 0).sum())
        n_unique_positive = int(x_pos.nunique())

        report_rows.append({
            "variable": col,
            "n_total_nonmissing": n_total,
            "n_positive_used": n_positive,
            "n_nonpositive_dropped": n_nonpositive,
            "positive_fraction": n_positive / n_total if n_total else np.nan,
            "min_positive": float(x_pos.min()) if n_positive else np.nan,
            "max_positive": float(x_pos.max()) if n_positive else np.nan,
            "n_unique_positive": n_unique_positive,
        })

        panel_title, unit_label = axis_labels.get(col, (col, "Value"))

        if n_positive < min_positive_count or n_unique_positive < 2:
            ax.axis("off")
            ax.text(
                0.02,
                0.72,
                f"{panel_title}\n\n"
                f"Not enough positive values for log-log axes\n"
                f"positive n = {n_positive}\n"
                f"unique positive = {n_unique_positive}\n"
                f"non-positive dropped = {n_nonpositive}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            continue

        xmin = float(x_pos.min())
        xmax = float(x_pos.max())

        if xmin <= 0 or xmax <= xmin:
            ax.axis("off")
            ax.text(
                0.02,
                0.72,
                f"{panel_title}\n\nSkipped: invalid positive range.",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            continue

        edges = np.geomspace(xmin, xmax, bins + 1)
        counts, edges = np.histogram(x_pos, bins=edges, density=True)

        centres = np.sqrt(edges[:-1] * edges[1:])
        mask = counts > 0

        ax.plot(
            centres[mask],
            counts[mask],
            marker="o",
            markersize=3.0,
            linewidth=0.85,
            color=THESIS_COLORS["series"],
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_title, loc="left")
        ax.set_xlabel(unit_label)
        ax.set_ylabel("Density")
        ax.grid(True, which="both", alpha=0.14, linewidth=0.42)

        ax.text(
            0.98,
            0.95,
            f"positive n={n_positive}\n≤0 dropped={n_nonpositive}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color="0.30",
        )

    for ax in axes[len(cols):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, y=1.01, fontsize=11.0, fontweight="bold")

    plt.tight_layout()

    _save_current_figure(
        fig,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
    )

    plt.show()

    report = pd.DataFrame(report_rows)
    return fig, axes[:len(cols)], report

def plot_etna_loglog_distributions(
    df,
    cols=None,
    *,
    bins=35,
    filename="etna_loglog_all_raw_variables",
    save_dir="figures",
):
    if cols is None:
        cols = [c for c in ETNA_RAW_ORDER if c in df.columns]

    return plot_loglog_distributions(
        df=df,
        cols=cols,
        axis_labels=ETNA_AXIS_LABELS,
        bins=bins,
        filename=filename,
        save_dir=save_dir,
        title="",
    )