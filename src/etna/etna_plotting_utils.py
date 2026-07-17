from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from etna_waveform import (
    hourly_teleseismic_feature,
    preprocess_stream_safely,
    teleseismic_rms_windows,
)


THESIS_COLORS = {
    "series": "#0072B2",
    "event": "#D55E00",
    "secondary": "#009E73",
}

ETNA_AXIS_LABELS = {
    "teleseismic": (
        "Teleseismic-band, 0.03–0.30 Hz",
        "RMS ground velocity (m s⁻¹)",
    ),
    "local_event_rate_state": (
        "Past 48-hour local event-rate state",
        "log1p count",
    ),
    "local_event_rate_response": (
        "Catalogue local event-rate response (Effect)",
        "log1p-count excess",
    ),
    "CO2_3": ("Soil CO₂ concentration", "%"),
    "rainfall_mm": ("Hourly precipitation", "mm"),
    "pressure_drop": ("Atmospheric pressure drop", "hPa"),
}

ETNA_SEISMIC_COLS = [
    "teleseismic",
    "local_event_rate_state",
    "local_event_rate_response",
]

ETNA_EXTERNAL_COLS = [
    "CO2_3",
    "rainfall_mm",
    "pressure_drop",
]


def set_thesis_style() -> None:
    """Apply the shared thesis figure style."""
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


def _save_figure(
    figure,
    filename: str | None,
    save_dir: str | Path | None = "figures",
    *,
    formats=("pdf", "png"),
    dpi: int = 450,
) -> None:
    if save_dir is None or filename is None:
        return

    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    if isinstance(formats, str):
        formats = (formats,)

    for extension in formats:
        figure.savefig(
            directory / f"{stem}.{extension}",
            bbox_inches="tight",
            pad_inches=0.055,
            dpi=dpi,
            facecolor="white",
        )


def _time_index(dataframe: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(dataframe.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(
            pd.to_datetime(dataframe.index, utc=True)
        )

    for column in ("time", "timestamp"):
        if column in dataframe.columns:
            return pd.DatetimeIndex(
                pd.to_datetime(
                    dataframe[column],
                    utc=True,
                    errors="coerce",
                )
            )

    raise ValueError("Dataframe has no DatetimeIndex, 'time', or 'timestamp'.")


def dataset_health_report(
    dataframe: pd.DataFrame,
    name: str = "Etna dataset",
) -> pd.DataFrame:
    index = _time_index(dataframe)
    valid_time = not index.isna().any()

    deltas = index.to_series().diff().dropna()
    regular_hourly = bool(
        valid_time
        and index.is_monotonic_increasing
        and not index.has_duplicates
        and deltas.eq(pd.Timedelta("1h")).all()
    )

    return pd.DataFrame([{
        "dataset": name,
        "rows": int(len(dataframe)),
        "variables": int(
            dataframe.shape[1]
            - int("time" in dataframe.columns)
            - int("timestamp" in dataframe.columns)
        ),
        "start": index.min() if valid_time and len(index) else pd.NaT,
        "end": index.max() if valid_time and len(index) else pd.NaT,
        "duplicate_timestamps": int(index.duplicated().sum()),
        "time_sorted": bool(index.is_monotonic_increasing),
        "regular_hourly_grid": regular_hourly,
        "missing_cells": int(dataframe.isna().sum().sum()),
    }])


def distribution_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in dataframe.columns
        if column not in {"time", "timestamp", "station"}
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    summary = dataframe[columns].agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
        "skew",
    ]).T

    summary["n_unique"] = dataframe[columns].nunique()
    summary["missing_fraction"] = dataframe[columns].isna().mean()

    return summary.sort_index()



ETNA_LOGLOG_COLS = [
    "teleseismic",
    "local_event_rate_state",
    "local_event_rate_response",
    "CO2_3",
    "rainfall_mm",
    "pressure_drop",
]


def plot_loglog_distributions(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    axis_labels: dict | None = None,
    bins: int = 35,
    filename: str | None = None,
    save_dir: str | Path | None = "figures",
    formats=("pdf", "png"),
    title: str | None = None,
    ncols: int | None = None,
    min_positive_count: int = 5,
):
    """
    Plot log-binned empirical densities on log-log axes.

    True logarithmic axes require strictly positive observations. Non-positive
    values are excluded separately for each variable and counted in the
    returned report; they are never shifted or transformed for this diagnostic.
    """
    set_thesis_style()
    axis_labels = ETNA_AXIS_LABELS if axis_labels is None else axis_labels

    if columns is None:
        columns = [
            column
            for column in dataframe.columns
            if column not in {"time", "timestamp", "station"}
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

    columns = [column for column in columns if column in dataframe.columns]
    if not columns:
        raise ValueError("No requested numeric columns are available for log-log plotting.")

    if ncols is None:
        ncols = len(columns) if len(columns) <= 3 else 3
    nrows = int(np.ceil(len(columns) / ncols))

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.9 * ncols, 2.85 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()
    report_rows = []

    for axis, column in zip(axes, columns):
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        positive = values[values > 0]

        n_total = int(len(values))
        n_positive = int(len(positive))
        n_nonpositive = int((values <= 0).sum())
        n_unique_positive = int(positive.nunique())

        report_rows.append({
            "variable": column,
            "n_total_nonmissing": n_total,
            "n_positive_used": n_positive,
            "n_nonpositive_excluded": n_nonpositive,
            "positive_fraction": n_positive / n_total if n_total else np.nan,
            "min_positive": float(positive.min()) if n_positive else np.nan,
            "max_positive": float(positive.max()) if n_positive else np.nan,
            "n_unique_positive": n_unique_positive,
        })

        panel_title, unit_label = axis_labels.get(column, (column, "Value"))
        if n_positive < min_positive_count or n_unique_positive < 2:
            axis.axis("off")
            axis.text(
                0.02,
                0.72,
                f"{panel_title}\n\n"
                "Not enough positive observations for log-log axes\n"
                f"positive n = {n_positive}\n"
                f"unique positive = {n_unique_positive}\n"
                f"non-positive excluded = {n_nonpositive}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            continue

        edges = np.geomspace(float(positive.min()), float(positive.max()), bins + 1)
        density, edges = np.histogram(positive, bins=edges, density=True)
        centres = np.sqrt(edges[:-1] * edges[1:])
        valid = density > 0

        axis.plot(
            centres[valid],
            density[valid],
            marker="o",
            markersize=2.8,
            linewidth=0.80,
            color=THESIS_COLORS["series"],
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(panel_title, loc="left", pad=4)
        axis.set_xlabel(unit_label)
        axis.set_ylabel("Density")
        axis.grid(True, which="both", alpha=0.14, linewidth=0.42)
        axis.text(
            0.98,
            0.95,
            f"positive n={n_positive}\n≤0 excluded={n_nonpositive}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color="0.30",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.4,
            },
        )

    for axis in axes[len(columns):]:
        axis.axis("off")

    if title:
        figure.suptitle(title, y=1.01, fontsize=11.0, fontweight="bold")

    figure.tight_layout()
    _save_figure(
        figure,
        filename=filename,
        save_dir=save_dir,
        formats=formats,
    )
    plt.show()

    return figure, axes[:len(columns)], pd.DataFrame(report_rows)


def plot_etna_loglog_distributions(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    bins: int = 35,
    filename: str = "etna_loglog_raw_variables",
    save_dir: str | Path | None = "figures",
):
    """Plot positive-subset log-log densities for Etna variables."""
    if columns is None:
        columns = [column for column in ETNA_LOGLOG_COLS if column in dataframe.columns]

    return plot_loglog_distributions(
        dataframe=dataframe,
        columns=columns,
        axis_labels=ETNA_AXIS_LABELS,
        bins=bins,
        filename=filename,
        save_dir=save_dir,
    )

def _event_timestamp(value) -> pd.Timestamp:
    if isinstance(value, UTCDateTime):
        return pd.Timestamp(value.datetime, tz="UTC")
    return pd.to_datetime(value, utc=True)


def run_teleseismic_checks(
    client,
    station: str,
    config: dict,
    *,
    event_time,
    pre_sec: int = 3600,
    post_sec: int = 3600,
    save_dir: str | Path | None = "figures",
    spectrogram_cmap: str = "cividis",
) -> dict:
    """
    Validate the exact teleseismic feature used in the canonical dataset.

    The diagnostic uses the same response correction, 0.03–0.30 Hz band,
    non-overlapping RMS window, and hourly-maximum aggregation as the data
    construction pipeline.
    """
    set_thesis_style()

    event = _event_timestamp(event_time)
    display_start = event - pd.Timedelta(seconds=pre_sec)
    display_end = event + pd.Timedelta(seconds=post_sec)
    padding = pd.Timedelta(seconds=int(config.get("pad_sec", 0)))

    fetch_start = display_start.floor("h") - padding
    fetch_end = display_end.ceil("h") + padding

    stream = client.get_waveforms(
        network=config["network"],
        station=station,
        location=config["location"],
        channel=config["channel"],
        starttime=UTCDateTime(fetch_start.to_pydatetime()),
        endtime=UTCDateTime(fetch_end.to_pydatetime()),
        attach_response=True,
    )

    trace = preprocess_stream_safely(stream, config)

    rms_windows = teleseismic_rms_windows(trace, config).loc[
        display_start:display_end
    ]
    hourly_feature = hourly_teleseismic_feature(trace, config).loc[
        display_start.floor("h"):display_end.ceil("h")
    ]

    display_trace = trace.copy()
    display_trace.trim(
        UTCDateTime(display_start.to_pydatetime()),
        UTCDateTime(display_end.to_pydatetime()),
    )

    event_mpl = mdates.date2num(event.to_pydatetime())
    event_label = event.strftime("%Y-%m-%d %H:%M UTC")

    response_output = str(config.get("response_output", "VEL")).upper()
    motion_label = {
        "VEL": "Response-corrected ground velocity (m s⁻¹)",
        "DISP": "Response-corrected ground displacement (m)",
        "ACC": "Response-corrected ground acceleration (m s⁻²)",
    }.get(response_output, "Response-corrected ground motion")

    def format_time_axis(axis) -> None:
        axis.axvline(
            event_mpl,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=1.0,
            label="Wenchuan earthquake",
            zorder=5,
        )
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(locator)
        )
        axis.set_xlabel("Time (UTC)")

    station_slug = station.lower()

    figure, axis = plt.subplots(figsize=(12.0, 3.6))
    axis.plot(
        display_trace.times("matplotlib"),
        display_trace.data,
        color=THESIS_COLORS["series"],
        linewidth=0.65,
        rasterized=True,
    )
    format_time_axis(axis)
    axis.set_ylabel(motion_label)
    axis.set_title(
        f"{station}: response-corrected waveform around Wenchuan",
        loc="left",
        pad=6,
    )
    axis.text(
        0.99,
        0.96,
        event_label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.25",
    )
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    _save_figure(
        figure,
        f"etna_{station_slug}_waveform",
        save_dir,
    )
    plt.show()

    window_seconds = int(config["windows_sec"]["teleseismic"])

    figure, axis = plt.subplots(figsize=(12.0, 3.8))
    axis.plot(
        rms_windows.index,
        rms_windows,
        color=THESIS_COLORS["series"],
        linewidth=0.9,
        alpha=0.75,
        label=f"{window_seconds} s RMS",
    )
    axis.plot(
        hourly_feature.index,
        hourly_feature,
        color=THESIS_COLORS["secondary"],
        marker="o",
        markersize=4.2,
        linewidth=1.25,
        label=f"Hourly maximum of {window_seconds} s RMS",
    )
    format_time_axis(axis)
    axis.set_ylabel("RMS ground velocity")
    axis.set_title(
        f"{station}: exact teleseismic-feature aggregation",
        loc="left",
        pad=6,
    )
    axis.legend(loc="upper left", frameon=False, ncols=2)
    figure.tight_layout()
    _save_figure(
        figure,
        f"etna_{station_slug}_teleseismic_aggregation",
        save_dir,
    )
    plt.show()

    from matplotlib import mlab

    sampling_rate = float(display_trace.stats.sampling_rate)
    power, frequencies, bins = mlab.specgram(
        display_trace.data,
        NFFT=2048,
        Fs=sampling_rate,
        noverlap=1536,
    )

    start = display_trace.stats.starttime.datetime
    bin_times = mdates.date2num([
        start + pd.Timedelta(seconds=float(offset))
        for offset in bins
    ])
    power_db = 10 * np.log10(power + np.finfo(float).eps)
    lower, upper = np.nanpercentile(power_db, [2, 98])

    figure, axis = plt.subplots(figsize=(12.0, 5.1))
    mesh = axis.pcolormesh(
        bin_times,
        frequencies,
        power_db,
        shading="auto",
        cmap=spectrogram_cmap,
        vmin=lower,
        vmax=upper,
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.018)
    colorbar.set_label("Power spectral density (dB)")

    format_time_axis(axis)

    band_low, band_high = config["bands"]["teleseismic"]
    axis.axhspan(
        band_low,
        band_high,
        color="white",
        alpha=0.16,
        zorder=7,
    )
    for frequency in (band_low, band_high):
        axis.axhline(
            frequency,
            color="black",
            linestyle="--",
            linewidth=1.0,
            zorder=9,
        )

    axis.set_ylim(0, 15)
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(
        f"{station}: spectrogram around Wenchuan",
        loc="left",
        pad=6,
    )
    axis.text(
        0.015,
        0.055,
        f"Teleseismic proxy band: {band_low:.2f}–{band_high:.2f} Hz",
        transform=axis.transAxes,
        fontsize=8.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.35,
            "alpha": 0.88,
            "pad": 2.2,
        },
        zorder=20,
    )
    figure.tight_layout()
    _save_figure(
        figure,
        f"etna_{station_slug}_spectrogram",
        save_dir,
    )
    plt.show()

    return {
        "station": station,
        "trace": display_trace,
        "rms_windows": rms_windows,
        "hourly_feature": hourly_feature,
    }


def _prepare_etna_dataframe(
    csv_path: str | Path,
    event_time,
    time_window=None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    dataframe = pd.read_csv(csv_path, parse_dates=["time"])
    dataframe["time"] = pd.to_datetime(dataframe["time"], utc=True)
    dataframe = dataframe.set_index("time").sort_index()

    event = _event_timestamp(event_time)

    if time_window is not None:
        start = pd.to_datetime(time_window[0], utc=True)
        end = pd.to_datetime(time_window[1], utc=True)
        dataframe = dataframe.loc[start:end]

    return dataframe, event


def _plot_etna_group(
    csv_path,
    event_time,
    columns,
    filename,
    *,
    save_dir="figures",
    title=None,
    figure_width=8.0,
    panel_height=1.30,
    tick_interval_days=4,
    line_width=0.70,
):
    set_thesis_style()
    dataframe, event = _prepare_etna_dataframe(csv_path, event_time)

    columns = [column for column in columns if column in dataframe.columns]
    if not columns:
        raise ValueError("None of the requested columns are present.")

    figure_height = panel_height * len(columns) + (
        0.50 if title is None else 0.82
    )

    figure, axes = plt.subplots(
        len(columns),
        1,
        figsize=(figure_width, figure_height),
        sharex=True,
    )

    if len(columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, columns):
        values = pd.to_numeric(dataframe[column], errors="coerce")
        panel_title, ylabel = ETNA_AXIS_LABELS.get(
            column,
            (column, "Value"),
        )

        axis.plot(
            dataframe.index,
            values,
            color=THESIS_COLORS["series"],
            linewidth=line_width,
        )
        axis.axvline(
            event,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=0.75,
            zorder=5,
        )
        axis.set_title(panel_title, loc="left", pad=4.5)
        axis.set_ylabel(ylabel, labelpad=7, fontsize=8.8)
        axis.tick_params(
            axis="both",
            which="major",
            length=2.8,
            width=0.55,
            pad=3,
        )
        axis.margins(x=0.01)

        finite = values.to_numpy(dtype=float)
        if (
            np.isfinite(finite).any()
            and np.nanmin(finite) <= 0 <= np.nanmax(finite)
        ):
            axis.axhline(
                0,
                color="0.25",
                linewidth=0.35,
                alpha=0.12,
                zorder=0,
            )

    if title is not None:
        figure.suptitle(title, fontsize=11.0, fontweight="bold", y=0.995)
        top = 0.925
    else:
        top = 0.985

    locator = mdates.DayLocator(interval=tick_interval_days)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d", tz=mdates.UTC)
    )
    axes[-1].set_xlabel("Time (UTC)", labelpad=6)

    figure.align_ylabels(axes)
    figure.subplots_adjust(
        left=0.112,
        right=0.992,
        bottom=0.095,
        top=top,
        hspace=0.64,
    )

    _save_figure(figure, filename, save_dir)
    plt.show()

    return figure, axes


def plot_etna_thesis_figures(
    csv_path,
    event_time,
    *,
    save_dir="figures",
    include_titles=False,
):
    seismic_title = (
        "Teleseismic and catalogue-seismicity variables"
        if include_titles
        else None
    )
    external_title = (
        "External variables"
        if include_titles
        else None
    )

    seismic = _plot_etna_group(
        csv_path,
        event_time,
        ETNA_SEISMIC_COLS,
        "etna_seismic_variables",
        save_dir=save_dir,
        title=seismic_title,
        panel_height=1.42,
        line_width=0.75,
    )

    external = _plot_etna_group(
        csv_path,
        event_time,
        ETNA_EXTERNAL_COLS,
        "etna_external_variables",
        save_dir=save_dir,
        title=external_title,
        panel_height=1.18,
        line_width=0.70,
    )

    return seismic, external


FAMILY_STYLE = {
    "seismic": {
        "marker": "^",
        "color": "#1f77b4",
        "label": "Teleseismic station",
        "size": 115,
    },
    "catalogue_seismicity": {
        "marker": "*",
        "color": "#d62728",
        "label": "Catalogue seismicity",
        "size": 190,
    },
    "gas_meteo": {
        "marker": "o",
        "color": "#2ca02c",
        "label": "ETNAGAS gas/meteo",
        "size": 120,
    },
    "weather_proxy": {
        "marker": "P",
        "color": "#bcbd22",
        "label": "Hourly precipitation",
        "size": 135,
    },
    "summit": {
        "marker": "X",
        "color": "black",
        "label": "Etna summit",
        "size": 135,
    },
}

SOURCE_LABELS = {
    "ESLN": "ESLN",
    "ETNAGAS_3": "ETNAGAS network",
    "ETNA_OPENMETEO_PROXY": "Open-Meteo",
    "EtnaSC_2000_2010": "EtnaSC catalogue",
    "Etna summit": "Etna summit",
}

LABEL_OFFSETS = {
    "ESLN": (0, -18),
    "EtnaSC_2000_2010": (-8, 18),
    "ETNA_OPENMETEO_PROXY": (-12, -18),
    "Etna summit": (16, -12),
    "ETNAGAS_3": (0, 34),
}

DISPLAY_NUDGES_M = {
    "EtnaSC_2000_2010": (-650, 430),
    "ETNA_OPENMETEO_PROXY": (-650, -430),
    "Etna summit": (520, -260),
}


def _source_variable_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source_id, group in metadata.groupby("source_id", sort=False):
        labels = list(dict.fromkeys(
            group["observable_label"].astype(str).tolist()
        ))
        names = list(dict.fromkeys(
            group["observable"].astype(str).tolist()
        ))

        rows.append({
            "Source": SOURCE_LABELS.get(source_id, source_id),
            "Source ID": source_id,
            "Variables": "; ".join(labels),
            "Internal names": ", ".join(names),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Source")
        .reset_index(drop=True)
    )


def _collapse_sources(metadata: pd.DataFrame) -> pd.DataFrame:
    family_by_source = {
        "ESLN": "seismic",
        "EtnaSC_2000_2010": "catalogue_seismicity",
        "ETNAGAS_3": "gas_meteo",
        "ETNA_OPENMETEO_PROXY": "weather_proxy",
        "Etna summit": "summit",
    }

    rows = []
    for source_id, group in metadata.groupby("source_id", sort=False):
        row = group.iloc[0].copy()
        row["family"] = family_by_source.get(
            source_id,
            row["family"],
        )
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def _project_sources(dataframe: pd.DataFrame) -> pd.DataFrame:
    try:
        import geopandas as gpd

        geodata = gpd.GeoDataFrame(
            dataframe.copy(),
            geometry=gpd.points_from_xy(
                dataframe["lon"],
                dataframe["lat"],
            ),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")

        projected = pd.DataFrame(
            geodata.drop(columns="geometry")
        )
        projected["x"] = geodata.geometry.x
        projected["y"] = geodata.geometry.y
        projected["projected"] = True
        return projected

    except Exception as exc:
        warnings.warn(
            f"Using longitude/latitude fallback because projection failed: {exc}"
        )
        fallback = dataframe.copy()
        fallback["x"] = fallback["lon"]
        fallback["y"] = fallback["lat"]
        fallback["projected"] = False
        return fallback


def plot_etna_all_variables_map(
    metadata: pd.DataFrame,
    *,
    summit_lat: float = 37.748,
    summit_lon: float = 14.999,
    satellite: bool = True,
    title: str | None = None,
    figsize=(9.4, 7.2),
    filename: str = "etna_map",
    save_dir: str | Path | None = "figures",
):
    """Plot one marker per retained Etna measurement source."""
    metadata = metadata.copy()
    metadata["lat"] = pd.to_numeric(metadata["lat"], errors="coerce")
    metadata["lon"] = pd.to_numeric(metadata["lon"], errors="coerce")
    metadata = metadata.dropna(subset=["lat", "lon"])

    variable_table = _source_variable_table(metadata)

    summit = pd.DataFrame([{
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

    sources = pd.concat([metadata, summit], ignore_index=True)
    sources = _collapse_sources(sources)
    sources = _project_sources(sources)
    sources["x_plot"] = sources["x"]
    sources["y_plot"] = sources["y"]

    projected = bool(sources["projected"].iloc[0])

    for source_id, (dx_m, dy_m) in DISPLAY_NUDGES_M.items():
        mask = sources["source_id"] == source_id

        if projected:
            dx, dy = dx_m, dy_m
        else:
            dx, dy = dx_m / 111_200, dy_m / 111_200

        sources.loc[mask, "x_plot"] += dx
        sources.loc[mask, "y_plot"] += dy

    figure, axis = plt.subplots(figsize=figsize)

    xs = pd.concat([sources["x"], sources["x_plot"]])
    ys = pd.concat([sources["y"], sources["y_plot"]])
    minimum_padding = 1200 if projected else 1200 / 111_200
    x_padding = max((xs.max() - xs.min()) * 0.065, minimum_padding)
    y_padding = max((ys.max() - ys.min()) * 0.065, minimum_padding)

    axis.set_xlim(xs.min() - x_padding, xs.max() + x_padding)
    axis.set_ylim(ys.min() - y_padding, ys.max() + y_padding)

    if projected:
        try:
            import contextily as cx

            source = (
                cx.providers.Esri.WorldImagery
                if satellite
                else cx.providers.Esri.WorldTopoMap
            )
            cx.add_basemap(
                axis,
                crs="EPSG:3857",
                source=source,
                reset_extent=False,
                attribution_size=4,
            )
        except Exception as exc:
            warnings.warn(f"Could not add basemap: {exc}")

    for _, row in sources.iterrows():
        if row["x_plot"] != row["x"] or row["y_plot"] != row["y"]:
            axis.plot(
                [row["x"], row["x_plot"]],
                [row["y"], row["y_plot"]],
                color="white" if satellite else "black",
                linewidth=0.4,
                alpha=0.4,
                zorder=6,
            )

        style = FAMILY_STYLE[row["family"]]
        axis.scatter(
            row["x_plot"],
            row["y_plot"],
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.85,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

        dx, dy = LABEL_OFFSETS.get(row["source_id"], (12, 12))
        axis.annotate(
            SOURCE_LABELS.get(row["source_id"], row["source_id"]),
            xy=(row["x_plot"], row["y_plot"]),
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
                "color": "white" if satellite else "black",
                "linewidth": 0.85,
                "alpha": 0.90,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=45,
        )

    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    preferred = [
        "Teleseismic station",
        "Catalogue seismicity",
        "ETNAGAS gas/meteo",
        "Hourly precipitation",
        "Etna summit",
    ]
    legend_labels = [label for label in preferred if label in unique]

    axis.legend(
        [unique[label] for label in legend_labels],
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=True,
        framealpha=0.97,
        fontsize=8.0,
    )

    if title is not None:
        axis.set_title(title, fontsize=12, pad=8)

    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    figure.tight_layout(rect=[0, 0.085, 1, 1])

    _save_figure(figure, filename, save_dir)

    return figure, axis, variable_table
