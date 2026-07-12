import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


# -----------------------------------------------------------------------------
# Time-series variable groups / labels
# -----------------------------------------------------------------------------

WHAKAARI_ALL_COLS = [
    "hydro_2_5",
    "spectral_log_ratio_4p5_8_over_8_16",
    "event_rate_2_5",
    "effect_tremor_5_15",
    "rainfall_mm",
    "pressure_drop",
    "GNSS_deformation_rate",
]

WHAKAARI_SEISMIC_COLS = [
    "hydro_2_5",
    "spectral_log_ratio_4p5_8_over_8_16",
    "event_rate_2_5",
    "effect_tremor_5_15",
]

WHAKAARI_EXTERNAL_COLS = [
    "rainfall_mm",
    "pressure_drop",
    "GNSS_deformation_rate",
]

WHAKAARI_LOG_Y_COLS = {
    "hydro_2_5",
}

WHAKAARI_AXIS_LABELS = {
    "hydro_2_5": (
        "Hydrothermal RMS, 2–5 Hz",
        "RMS velocity"
    ),
    "spectral_log_ratio_4p5_8_over_8_16": (
        "Past-smoothed spectral contrast, 4.5–8 / 8–16 Hz",
        "Log amplitude ratio",
    ),
    "event_rate_2_5": (
        "STA/LTA event rate, 2–5 Hz",
        "Events h⁻¹"
    ),
    "effect_tremor_5_15": (
        "Tremor anomaly (Effect), 5–15 Hz",
        "log-RMS excess"
    ),
    "rainfall_mm": (
        "Hourly precipitation",
        "mm",
    ),
    "pressure_drop": (
        "Atmospheric pressure drop",
        "hPa"
    ),
    "GNSS_deformation_rate": (
        "Lagged daily GNSS deformation change",
        "m day⁻¹",
    ),

    "hydro_2_5_scaled": (
        "Hydrothermal RMS, 2–5 Hz",
        "Std. value"
    ),
    "spectral_log_ratio_4p5_8_over_8_16_scaled": (
        "Past-smoothed spectral contrast, 4.5–8 / 8–16 Hz",
        "Std. value"
    ),
    "event_rate_2_5_scaled": (
        "STA/LTA event rate, 2–5 Hz",
        "Std. value"
    ),
    "effect_tremor_5_15_scaled": (
        "Tremor anomaly (Effect), 5–15 Hz",
        "Std. value"
    ),
    "rainfall_mm_scaled": (
        "Hourly precipitation",
        "Std. value"
    ),
    "pressure_drop_scaled": (
        "Atmospheric pressure drop",
        "Std. value"
    ),
    "GNSS_deformation_rate_scaled": (
        "Lagged daily GNSS deformation change",
        "Std. value"
    ),
}


def dataset_health_report(df, name):
    print(f"\n{name}")
    print("shape:", df.shape)

    if isinstance(df.index, pd.DatetimeIndex):
        print("duplicate timestamps:", df.index.duplicated().sum())
        print("time sorted:", df.index.is_monotonic_increasing)
    elif "timestamp" in df.columns:
        print("duplicate timestamps:", df["timestamp"].duplicated().sum())
        print("time sorted:", df["timestamp"].is_monotonic_increasing)
    elif "time" in df.columns:
        print("duplicate timestamps:", df["time"].duplicated().sum())
        print("time sorted:", df["time"].is_monotonic_increasing)
    else:
        print("No timestamp/time index/column found.")

    print("\nMissing fraction:")
    print(df.isna().mean().sort_values())

    return df.describe()


def _prepare_whakaari_dataframe(csv_path, eruption_time, time_window=None):
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        time_col = "timestamp"
    elif "time" in df.columns:
        time_col = "time"
    else:
        raise ValueError("Input CSV must contain either a 'timestamp' or 'time' column.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()

    eruption_time = pd.to_datetime(eruption_time, utc=True)

    if time_window is not None:
        start = pd.to_datetime(time_window[0], utc=True)
        end = pd.to_datetime(time_window[1], utc=True)
        df = df.loc[start:end]

    if "station" in df.columns:
        df = df.drop(columns=["station"])

    return df, eruption_time


def _plot_whakaari_group(
    csv_path,
    eruption_time,
    cols,
    filename,
    save_dir="figures",
    title=None,
    fig_width=8.0,
    panel_height=1.30,
    tick_interval_days=7,
    line_width=0.70,
    formats=("pdf", "png"),
    time_window=None,
):
    """Internal helper used by plot_whakaari_thesis_figures()."""
    set_thesis_style()

    df, eruption_time = _prepare_whakaari_dataframe(
        csv_path,
        eruption_time,
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
        panel_title, ylabel = WHAKAARI_AXIS_LABELS.get(col, (col, "Value"))

        if col in WHAKAARI_LOG_Y_COLS:
            y = y.where(y > 0)

        drawstyle = "steps-post" if col == "GNSS_deformation_rate" else "default"

        ax.plot(
            df.index,
            y,
            color=THESIS_COLORS["series"],
            linewidth=max(line_width, 0.72) if col == "GNSS_deformation_rate" else line_width,
            drawstyle=drawstyle,
            alpha=0.98,
            antialiased=True,
        )

        ax.axvline(
            eruption_time,
            color=THESIS_COLORS["event"],
            linestyle="--",
            linewidth=0.75,
            alpha=0.82,
            zorder=5,
        )

        if col in WHAKAARI_LOG_Y_COLS:
            ax.set_yscale("log")

        ax.set_title(panel_title, loc="left", pad=4.5)
        ax.set_ylabel(ylabel, labelpad=7)
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
        left=0.120,
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


def plot_whakaari_thesis_figures(
    csv_path,
    eruption_time,
    save_dir="figures",
    include_titles=False,
    tick_interval_days=7,
):
    """
    Create final thesis overview figures:
    1. seismic / hydrothermal variables
    2. external / deformation variables
    """
    seismic_title = "Seismic and hydrothermal variables" if include_titles else None
    external_title = "External and deformation variables" if include_titles else None

    seismic = _plot_whakaari_group(
        csv_path=csv_path,
        eruption_time=eruption_time,
        cols=WHAKAARI_SEISMIC_COLS,
        filename="whakaari_seismic_hydrothermal_variables",
        save_dir=save_dir,
        title=seismic_title,
        fig_width=8.0,
        panel_height=1.42,
        tick_interval_days=tick_interval_days,
        line_width=0.70,
    )

    external = _plot_whakaari_group(
        csv_path=csv_path,
        eruption_time=eruption_time,
        cols=WHAKAARI_EXTERNAL_COLS,
        filename="whakaari_external_deformation_variables",
        save_dir=save_dir,
        title=external_title,
        fig_width=8.0,
        panel_height=1.18,
        tick_interval_days=tick_interval_days,
        line_width=0.70,
    )

    return seismic, external


def plot_variable_pdfs(df, name=None, cols=None, bins=40, filename=None, save_dir="figures"):
    """
    Plot empirical probability density functions for numeric Whakaari variables.
    Uses density-normalized histograms.
    """
    set_thesis_style()

    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["timestamp", "time", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No numeric columns to plot.")
        return None

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

        panel_title, _ = WHAKAARI_AXIS_LABELS.get(col, (col, "Value"))
        ax.set_title(panel_title, loc="left")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.14, linewidth=0.42)
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    if name:
        fig.suptitle(str(name), y=1.01)

    plt.tight_layout()

    if filename is None:
        filename = "whakaari_variable_pdfs"

    _save_current_figure(fig, filename, save_dir)
    plt.show()
    return fig, axes[:len(cols)]


def distribution_summary(df, name=None):
    """Summary statistics useful for checking skewness, tails, constants, and missingness."""
    cols = [
        c for c in df.columns
        if c not in ["timestamp", "time", "station"]
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


# -----------------------------------------------------------------------------
# Whakaari map / geographic plotting utilities
# -----------------------------------------------------------------------------

WHAKAARI_FAMILY_STYLE = {
    "seismic": {
        "marker": "^",
        "color": "#1f77b4",
        "label": "WSRZ seismic station",
    },
    "deformation": {
        "marker": "*",
        "color": "#ff7f0e",
        "label": "GNSS deformation",
    },
    "weather_meteo": {
        "marker": "P",
        "color": "#bcbd22",
        "label": "Open-Meteo weather",
    },
}

WHAKAARI_SOURCE_LABELS = {
    "WSRZ": "WSRZ",
    "RGWC_RGWI": "RGWC–RGWI GNSS",
    "WHAKAARI_OPENMETEO_PROXY": "Open-Meteo",
}

WHAKAARI_DEFAULT_LABEL_OFFSETS = {
    "WSRZ": (-16, -14),
    "RGWC_RGWI": (14, 12),
    "WHAKAARI_OPENMETEO_PROXY": (12, -16),
}

WHAKAARI_SOURCE_DISPLAY_NUDGES_M = {
    "WHAKAARI_OPENMETEO_PROXY": (60, -70),
    "WSRZ": (-55, -35),
    "RGWC_RGWI": (65, 55),
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
    if source_nudges_m is None:
        source_nudges_m = WHAKAARI_SOURCE_DISPLAY_NUDGES_M

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


def _set_extent(ax, df, *, pad_fraction=0.12, min_pad_m=260):
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
        "WSRZ seismic station",
        "GNSS deformation",
        "Open-Meteo weather",
    ]

    ordered_labels = [label for label in preferred_order if label in unique]
    remaining_labels = [label for label in unique if label not in ordered_labels]
    final_labels = ordered_labels + remaining_labels
    final_handles = [unique[label] for label in final_labels]

    ax.legend(
        final_handles,
        final_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=True,
        framealpha=0.97,
        fontsize=8.0,
        handletextpad=0.85,
        columnspacing=1.8,
        borderpad=0.75,
        labelspacing=0.75,
        markerscale=1.0,
    )

def _annotate_sources(ax, df, satellite=False):
    line_color = "white" if satellite else "black"
    line_alpha = 0.95 if satellite else 0.80

    for _, row in df.iterrows():
        source_id = row["source_id"]
        label = WHAKAARI_SOURCE_LABELS.get(source_id, source_id)
        dx, dy = WHAKAARI_DEFAULT_LABEL_OFFSETS.get(source_id, (12, 12))

        ax.annotate(
            label,
            xy=(row["x_plot"], row["y_plot"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
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
                "linewidth": 0.80,
                "alpha": line_alpha,
                "shrinkA": 2,
                "shrinkB": 4,
                "connectionstyle": "arc3,rad=0.0",
            },
            zorder=45,
        )

def _make_source_variable_table(df):
    rows = []

    for source_id, g in df[df["source_id"] != "Whakaari reference"].groupby("source_id"):
        if "observable_label" in g.columns:
            labels = list(dict.fromkeys(g["observable_label"].astype(str).tolist()))
        else:
            labels = list(dict.fromkeys(g["observable"].astype(str).tolist()))
        raw_names = list(dict.fromkeys(g["observable"].astype(str).tolist()))

        rows.append({
            "Source": WHAKAARI_SOURCE_LABELS.get(source_id, source_id),
            "Source ID": source_id,
            "Variables": "; ".join(labels),
            "Internal names": ", ".join(raw_names),
        })

    return pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)

def _collapse_to_source_markers(df):
    """
    Collapse observable-level metadata to one displayed marker per source.

    The returned variable table still lists all observables at each source, but
    the map itself shows source locations rather than one symbol per variable.
    """
    rows = []

    family_by_source = {
        "WSRZ": "seismic",
        "RGWC_RGWI": "deformation",
        "WHAKAARI_OPENMETEO_PROXY": "weather_meteo",
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
    """Order final Whakaari map sources without adding map IDs."""
    source_order = [
        "WSRZ",
        "RGWC_RGWI",
        "WHAKAARI_OPENMETEO_PROXY",
    ]

    df = df.copy()
    df["source_order"] = df["source_id"].apply(
        lambda x: source_order.index(x) if x in source_order else 999
    )

    return (
        df.sort_values(["source_order", "source_id"])
        .drop(columns="source_order")
        .reset_index(drop=True)
    )

def plot_whakaari_all_variables_map(
    metadata,
    *,
    satellite=True,
    title=None,
    figsize=(8.4, 7.4),
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

    for _, row in df.iterrows():
        if row["x_plot"] != row["x"] or row["y_plot"] != row["y"]:
            ax.plot(
                [row["x"], row["x_plot"]],
                [row["y"], row["y_plot"]],
                color="white" if satellite else "black",
                linewidth=0.45,
                alpha=0.48,
                zorder=6,
            )

    size_by_family = {
        "seismic": 115,
        "deformation": 140,
        "weather_meteo": 125,
    }

    for _, row in df.iterrows():
        family = row["family"]
        style = WHAKAARI_FAMILY_STYLE.get(
            family,
            {"marker": "o", "color": "0.5", "label": family},
        )

        ax.scatter(
            row["x_plot"],
            row["y_plot"],
            s=size_by_family.get(family, 105),
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.85,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

    _annotate_sources(ax, df, satellite=satellite)
    _clean_legend(ax)

    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fig.tight_layout(rect=[0, 0.085, 1, 1])

    if filename is None:
        filename = "whakaari_map"

    _save_current_figure(fig, filename, save_dir)

    variable_table = _make_source_variable_table(df)

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
            markersize=2.8,
            linewidth=0.80,
            color=THESIS_COLORS["series"],
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_title, loc="left", pad=4)
        ax.set_xlabel(unit_label)
        ax.set_ylabel("Density")
        ax.grid(True, which="both", alpha=0.14, linewidth=0.42)

        if col in {"spectral_log_ratio_4p5_8_over_8_16", "GNSS_deformation_rate"}:
            text_x, text_y, text_ha, text_va = 0.04, 0.06, "left", "bottom"
        else:
            text_x, text_y, text_ha, text_va = 0.98, 0.95, "right", "top"

        ax.text(
            text_x,
            text_y,
            f"positive n={n_positive}\n≤0 dropped={n_nonpositive}",
            transform=ax.transAxes,
            ha=text_ha,
            va=text_va,
            fontsize=7.6,
            color="0.30",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.4,
            },
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

def plot_whakaari_loglog_distributions(
    df,
    cols=None,
    *,
    bins=35,
    filename="whakaari_loglog_all_raw_variables",
    save_dir="figures",
):
    """
    Whakaari all-variable log-log distribution diagnostic.

    Uses all raw Whakaari variables by default. Variables with zeros or negative
    values are represented by their positive subset only, with exclusions
    reported in the returned table.
    """
    if cols is None:
        cols = [c for c in WHAKAARI_ALL_COLS if c in df.columns]

    return plot_loglog_distributions(
        df=df,
        cols=cols,
        axis_labels=WHAKAARI_AXIS_LABELS,
        bins=bins,
        filename=filename,
        save_dir=save_dir,
        title=None,
    )