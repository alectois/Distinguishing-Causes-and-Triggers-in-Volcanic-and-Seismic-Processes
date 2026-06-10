import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from whakaari_config import WHAKAARI_LAT, WHAKAARI_LON, WHAKAARI_EQ_RADIUS_KM
except Exception:
    WHAKAARI_LAT = -37.52
    WHAKAARI_LON = 177.18
    WHAKAARI_EQ_RADIUS_KM = np.nan

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
        "axes.spines.top": False,
        "axes.spines.right": False,
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
    "ratio_4p5_8_over_8_16",
    "event_rate_2_5",
    "effect_tremor_5_15",
    "API",
    "pressure_drop",
    "SO2_flux",
    "GNSS_deformation",
]

WHAKAARI_SEISMIC_COLS = [
    "hydro_2_5",
    "ratio_4p5_8_over_8_16",
    "event_rate_2_5",
    "effect_tremor_5_15",
]

WHAKAARI_EXTERNAL_COLS = [
    "API",
    "pressure_drop",
    "SO2_flux",
    "GNSS_deformation",
]

WHAKAARI_AXIS_LABELS = {
    "hydro_2_5": ("Hydrothermal tremor RMS, 2–5 Hz", "RMS velocity"),
    "ratio_4p5_8_over_8_16": ("Spectral ratio, 4.5–8 / 8–16 Hz", "Ratio"),
    "event_rate_2_5": ("STA/LTA event-rate proxy, 2–5 Hz", "Events per hour"),
    "effect_tremor_5_15": ("Tremor response RMS (Effect), 5–15 Hz", "RMS velocity"),
    "API": ("Antecedent precipitation index", "mm"),
    "pressure_drop": ("Atmospheric pressure drop", "hPa"),
    "SO2_flux": ("SO₂ flux", "t d⁻¹"),
    "GNSS_deformation": ("GNSS deformation", "m"),
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


def _transform_series(y, ylabel, transform):
    y = pd.to_numeric(y, errors="coerce")

    if transform == "log10":
        positive = y[y > 0]
        eps = positive.min() / 10 if len(positive) else 1e-12
        y = np.log10(y.clip(lower=0) + eps)
        ylabel = f"log10 {ylabel}"

    elif transform == "log1p":
        y = np.log1p(y.clip(lower=0))
        ylabel = f"log1p {ylabel}"

    return y, ylabel


def _plot_whakaari_group(
    csv_path,
    eruption_time,
    cols,
    filename,
    save_dir="figures",
    title=None,
    fig_width=8.0,
    panel_height=1.25,
    tick_interval_days=7,
    line_width=0.70,
    formats=("pdf", "png"),
    time_window=None,
    display_transform=None,
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

    if display_transform is None:
        display_transform = {}

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
        panel_title, ylabel = WHAKAARI_AXIS_LABELS.get(col, (col, "Value"))
        y, ylabel = _transform_series(df[col], ylabel, display_transform.get(col))

        if col == "SO2_flux":
            ax.scatter(
                df.index,
                y,
                s=10,
                color=THESIS_COLORS["series"],
                alpha=0.72,
                linewidths=0,
                rasterized=False,
            )

        elif col == "GNSS_deformation":
            ax.plot(
                df.index,
                y,
                color=THESIS_COLORS["series"],
                linewidth=max(line_width, 0.72),
                drawstyle="steps-post",
                alpha=0.98,
                antialiased=True,
            )

        else:
            ax.plot(
                df.index,
                y,
                color=THESIS_COLORS["series"],
                linewidth=line_width,
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

        ax.set_title(panel_title, loc="left", pad=4.5)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.grid(True, alpha=0.14, linewidth=0.42)
        ax.tick_params(axis="both", which="major", length=2.8, width=0.55, pad=3)
        ax.margins(x=0.01)

        y_values = y.values.astype(float)
        if np.isfinite(y_values).any() and np.nanmin(y_values) <= 0 <= np.nanmax(y_values):
            ax.axhline(0, color="0.25", linewidth=0.35, alpha=0.12, zorder=0)

    if title is not None:
        fig.suptitle(title, fontsize=11.0, fontweight="bold", y=0.995)
        top = 0.925
    else:
        top = 0.985

    locator = mdates.DayLocator(interval=tick_interval_days)
    formatter = mdates.DateFormatter("%b %d", tz=mdates.UTC)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Time (UTC)", labelpad=6)

    fig.subplots_adjust(
        left=0.110,
        right=0.992,
        bottom=0.090,
        top=top,
        hspace=0.46,
    )

    _save_current_figure(fig, filename=filename, save_dir=save_dir, formats=formats)
    plt.show()
    return fig, axes


def plot_whakaari_thesis_figures(
    csv_path,
    eruption_time,
    save_dir="figures",
    include_titles=False,
    tick_interval_days=7,
    display_transform=None,
):
    """
    Create the two final thesis overview figures:
    1. seismic variables
    2. external / environmental variables

    Both PDF and PNG are saved under figures/ by default.
    """
    seismic_title = "Seismic variables" if include_titles else None
    external_title = "External variables" if include_titles else None

    seismic = _plot_whakaari_group(
        csv_path=csv_path,
        eruption_time=eruption_time,
        cols=WHAKAARI_SEISMIC_COLS,
        filename="whakaari_seismic_variables",
        save_dir=save_dir,
        title=seismic_title,
        fig_width=8.0,
        panel_height=1.22,
        tick_interval_days=tick_interval_days,
        line_width=0.70,
        display_transform=display_transform,
    )

    external = _plot_whakaari_group(
        csv_path=csv_path,
        eruption_time=eruption_time,
        cols=WHAKAARI_EXTERNAL_COLS,
        filename="whakaari_external_variables",
        save_dir=save_dir,
        title=external_title,
        fig_width=8.0,
        panel_height=1.12,
        tick_interval_days=tick_interval_days,
        line_width=0.70,
        display_transform=display_transform,
    )

    return seismic, external


def plot_with_eruption_time(
    csv_path,
    cols,
    eruption_time,
    title=None,
    figsize=None,
    tick_interval_days=7,
    save_dir="figures",
    filename=None,
    display_transform=None,
):
    """Backward-compatible general plotting function."""
    if filename is None:
        filename = "whakaari_variables"

    if figsize is None:
        fig_width = 8.0
        panel_height = 1.20
    else:
        fig_width = figsize[0]
        panel_height = max(0.95, (figsize[1] - 0.50) / max(len(cols), 1))

    return _plot_whakaari_group(
        csv_path=csv_path,
        eruption_time=eruption_time,
        cols=cols,
        filename=filename,
        save_dir=save_dir,
        title=title,
        fig_width=fig_width,
        panel_height=panel_height,
        tick_interval_days=tick_interval_days,
        display_transform=display_transform,
    )


def plot_variable_pdfs(df, name, cols=None, bins=40, filename=None, save_dir="figures"):
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
        print(f"No numeric columns to plot for {name}.")
        return None

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna()

        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.72,
            edgecolor="black",
            linewidth=0.35,
        )

        ax.axvline(x.mean(), linestyle="--", linewidth=1.1, label="mean")
        ax.axvline(x.median(), linestyle=":", linewidth=1.1, label="median")

        ax.set_title(col, loc="left")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.14, linewidth=0.42)
        ax.legend(fontsize=8, frameon=False)

    for ax in axes[len(cols):]:
        ax.axis("off")

    if name is not None:
        fig.suptitle(f"{name}", y=1.01)

    plt.tight_layout()
    if filename is None:
        filename = f"{name}_variable_pdfs"
    _save_current_figure(fig, filename, save_dir)
    plt.show()
    return fig, axes


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
        "label": "Seismic variables",
    },
    "deformation": {
        "marker": "*",
        "color": "#ff7f0e",
        "label": "GNSS deformation",
    },
    "gas": {
        "marker": "o",
        "color": "#2ca02c",
        "label": "SO₂ flux",
    },
    "meteorology": {
        "marker": "s",
        "color": "#9467bd",
        "label": "Atmospheric pressure drop",
    },
    "weather_proxy": {
        "marker": "P",
        "color": "#bcbd22",
        "label": "API",
    },
}


WHAKAARI_SOURCE_LABELS = {
    "WSRZ": "WSRZ",
    "WID01": "WID01 SO₂",
    "RGWC_RGWI": "RGWC–RGWI GNSS",
    "WHAKAARI_OPENMETEO_PROXY": "Open-Meteo",
}


WHAKAARI_DEFAULT_LABEL_OFFSETS = {
    "WSRZ": (-20, -14),
    "WID01": (-18, 14),
    "RGWC_RGWI": (16, 10),
    "WHAKAARI_OPENMETEO_PROXY": (12, 16),
}


WHAKAARI_SOURCE_DISPLAY_NUDGES_M = {
    "WHAKAARI_OPENMETEO_PROXY": (45, 90),
    "WSRZ": (0, 0),
    "WID01": (0, 0),
    "RGWC_RGWI": (0, 0),
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


def _offset_observables(df, spacing_m=350):
    df = df.copy()
    df["x_plot"] = df["x"]
    df["y_plot"] = df["y"]

    is_projected = bool(df["is_projected"].iloc[0])

    for source_id, idx in df.groupby("source_id").groups.items():
        idx = list(idx)
        n = len(idx)

        if n <= 1:
            continue

        if n == 2:
            offsets = [(-0.8, 0.0), (0.8, 0.0)]
        elif n == 3:
            offsets = [(-0.85, -0.50), (0.85, -0.50), (0.00, 0.75)]
        elif n == 4:
            offsets = [(-0.85, -0.85), (0.85, -0.85), (-0.85, 0.85), (0.85, 0.85)]
        else:
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))
            offsets = []
            for i in range(n):
                r = i // ncols
                c = i % ncols
                offsets.append((
                    1.05 * (c - (ncols - 1) / 2),
                    1.05 * ((nrows - 1) / 2 - r),
                ))

        for row_idx, (ox, oy) in zip(idx, offsets):
            if is_projected:
                dx = ox * spacing_m
                dy = oy * spacing_m
            else:
                dx = ox * spacing_m / 111_200
                dy = oy * spacing_m / 111_200

            df.loc[row_idx, "x_plot"] = df.loc[row_idx, "x"] + dx
            df.loc[row_idx, "y_plot"] = df.loc[row_idx, "y"] + dy

    return df


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


def _set_extent(
    ax,
    df,
    *,
    include_eq_radius=False,
    pad_fraction=0.08,
    min_pad_m=650,
):
    xmin, xmax = df["x_plot"].min(), df["x_plot"].max()
    ymin, ymax = df["y_plot"].min(), df["y_plot"].max()

    if include_eq_radius and "radius_km" in df.columns:
        eq = df[df["spatial_type"] == "search_radius"]

        if not eq.empty:
            r_m = float(eq["radius_km"].iloc[0]) * 1000
            cx = float(eq["x"].iloc[0])
            cy = float(eq["y"].iloc[0])

            xmin = min(xmin, cx - r_m)
            xmax = max(xmax, cx + r_m)
            ymin = min(ymin, cy - r_m)
            ymax = max(ymax, cy + r_m)

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
        "Source centre",
        "Seismic variables",
        "SO₂ flux",
        "GNSS deformation",
        "Meteorological variables",
        "API",
        "Local earthquake count",
    ]

    ordered_labels = [label for label in preferred_order if label in unique]
    remaining_labels = [label for label in unique if label not in ordered_labels]
    final_labels = ordered_labels + remaining_labels
    final_handles = [unique[label] for label in final_labels]

    ax.legend(
        final_handles,
        final_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.040),
        ncol=4,
        frameon=True,
        framealpha=0.97,
        fontsize=8.0,
        handletextpad=0.85,
        columnspacing=2.2,
        borderpad=0.85,
        labelspacing=0.85,
        markerscale=1.0,
    )


def _source_centres(df):
    rows = []

    for source_id, g in df[df["source_id"] != "Whakaari reference"].groupby("source_id"):
        rows.append({
            "source_id": source_id,
            "x": g["x"].iloc[0],
            "y": g["y"].iloc[0],
            "x_plot": g["x_plot"].mean(),
            "y_plot": g["y_plot"].mean(),
        })

    return pd.DataFrame(rows)


def _annotate_sources(ax, centres, reference_df, satellite=False):
    label_rows = pd.concat([centres, reference_df], ignore_index=True)

    line_color = "white" if satellite else "black"
    line_alpha = 0.95 if satellite else 0.80

    for _, row in label_rows.iterrows():
        source_id = row["source_id"]
        label = WHAKAARI_SOURCE_LABELS.get(source_id, source_id)
        dx, dy = WHAKAARI_DEFAULT_LABEL_OFFSETS.get(source_id, (12, 12))

        x_anchor = row["x_plot"] if "x_plot" in row else row["x"]
        y_anchor = row["y_plot"] if "y_plot" in row else row["y"]

        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
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


def _assign_observable_numbers(df):
    df = df.copy()

    source_order = [
        "WSRZ",
        "WID01",
        "RGWC_RGWI",
        "WHAKAARI_OPENMETEO_PROXY",
    ]

    df["source_order"] = df["source_id"].apply(
        lambda x: source_order.index(x) if x in source_order else 999
    )

    df = df.sort_values(["source_order", "source_id", "observable"]).reset_index(drop=True)
    df["map_id"] = [f"{i + 1}" for i in range(len(df))]

    return df.drop(columns=["source_order"])


def plot_whakaari_all_variables_map(
    metadata,
    *,
    satellite=True,
    title=None,
    offset_radius_m=55,
    figsize=(9.2, 8.0),
    filename=None,
    save_dir="figures",
    include_eq_radius=False,
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

    df = _assign_observable_numbers(df)
    df = _project_to_web_mercator(df)
    df = _offset_observables(df, spacing_m=offset_radius_m)
    df = _nudge_overlapping_sources(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax_panel = None

    _set_extent(ax, df, include_eq_radius=include_eq_radius)

    if bool(df["is_projected"].iloc[0]):
        _add_basemap(ax, satellite=satellite)

    for _, row in df.iterrows():
        if row["source_id"] == "Whakaari reference":
            continue

        if row["x_plot"] != row["x"] or row["y_plot"] != row["y"]:
            ax.plot(
                [row["x"], row["x_plot"]],
                [row["y"], row["y_plot"]],
                color="white" if satellite else "black",
                linewidth=0.45,
                alpha=0.48,
                zorder=6,
            )

    centres = _source_centres(df)

    ax.scatter(
        centres["x"],
        centres["y"],
        s=36,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.15,
        alpha=0.98,
        zorder=10,
        label="Source centre",
    )

    reference_df = pd.DataFrame(columns=df.columns)

    for _, row in df.iterrows():
        family = row["family"]
        style = WHAKAARI_FAMILY_STYLE.get(
            family,
            {"marker": "o", "color": "0.5", "label": family},
        )

        size = 90

        ax.scatter(
            row["x_plot"],
            row["y_plot"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.70,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

        ax.annotate(
            row["map_id"],
            xy=(row["x_plot"], row["y_plot"]),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.15",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.55,
                "alpha": 0.95,
            },
            zorder=35,
        )

    _annotate_sources(
        ax,
        centres,
        reference_df,
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

    if filename is None:
        filename = "whakaari_all_variables_map"
    _save_current_figure(fig, filename, save_dir)

    variable_table = _make_source_variable_table(df)

    return fig, ax, variable_table
