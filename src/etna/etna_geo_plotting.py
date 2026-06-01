import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

def _save_current_figure(fig, filename, save_dir="figures"):
    """Save a displayed plotting figure to figures/ as PDF and PNG."""
    if save_dir is None or filename is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=300,
            facecolor="white",
        )

FAMILY_STYLE = {
    "seismic": {
        "marker": "^",
        "color": "#1f77b4",
        "label": "Seismic variables",
    },
    "gas": {
        "marker": "o",
        "color": "#2ca02c",
        "label": "Soil CO₂ concentration",
    },
    "meteorology": {
        "marker": "s",
        "color": "#9467bd",
        "label": "Meteorological variables",
    },
    "gas_plume": {
        "marker": "D",
        "color": "#17becf",
        "label": "Plume CO₂/SO₂ ratio",
    },
    "weather_proxy": {
        "marker": "P",
        "color": "#bcbd22",
        "label": "API",
    },
    "summit": {
        "marker": "X",
        "color": "black",
        "label": "Etna summit",
    },
}


SOURCE_LABELS = {
    "ME01": "ME01",
    "ME02": "ME02",
    "ETNAGAS_3": "ETNAGAS network, 3c",
    "ETNA_SUMMIT_PLUME": "INGV-PA, Multi-GAS",
    "ETNA_OPENMETEO_PROXY": "Open-Meteo",
    "Etna summit": "Etna summit",
}

DEFAULT_LABEL_OFFSETS = {
    # Seismic stations
    "ME01": (-18, -16),
    "ME02": (18, -16),

    # Summit/proxy area
    "ETNA_OPENMETEO_PROXY": (-18, 14),
    "ETNA_SUMMIT_PLUME": (18, 14),
    "Etna summit": (18, -18),

    # Gas/meteo station
    "ETNAGAS_3": (26, -24),
}

SOURCE_DISPLAY_NUDGES_M = {
    # Separate proxy/summit sources that are geographically almost identical.
    # These move only the displayed marker positions, not the true source coordinates.
    "ETNA_OPENMETEO_PROXY": (-2600, 1300),
    "ETNA_SUMMIT_PLUME": (2200, 1100),

    # Keep summit at the actual summit reference point.
    "Etna summit": (0, 0),

    # Leave the other sources unchanged.
    "ME01": (0, 0),
    "ME02": (0, 0),
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


def _offset_observables(df, spacing_m=2300):
    df = df.copy()
    df["x_plot"] = df["x"]
    df["y_plot"] = df["y"]

    is_projected = bool(df["is_projected"].iloc[0])

    for source_id, idx in df.groupby("source_id").groups.items():
        idx = list(idx)
        n = len(idx)

        if n <= 1:
            continue

        # Wider, readable layouts by group size.
        if n == 2:
            offsets = [(-0.8, 0.0), (0.8, 0.0)]

        elif n == 3:
            offsets = [
                (-0.85, -0.50),
                (0.85, -0.50),
                (0.00, 0.75),
            ]

        elif n == 4:
            offsets = [
                (-0.85, -0.85),
                (0.85, -0.85),
                (-0.85, 0.85),
                (0.85, 0.85),
            ]

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

def _set_extent(ax, df, pad_fraction=0.055, min_pad_m=1800):
    xmin, xmax = df["x_plot"].min(), df["x_plot"].max()
    ymin, ymax = df["y_plot"].min(), df["y_plot"].max()

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
        "Soil CO₂ concentration",
        "Meteorological variables",
        "API",
        "Plume CO₂/SO₂ ratio",
        "Etna summit",
    ]

    ordered_labels = [l for l in preferred_order if l in unique]
    ordered_handles = [unique[l] for l in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.040),
        ncol=4,
        frameon=True,
        framealpha=0.97,
        fontsize=8.4,
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

        x_anchor = row["x_plot"] if "x_plot" in row else row["x"]
        y_anchor = row["y_plot"] if "y_plot" in row else row["y"]

        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.6,
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.35,
                "alpha": 0.90,
                "pad": 1.45,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": line_color,
                "linewidth": 1.15,
                "alpha": line_alpha,
                "shrinkA": 2,
                "shrinkB": 5,
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


def _assign_observable_numbers(df):
    """
    Give each observable a short map ID.
    Full readable labels are shown in the side panel.
    """
    df = df.copy()

    source_order = [
        "ME01",
        "ME02",
        "ETNAGAS_3",
        "ETNA_OPENMETEO_PROXY",
        "ETNA_SUMMIT_PLUME",
        "Etna summit",
    ]

    df["source_order"] = df["source_id"].apply(
        lambda x: source_order.index(x) if x in source_order else 999
    )

    df = df.sort_values(["source_order", "source_id", "observable"]).reset_index(drop=True)

    df["map_id"] = [
        f"{i + 1}" if sid != "Etna summit" else "S"
        for i, sid in enumerate(df["source_id"])
    ]

    return df.drop(columns=["source_order"])


def _add_observable_label_panel(fig, ax_panel, df):
    ax_panel.axis("off")

    rows = df[df["source_id"] != "Etna summit"].copy()

    ax_panel.text(
        0.0,
        1.0,
        "Observable labels",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )

    y = 0.94

    for source_id, g in rows.groupby("source_id", sort=False):
        source_label = SOURCE_LABELS.get(source_id, source_id)

        ax_panel.text(
            0.0,
            y,
            source_label,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="top",
        )
        y -= 0.035

        for _, row in g.iterrows():
            label = row.get("observable_label", row["observable"])

            ax_panel.text(
                0.02,
                y,
                f"{row['map_id']}. {label}",
                fontsize=8.2,
                ha="left",
                va="top",
                wrap=True,
            )

            y -= 0.050

        y -= 0.018

    ax_panel.text(
        0.0,
        0.02,
        "Clustered markers share the same measurement source.",
        fontsize=7.8,
        color="0.25",
        ha="left",
        va="bottom",
    )


def plot_etna_all_variables_map(
    metadata,
    *,
    summit_lat=37.748,
    summit_lon=14.999,
    satellite=True,
    title=None,
    offset_radius_m=2300,
    figsize=(10.5, 8.0),
    show_label_panel=False,
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

    df = _assign_observable_numbers(df)
    df = _project_to_web_mercator(df)
    df = _offset_observables(df, spacing_m=offset_radius_m)
    df = _nudge_overlapping_sources(df)

    if show_label_panel:
        fig = plt.figure(figsize=(13.5, 8.0))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.8, 1.45],
            wspace=0.035,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_panel = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_panel = None

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
                linewidth=0.55,
                alpha=0.60,
                zorder=6,
            )

    centres = _source_centres(df)

    # True measurement source centres.
    ax.scatter(
        centres["x"],
        centres["y"],
        s=58,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        alpha=0.98,
        zorder=10,
        label="Source centre",
    )

    summit_df = df[df["source_id"] == "Etna summit"].drop_duplicates("source_id")

    # Plot all observable markers.
    for _, row in df.iterrows():
        family = row["family"]
        style = FAMILY_STYLE.get(
            family,
            {"marker": "o", "color": "0.5", "label": family},
        )

        size = 155 if family == "summit" else 105

        ax.scatter(
            row["x_plot"],
            row["y_plot"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

        # Short readable numbered label on top of marker.
        ax.annotate(
            row["map_id"],
            xy=(row["x_plot"], row["y_plot"]),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.20",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.6,
                "alpha": 0.95,
            },
            zorder=35,
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

    if show_label_panel and ax_panel is not None:
        _add_observable_label_panel(fig, ax_panel, df)

    fig.tight_layout(rect=[0, 0.085, 1, 1])

    variable_table = _make_source_variable_table(df)

    if filename is not None:
        _save_current_figure(fig, filename, save_dir)

    return fig, ax, variable_table