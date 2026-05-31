import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from whakaari_config import WHAKAARI_LAT, WHAKAARI_LON, WHAKAARI_EQ_RADIUS_KM


FAMILY_STYLE = {
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
    "earthquake_catalogue": {
        "marker": "X",
        "color": "#8c564b",
        "label": "Local earthquake count",
    },
}


SOURCE_LABELS = {
    "WSRZ": "WSRZ",
    "WID01": "WID01 SO₂",
    "RGWC_RGWI": "RGWC–RGWI GNSS",
    "WHAKAARI_OPENMETEO_PROXY": "Open-Meteo",
    "WHAKAARI_EQ_RADIUS": "EQ catalogue centre",
}


DEFAULT_LABEL_OFFSETS = {
    "WSRZ": (-20, -14),
    "WID01": (-18, 14),
    "RGWC_RGWI": (16, 10),

    "WHAKAARI_OPENMETEO_PROXY": (12, 16),

    "WHAKAARI_EQ_RADIUS": (14, 10),
}


SOURCE_DISPLAY_NUDGES_M = {
    "WHAKAARI_OPENMETEO_PROXY": (45, 90),
    "WHAKAARI_EQ_RADIUS": (120, 35),

    # Real sources stay fixed.
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

    # Keep any unexpected labels instead of silently dropping them.
    remaining_labels = [
        label for label in unique
        if label not in ordered_labels
    ]

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
        fontsize=8.3,
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


def _draw_station_pair_lines(ax, df, satellite=False):
    if "lat2" not in df.columns or "lon2" not in df.columns:
        return

    pairs = df[df["spatial_type"] == "station_pair"].dropna(subset=["lat2", "lon2"])

    if pairs.empty:
        return

    pair_points = pairs[["source_id", "lon", "lat", "lon2", "lat2"]].drop_duplicates()

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        for _, row in pair_points.iterrows():
            points = gpd.GeoSeries(
                [
                    Point(row["lon"], row["lat"]),
                    Point(row["lon2"], row["lat2"]),
                ],
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")

            x1, y1 = points.iloc[0].x, points.iloc[0].y
            x2, y2 = points.iloc[1].x, points.iloc[1].y

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white" if satellite else "black",
                linestyle="--",
                linewidth=1.1,
                alpha=0.8,
                zorder=5,
            )

    except Exception:
        return


def _draw_eq_radius(ax, df):
    eq = df[df["spatial_type"] == "search_radius"]

    if eq.empty or "radius_km" not in eq.columns:
        return

    row = eq.iloc[0]
    radius_m = float(row["radius_km"]) * 1000

    circle = plt.Circle(
        (row["x"], row["y"]),
        radius_m,
        fill=False,
        edgecolor="#8c564b",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        zorder=4,
    )

    ax.add_patch(circle)


def _make_source_variable_table(df):
    rows = []

    for source_id, g in df[df["source_id"] != "Whakaari reference"].groupby("source_id"):
        labels = list(dict.fromkeys(g["observable_label"].astype(str).tolist()))
        raw_names = list(dict.fromkeys(g["observable"].astype(str).tolist()))

        rows.append({
            "Source": SOURCE_LABELS.get(source_id, source_id),
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
        "WHAKAARI_EQ_RADIUS",
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
    show_label_panel=False,
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

    if show_label_panel:
        fig = plt.figure(figsize=(12.5, 8.0))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.5, 1.4],
            wspace=0.035,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_panel = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_panel = None

    _set_extent(ax, df, include_eq_radius=False)

    if bool(df["is_projected"].iloc[0]):
        _add_basemap(ax, satellite=satellite)


    # _draw_eq_radius(ax, df)
    #_draw_station_pair_lines(ax, df, satellite=satellite)

    # Thin connector lines from offset observable marker to true source centre.
    for _, row in df.iterrows():
        if row["source_id"] == "Whakaari reference":
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

    ax.scatter(
        centres["x"],
        centres["y"],
        s=36,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        alpha=0.98,
        zorder=10,
        label="Source centre",
    )

    reference_df = pd.DataFrame(columns=df.columns)

    for _, row in df.iterrows():
        family = row["family"]
        style = FAMILY_STYLE.get(
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
            linewidth=0.8,
            alpha=0.96,
            zorder=12,
            label=style["label"],
        )

        ax.annotate(
            row["map_id"],
            xy=(row["x_plot"], row["y_plot"]),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.15",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.6,
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

    variable_table = _make_source_variable_table(df)

    return fig, ax, variable_table