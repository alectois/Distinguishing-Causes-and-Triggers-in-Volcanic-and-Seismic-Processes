import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ETNA_FAMILY_MARKERS = {
    "seismic": "^",
    "seismic_effect": "*",
    "gas": "o",
    "gas_plume": "o",
    "meteorology": "s",
    "weather_proxy": "P",
}

ETNA_FAMILY_COLORS = {
    "seismic": "#0072B2",
    "seismic_effect": "#D55E00",
    "gas": "#009E73",
    "gas_plume": "#56B4E9",
    "meteorology": "#CC79A7",
    "weather_proxy": "#F0E442",
}


def _source_level_metadata(metadata):
    """
    Collapse observable-level metadata to one row per measurement source.

    This prevents stacked markers and unreadable labels when several variables
    come from the same station/source.
    """
    rows = []

    for source_id, g in metadata.groupby("source_id", dropna=False):
        g = g.copy()

        if g[["lat", "lon"]].isna().any(axis=None):
            continue

        # Prefer effect role/family only for role information, but for plotting
        # the station/source itself should usually be the broader source family.
        families = list(dict.fromkeys(g["family"].astype(str).tolist()))
        spatial_types = list(dict.fromkeys(g["spatial_type"].astype(str).tolist()))
        observables = list(dict.fromkeys(g["observable"].astype(str).tolist()))

        if "seismic" in families or "seismic_effect" in families:
            plot_family = "seismic"
        elif "gas" in families:
            plot_family = "gas"
        elif "meteorology" in families:
            plot_family = "meteorology"
        elif "gas_plume" in families:
            plot_family = "gas_plume"
        elif "weather_proxy" in families:
            plot_family = "weather_proxy"
        else:
            plot_family = families[0]

        rows.append({
            "source_id": source_id,
            "source_label": g["source_label"].iloc[0],
            "lat": float(g["lat"].iloc[0]),
            "lon": float(g["lon"].iloc[0]),
            "family": plot_family,
            "families": ", ".join(families),
            "spatial_type": spatial_types[0],
            "observables": observables,
            "observable_text": ", ".join(observables),
        })

    return pd.DataFrame(rows)


def _add_basemap_if_possible(ax, crs="EPSG:3857", source=None):
    """
    Add a map-tile basemap if contextily is installed and internet/cache is available.
    The plot still works without a basemap.
    """
    try:
        import contextily as cx

        if source is None:
            source = cx.providers.Esri.WorldTopoMap

        cx.add_basemap(
            ax,
            crs=crs,
            source=source,
            attribution_size=6,
        )
        return True

    except Exception as exc:
        warnings.warn(
            f"Could not add contextily basemap. Falling back to plain map axes. "
            f"Reason: {exc}"
        )
        return False


def _to_web_mercator(df):
    """
    Convert lon/lat dataframe to Web Mercator coordinates for contextily basemaps.
    """
    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        return pd.DataFrame(gdf.drop(columns="geometry"))

    except Exception as exc:
        warnings.warn(
            f"Could not use GeoPandas projection. Falling back to lon/lat plot. "
            f"Reason: {exc}"
        )

        out = df.copy()
        out["x"] = out["lon"]
        out["y"] = out["lat"]
        return out


def _set_extent_from_center(ax, center_lon, center_lat, buffer_km, use_web_mercator=True):
    """
    Set map extent around a center point.
    """
    if use_web_mercator:
        center = _to_web_mercator(pd.DataFrame({
            "lon": [center_lon],
            "lat": [center_lat],
        }))
        cx = float(center["x"].iloc[0])
        cy = float(center["y"].iloc[0])
        b = buffer_km * 1000.0

        ax.set_xlim(cx - b, cx + b)
        ax.set_ylim(cy - b, cy + b)
    else:
        # Rough degree fallback.
        b_lat = buffer_km / 111.2
        b_lon = buffer_km / (111.2 * np.cos(np.deg2rad(center_lat)))

        ax.set_xlim(center_lon - b_lon, center_lon + b_lon)
        ax.set_ylim(center_lat - b_lat, center_lat + b_lat)


def _plot_source_points(ax, source_df, annotate=True):
    """
    Plot one marker per source.
    """
    for _, row in source_df.iterrows():
        family = row["family"]

        marker = ETNA_FAMILY_MARKERS.get(family, "o")
        color = ETNA_FAMILY_COLORS.get(family, "0.5")

        ax.scatter(
            row["x"],
            row["y"],
            marker=marker,
            s=170,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
            zorder=5,
            label=family,
        )

        if annotate:
            ax.annotate(
                row["source_id"],
                xy=(row["x"], row["y"]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
                zorder=6,
            )


def _make_source_table(ax, source_df, title="Observable source"):
    """
    Add a compact source-to-observable table below/next to the map.
    """
    table_rows = []

    for _, row in source_df.sort_values("source_id").iterrows():
        table_rows.append([
            row["source_id"],
            row["spatial_type"],
            row["observable_text"],
        ])

    table = ax.table(
        cellText=table_rows,
        colLabels=["Source", "Spatial type", "Observables"],
        loc="bottom",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, -0.48, 1.0, 0.38],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.3)

    ax.text(
        0.0,
        -0.08,
        title,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _clean_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        frameon=True,
        fontsize=8,
    )


def plot_etna_local_observable_map(
    metadata,
    volcano_lat=37.748,
    volcano_lon=14.999,
    buffer_km=18,
    exclude_weather_proxy=True,
    title="Etna: local spatial provenance of observables",
    add_table=True,
):
    """
    Local map around Etna.

    This should be the main thesis map. It shows direct station/source locations
    near the volcano. Weather proxy can be excluded here because it stretches the
    extent and makes the station map less readable.
    """
    source_df = _source_level_metadata(metadata)

    if exclude_weather_proxy:
        source_df = source_df[source_df["family"] != "weather_proxy"].copy()

    # Add summit reference as its own source.
    summit = pd.DataFrame([{
        "source_id": "Etna summit",
        "source_label": "Etna summit reference",
        "lat": volcano_lat,
        "lon": volcano_lon,
        "family": "summit",
        "families": "summit",
        "spatial_type": "reference_point",
        "observables": [],
        "observable_text": "volcano reference",
    }])

    plot_df = pd.concat([source_df, summit], ignore_index=True)
    plot_df = _to_web_mercator(plot_df)

    use_web_mercator = not np.allclose(plot_df["x"], plot_df["lon"])

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    _set_extent_from_center(
        ax,
        center_lon=volcano_lon,
        center_lat=volcano_lat,
        buffer_km=buffer_km,
        use_web_mercator=use_web_mercator,
    )

    if use_web_mercator:
        _add_basemap_if_possible(ax, crs="EPSG:3857")

    # Plot summit separately.
    summit_row = plot_df[plot_df["source_id"] == "Etna summit"].iloc[0]
    ax.scatter(
        summit_row["x"],
        summit_row["y"],
        marker="^",
        s=230,
        color="black",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
        label="Etna summit reference",
    )
    ax.annotate(
        "Etna summit",
        xy=(summit_row["x"], summit_row["y"]),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 1.5,
        },
        zorder=8,
    )

    source_plot_df = plot_df[plot_df["source_id"] != "Etna summit"].copy()
    _plot_source_points(ax, source_plot_df, annotate=True)

    _clean_legend(ax)

    ax.set_title(title)
    ax.set_xlabel("Web Mercator x" if use_web_mercator else "Longitude")
    ax.set_ylabel("Web Mercator y" if use_web_mercator else "Latitude")
    ax.grid(True, alpha=0.22)

    if add_table:
        _make_source_table(ax, source_plot_df)
        fig.subplots_adjust(bottom=0.33)
    else:
        fig.tight_layout()

    return fig, ax


def plot_etna_proxy_context_map(
    metadata,
    volcano_lat=37.748,
    volcano_lon=14.999,
    buffer_km=55,
    title="Etna: regional context of proxy observables",
):
    """
    Regional map showing Etna and the Open-Meteo / other proxy points.

    Use this as a secondary panel, not as the main local station map.
    """
    source_df = _source_level_metadata(metadata)

    proxy_df = source_df[
        source_df["family"].isin(["weather_proxy", "gas_plume"])
    ].copy()

    summit = pd.DataFrame([{
        "source_id": "Etna summit",
        "source_label": "Etna summit reference",
        "lat": volcano_lat,
        "lon": volcano_lon,
        "family": "summit",
        "families": "summit",
        "spatial_type": "reference_point",
        "observables": [],
        "observable_text": "volcano reference",
    }])

    plot_df = pd.concat([proxy_df, summit], ignore_index=True)
    plot_df = _to_web_mercator(plot_df)

    use_web_mercator = not np.allclose(plot_df["x"], plot_df["lon"])

    fig, ax = plt.subplots(figsize=(8.2, 7.4))

    _set_extent_from_center(
        ax,
        center_lon=volcano_lon,
        center_lat=volcano_lat,
        buffer_km=buffer_km,
        use_web_mercator=use_web_mercator,
    )

    if use_web_mercator:
        _add_basemap_if_possible(ax, crs="EPSG:3857")

    summit_row = plot_df[plot_df["source_id"] == "Etna summit"].iloc[0]

    ax.scatter(
        summit_row["x"],
        summit_row["y"],
        marker="^",
        s=230,
        color="black",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
        label="Etna summit reference",
    )

    _plot_source_points(
        ax,
        plot_df[plot_df["source_id"] != "Etna summit"],
        annotate=True,
    )

    # Draw dashed lines from proxy points to summit.
    for _, row in plot_df[plot_df["source_id"] != "Etna summit"].iterrows():
        ax.plot(
            [summit_row["x"], row["x"]],
            [summit_row["y"], row["y"]],
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
            zorder=4,
        )

    _clean_legend(ax)

    ax.set_title(title)
    ax.set_xlabel("Web Mercator x" if use_web_mercator else "Longitude")
    ax.set_ylabel("Web Mercator y" if use_web_mercator else "Latitude")
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    return fig, ax