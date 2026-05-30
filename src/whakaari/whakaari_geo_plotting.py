import matplotlib.pyplot as plt
import numpy as np

from whakaari_config import WHAKAARI_LAT, WHAKAARI_LON


WHAKAARI_FAMILY_MARKERS = {
    "seismic": "^",
    "seismic_effect": "*",
    "gas": "o",
    "deformation": "D",
    "weather_proxy": "P",
    "earthquake_catalogue": "X",
}

WHAKAARI_FAMILY_COLORS = {
    "seismic": "#0072B2",
    "seismic_effect": "#D55E00",
    "gas": "#009E73",
    "deformation": "#E69F00",
    "weather_proxy": "#F0E442",
    "earthquake_catalogue": "0.35",
}


def plot_whakaari_observable_locations(
    metadata,
    volcano_lat=WHAKAARI_LAT,
    volcano_lon=WHAKAARI_LON,
    title="Whakaari: spatial provenance of observables",
    annotate=True,
    ax=None,
):
    df = metadata.dropna(subset=["lat", "lon"]).copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
    else:
        fig = ax.figure

    ax.scatter(
        volcano_lon,
        volcano_lat,
        marker="^",
        s=180,
        c="black",
        label="Whakaari reference point",
        zorder=5,
    )

    grouped = df.groupby(["source_id", "family", "spatial_type"], dropna=False)

    for (source_id, family, spatial_type), g in grouped:
        lat = float(g["lat"].iloc[0])
        lon = float(g["lon"].iloc[0])

        marker = WHAKAARI_FAMILY_MARKERS.get(family, "o")
        color = WHAKAARI_FAMILY_COLORS.get(family, "0.5")
        size = 230 if family == "seismic_effect" else 150

        ax.scatter(
            lon,
            lat,
            marker=marker,
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            label=family,
            zorder=4,
        )

        if spatial_type == "station_pair":
            lat2 = g["lat2"].iloc[0] if "lat2" in g.columns else np.nan
            lon2 = g["lon2"].iloc[0] if "lon2" in g.columns else np.nan

            if not np.isnan(lat2) and not np.isnan(lon2):
                ax.plot(
                    [lon, lon2],
                    [lat, lat2],
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.85,
                    zorder=3,
                )

        if spatial_type == "search_radius":
            radius_km = g["radius_km"].iloc[0] if "radius_km" in g.columns else np.nan

            if not np.isnan(radius_km):
                radius_deg = radius_km / 111.2
                circle = plt.Circle(
                    (lon, lat),
                    radius_deg,
                    fill=False,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                )
                ax.add_patch(circle)

        if annotate:
            obs = ", ".join(g["observable"].tolist())
            ax.annotate(
                f"{source_id}\n{obs}",
                xy=(lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="bottom",
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig, ax