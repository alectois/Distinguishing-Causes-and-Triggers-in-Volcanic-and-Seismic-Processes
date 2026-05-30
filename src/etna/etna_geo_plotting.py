import matplotlib.pyplot as plt
import numpy as np


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


def plot_etna_observable_locations(
    metadata,
    volcano_lat=37.748,
    volcano_lon=14.999,
    title="Etna: spatial provenance of observables",
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
        label="Etna summit reference",
        zorder=5,
    )

    grouped = df.groupby(["source_id", "family", "spatial_type"], dropna=False)

    for (source_id, family, spatial_type), g in grouped:
        lat = float(g["lat"].iloc[0])
        lon = float(g["lon"].iloc[0])

        marker = ETNA_FAMILY_MARKERS.get(family, "o")
        color = ETNA_FAMILY_COLORS.get(family, "0.5")
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