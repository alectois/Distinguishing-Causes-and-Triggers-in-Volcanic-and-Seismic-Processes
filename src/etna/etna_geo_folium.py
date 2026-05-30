import folium
import pandas as pd


ETNA_SUMMIT_LAT = 37.748
ETNA_SUMMIT_LON = 14.999


FAMILY_COLORS = {
    "seismic": "blue",
    "seismic_effect": "orange",
    "gas": "green",
    "gas_plume": "purple",
    "meteorology": "pink",
    "weather_proxy": "cadetblue",
    "summit": "black",
}


def collapse_to_sources(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Convert observable-level metadata into one row per measurement source.
    This prevents stacked markers when several observables come from the same station.
    """
    rows = []

    for source_id, g in metadata.groupby("source_id", dropna=False):
        g = g.copy()

        if g[["lat", "lon"]].isna().any(axis=None):
            continue

        families = list(dict.fromkeys(g["family"].astype(str).tolist()))
        observables = list(dict.fromkeys(g["observable"].astype(str).tolist()))
        labels = list(dict.fromkeys(g["observable_label"].astype(str).tolist()))

        if "seismic" in families or "seismic_effect" in families:
            plot_family = "seismic"
        elif "gas" in families:
            plot_family = "gas"
        elif "gas_plume" in families:
            plot_family = "gas_plume"
        elif "meteorology" in families:
            plot_family = "meteorology"
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
            "spatial_type": g["spatial_type"].iloc[0],
            "observables": observables,
            "observable_labels": labels,
        })

    return pd.DataFrame(rows)


def make_popup_html(row):
    obs_items = "".join(
        f"<li>{obs}</li>" for obs in row["observable_labels"]
    )

    return f"""
    <b>{row['source_id']}</b><br>
    {row['source_label']}<br><br>
    <b>Spatial type:</b> {row['spatial_type']}<br>
    <b>Observables:</b>
    <ul>{obs_items}</ul>
    """


def plot_etna_observable_folium_map(
    metadata: pd.DataFrame,
    output_html=None,
    include_openmeteo=True,
    zoom_start=11,
):
    """
    Create a satellite-style interactive map of Etna observable sources.

    Parameters
    ----------
    metadata:
        Output of etna_observable_metadata(), with lat/lon filled.
    output_html:
        Optional path for saving the map as HTML.
    include_openmeteo:
        If False, omits regional weather proxy point to keep focus near Etna.
    zoom_start:
        Initial zoom level.
    """

    source_df = collapse_to_sources(metadata)

    if not include_openmeteo:
        source_df = source_df[source_df["family"] != "weather_proxy"].copy()

    m = folium.Map(
        location=[ETNA_SUMMIT_LAT, ETNA_SUMMIT_LON],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    # Satellite basemap similar to your example.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # Optional labels/roads layer.
    folium.TileLayer(
        tiles=(
            "https://services.arcgisonline.com/ArcGIS/rest/services/"
            "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Labels © Esri",
        name="Labels",
        overlay=True,
        control=True,
    ).add_to(m)

    # Summit marker.
    folium.Marker(
        location=[ETNA_SUMMIT_LAT, ETNA_SUMMIT_LON],
        tooltip="Etna summit reference",
        popup="Etna summit reference",
        icon=folium.Icon(color="black", icon="triangle", prefix="fa"),
    ).add_to(m)

    # Source markers.
    for _, row in source_df.iterrows():
        color = FAMILY_COLORS.get(row["family"], "gray")

        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=row["source_id"],
            popup=folium.Popup(make_popup_html(row), max_width=360),
            icon=folium.Icon(color=color, icon="map-marker", prefix="fa"),
        ).add_to(m)

        # Draw line from Open-Meteo proxy to summit, to show it is a proxy.
        if row["family"] == "weather_proxy":
            folium.PolyLine(
                locations=[
                    [ETNA_SUMMIT_LAT, ETNA_SUMMIT_LON],
                    [row["lat"], row["lon"]],
                ],
                color="white",
                weight=2,
                opacity=0.8,
                dash_array="5, 5",
                tooltip="Proxy distance from Etna summit",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    if output_html is not None:
        m.save(output_html)

    return m