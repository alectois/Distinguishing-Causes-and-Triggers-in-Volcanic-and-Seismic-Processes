def add_fdsn_station_coordinates(
    metadata,
    client,
    network,
    station_ids,
    location="*",
    channel="*",
    starttime=None,
    endtime=None,
):
    metadata = metadata.copy()

    for station in station_ids:
        try:
            inv = client.get_stations(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                level="station",
            )

            sta = inv[0][0]
            mask = metadata["source_id"] == station

            metadata.loc[mask, "lat"] = sta.latitude
            metadata.loc[mask, "lon"] = sta.longitude
            metadata.loc[mask, "elevation_m"] = sta.elevation

        except Exception as exc:
            print(f"Could not fetch coordinates for {network}.{station}: {exc}")

    return metadata