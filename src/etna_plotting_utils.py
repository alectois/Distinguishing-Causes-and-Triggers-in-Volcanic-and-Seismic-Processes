import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime

def dataset_health_report(df, name):
    print(f"\n{name}")
    print("shape:", df.shape)

    if "time" in df.columns:
        print("duplicate timestamps:", df["time"].duplicated().sum())
        print("time sorted:", df["time"].is_monotonic_increasing)
    else:
        print("No 'time' column found.")

    print("\nMissing fraction:")
    print(df.isna().mean().sort_values())


def plot_scaled_dataset(df, station_name):
    cols = [c for c in df.columns if c.endswith("_scaled")]

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.5 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df["time"], df[col], linewidth=0.8)
        ax.set_title(f"{station_name} — {col}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_station_variable(df1, df2, var, name1="ME01", name2="ME02"):
    plt.figure(figsize=(14,4))
    plt.plot(df1["time"], df1[var], label=name1, linewidth=0.8)
    plt.plot(df2["time"], df2[var], label=name2, linewidth=0.8)
    plt.title(var)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_variable_pdfs(df, name, cols=None, bins=40):
    """
    Plot empirical probability density functions for numeric variables.
    Uses histograms normalized to density.
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["time", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    n = len(cols)
    if n == 0:
        print(f"No numeric columns to plot for {name}.")
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
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

        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle(f"Empirical probability density functions — {name}", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_raw_vs_scaled_pdfs(raw_df, scaled_df, name, variable_pairs):
    if len(variable_pairs) == 0:
        print(f"No variable pairs to plot for {name}.")
        return
    n = len(variable_pairs)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(12, 3.2 * n),
        squeeze=False,
    )

    for i, (raw_col, scaled_col) in enumerate(variable_pairs):
        raw = pd.to_numeric(raw_df[raw_col], errors="coerce").dropna()
        scaled = pd.to_numeric(scaled_df[scaled_col], errors="coerce").dropna()

        axes[i, 0].hist(
            raw,
            bins=40,
            density=True,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )
        axes[i, 0].set_title(f"{raw_col} raw/log")

        axes[i, 1].hist(
            scaled,
            bins=40,
            density=True,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )
        axes[i, 1].axvline(scaled.mean(), linestyle="--", linewidth=1.5, label="mean")
        axes[i, 1].axvline(scaled.median(), linestyle=":", linewidth=1.5, label="median")
        axes[i, 1].set_title(f"{scaled_col} scaled")
        axes[i, 1].legend(fontsize=8)

    fig.suptitle(f"Raw/log vs scaled distributions — {name}", y=1.01)
    plt.tight_layout()
    plt.show()

def distribution_summary(df, name=None):
    cols = [
        c for c in df.columns
        if c not in ["time", "station"]
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

def run_teleseismic_checks(
    client,
    station,
    cfg,
    event_time=UTCDateTime("2008-05-12T06:28:00"),
    pre_sec=3600,
    post_sec=3600,
):
    if not isinstance(event_time, UTCDateTime):
        event_time = UTCDateTime(event_time)

    t1 = event_time - pre_sec
    t2 = event_time + post_sec

    st = client.get_waveforms(
        network=cfg["network"],
        station=station,
        location=cfg["location"],
        channel=cfg["channel"],
        starttime=t1,
        endtime=t2,
        attach_response=True,
    )

    st.remove_response(
        output=cfg.get("response_output", "VEL"),
        pre_filt=cfg.get("pre_filt", None),
    )
    tr = st[0]

    # -----------------------------
    # Raw waveform
    # -----------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(tr.times("matplotlib"), tr.data)
    plt.title(f"Raw waveform around teleseismic arrival — {station}")
    plt.show()

    # -----------------------------
    # 1-min RMS
    # -----------------------------
    df = pd.DataFrame({
        "time": pd.to_datetime(tr.times("timestamp"), unit="s", utc=True),
        "amp": tr.data
    }).set_index("time")

    rms_1min = (
        df["amp"]
        .pow(2)
        .rolling("60s")
        .mean()
        .pow(0.5)
        .resample("1min")
        .mean()
    )

    # -----------------------------
    # Hourly RMS comparison
    # -----------------------------
    # Use max to preserve the transient teleseismic arrival within the hour.
    rms_hourly = rms_1min.resample("1h").max()

    plt.figure(figsize=(12, 4))
    plt.plot(rms_1min.index, rms_1min, alpha=0.5, label="1-min RMS")
    plt.plot(rms_hourly.index, rms_hourly, marker="o", label="Hourly max RMS")
    plt.legend()
    plt.title(f"Effect of hourly max aggregation — {station}")
    plt.show()

    t_peak_1m = rms_1min.idxmax()
    t_peak_hourly = rms_hourly.idxmax()

    print(f"\n{station}")
    print("1-min peak:", t_peak_1m)
    print("hourly peak bin:", t_peak_hourly)
    print("difference:", t_peak_1m - t_peak_hourly)

    # -----------------------------
    # Spectrogram
    # -----------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    tr_spec = tr.copy()

    Pxx, freqs, bins, _ = ax.specgram(
        tr_spec.data,
        NFFT=2048,
        Fs=tr_spec.stats.sampling_rate,
        noverlap=1536
    )

    ax.clear()

    start = tr_spec.stats.starttime.datetime
    bin_times = np.array([start + pd.Timedelta(seconds=b) for b in bins])
    bin_times = mdates.date2num(bin_times)

    # Convert power to dB and clip color range for clearer contrast
    Pxx_db = 10 * np.log10(Pxx + 1e-20)

    vmin, vmax = np.nanpercentile(Pxx_db, [5, 99.5])

    mesh = ax.pcolormesh(
        bin_times,
        freqs,
        Pxx_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Power spectral density (dB)")

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    ax.axvline(
        mdates.date2num(event_time.datetime),
        color="cyan",
        linestyle="--",
        linewidth=2.5,
        label="EQ origin"
    )

    ax.axhline(0.2, color="cyan", linestyle=":", linewidth=2.2, label="T band upper")
    ax.axhline(4, color="yellow", linestyle=":", linewidth=2.2, label="S band upper")
    ax.axhline(12, color="red", linestyle=":", linewidth=2.2, label="Y band upper")
    ax.set_ylim(0, 15)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"Spectrogram around teleseismic arrival — Etna 2008-05-12 {station}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    return {
        "station": station,
        "trace": tr,
        "rms_1min": rms_1min,
        "rms_hourly": rms_hourly,
        "t_peak_1m": t_peak_1m,
        "t_peak_hourly": t_peak_hourly,
        "peak_diff": t_peak_1m - t_peak_hourly,
    }

def plot_final_dataset_with_event(csv_path, station, event_time, title=None):
    df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")

    if "station" in df.columns:
        df = df.drop(columns=["station"])

    variables = df.columns

    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(12, 2.5 * len(variables)),
        sharex=True,
    )

    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        ax.plot(df.index, df[var])
        ax.axvline(event_time, color="red", linestyle="--")
        ax.set_ylabel(var)

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title or f"Etna final dataset variables — {station}")

    plt.tight_layout()
    plt.show()