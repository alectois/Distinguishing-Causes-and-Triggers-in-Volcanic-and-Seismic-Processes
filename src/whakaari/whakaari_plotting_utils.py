import pandas as pd
import matplotlib.pyplot as plt


def dataset_health_report(df, name):
    print(f"\n{name}")
    print("shape:", df.shape)

    if isinstance(df.index, pd.DatetimeIndex):
        print("duplicate timestamps:", df.index.duplicated().sum())
        print("time sorted:", df.index.is_monotonic_increasing)
    elif "timestamp" in df.columns:
        print("duplicate timestamps:", df["timestamp"].duplicated().sum())
        print("time sorted:", df["timestamp"].is_monotonic_increasing)
    else:
        print("No timestamp index/column found.")

    print("\nMissing fraction:")
    print(df.isna().mean().sort_values())

    return df.describe()


def plot_whakaari_waveforms(df):
    cols = [
        "hydro_rms_2_5",
        "ratio_4p5_8_over_8_16",
        "hf_event_rate_2_5",
        "effect_tremor_rms_5_15",
    ]

    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No waveform columns found.")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 8), sharex=True)

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], linewidth=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_whakaari_external(df):
    cols = ["SO2_flux", "GNSS_deformation", "API", "pressure_drop"]
    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No external columns found.")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.8 * len(cols)), sharex=True)

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], linewidth=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_whakaari_scaled(df):
    cols = [c for c in df.columns if c != "eruption"]

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], linewidth=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_with_eruption_time(df, cols, eruption_time):
    eruption_time = pd.to_datetime(eruption_time, utc=True)
    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No requested columns found.")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.8 * len(cols)), sharex=True)

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], linewidth=0.8)
        ax.axvline(eruption_time, color="red", linestyle="--", alpha=0.7)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
