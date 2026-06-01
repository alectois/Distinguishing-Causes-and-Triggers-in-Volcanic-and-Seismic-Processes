import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

def _save_current_figure(fig, filename, save_dir="figures"):
    """Save a displayed plotting figure to figures/ as PNG."""
    if save_dir is None or filename is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    fig.savefig(
        out_dir / f"{stem}.png",
        bbox_inches="tight",
        pad_inches=0.03,
        dpi=300,
        facecolor="white",
    )
        
WHAKAARI_ALL_COLS = [
    "hydro_rms_2_5",
    "ratio_4p5_8_over_8_16",
    "hf_event_rate_2_5",
    "effect_tremor_rms_5_15",
    "API",
    "pressure_drop",
    "SO2_flux",
    "GNSS_deformation",
    "local_eq_count_1h",
    
]

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

def plot_with_eruption_time(
    csv_path,
    cols,
    eruption_time,
    title=None,
    figsize=None,
    tick_interval_days=7,
    save_dir="figures",
    filename=None,
):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
    eruption_time = pd.to_datetime(eruption_time, utc=True)
    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print("No requested columns found.")
        return

    label_map = {
        "hydro_rms_2_5": ("Hydrothermal tremor RMS, 2–5 Hz", "RMS"),
        "ratio_4p5_8_over_8_16": ("Spectral ratio, 4.5–8 / 8–16 Hz", "Ratio"),
        "hf_event_rate_2_5": ("High-frequency event rate, 2–5 Hz", "Events per hour"),
        "effect_tremor_rms_5_15": ("Tremor response RMS (Effect), 5–15 Hz", "RMS"),
        "API": ("Antecedent precipitation index", "mm"),
        "pressure_drop": ("Atmospheric pressure drop", "hPa"),
        "GNSS_deformation": ("GNSS deformation", "m"),
        "local_eq_count_1h": ("Local earthquakes, 1h count", "Count"),
        "SO2_flux": ("SO₂ flux", "t d⁻¹"),
    }

    series_color = "#0072B2"
    eruption_color = "#D55E00"

    if figsize is None:
        figsize = (14, 2.15 * len(cols))

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=figsize,
        sharex=True,
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        panel_title, ylabel = label_map.get(col, (col, "Value"))

        if col == "SO2_flux":
            ax.scatter(
                df.index,
                df[col],
                s=9,
                color=series_color,
                alpha=0.80,
                linewidths=0,
            )
        else:
            ax.plot(
                df.index,
                df[col],
                color=series_color,
                linewidth=0.75,
            )

        ax.axvline(
            eruption_time,
            color=eruption_color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )

        ax.set_title(panel_title, loc="left", fontsize=10, pad=3)
        ax.set_ylabel(ylabel, fontsize=9)

        ax.grid(True, alpha=0.22, linewidth=0.6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="both", labelsize=9)

    axes[0].legend(
        ["Eruption"],
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )

    # The legend above can attach to the wrong artist if SO2 is first,
    # so manually create a clean event legend.
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0], [0],
                color=eruption_color,
                linestyle="--",
                linewidth=1.1,
                label="Eruption",
            )
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    locator = mdates.DayLocator(interval=tick_interval_days)
    formatter = mdates.DateFormatter("%b %d")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel("Time (UTC)", fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    plt.tight_layout()
    if filename is None:
        filename = "whakaari_variables"
    _save_current_figure(fig, filename, save_dir)

    plt.show()

def plot_variable_pdfs(df, name, cols=None, bins=40, filename=None, save_dir="figures"):
    """
    Plot empirical probability density functions for numeric Whakaari variables.
    Uses density-normalized histograms.
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if c not in ["timestamp", "time", "station"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    cols = [c for c in cols if c in df.columns]

    if len(cols) == 0:
        print(f"No numeric columns to plot for {name}.")
        return

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.4 * nrows),
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

        ax.axvline(x.mean(), linestyle="--", linewidth=1.4, label="mean")
        ax.axvline(x.median(), linestyle=":", linewidth=1.4, label="median")

        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle(f"{name}", y=1.02)
    plt.tight_layout()
    if filename is None:
        filename = f"{name}_variable_pdfs"
    _save_current_figure(fig, filename, save_dir)
    plt.show()


def distribution_summary(df, name=None):
    """
    Summary statistics useful for checking skewness, tails, constants, and missingness.
    """
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