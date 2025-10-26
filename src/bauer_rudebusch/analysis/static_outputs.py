"""
Plotting routines corresponding to the static figures in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .. import data as data_module
from ..util import plot_recessions


def _configure_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(False)


def generate_data_figures(output_dir: Path | str = "figures") -> None:
    """
    Replicate Figures 1 and 2 from `plot_data.R`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = data_module.load_data()
    data["istar"] = data["pistar.ptr"] + data["rstar.mean"]

    date = data["date"]
    y10 = data["y10"]
    pistar = data["pistar.ptr"]
    rstar_mean = data["rstar.mean"]
    istar = data["istar"]

    fig1, ax1 = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    yrange = (min(0, y10.min()), y10.max())
    ax1.set_ylim(yrange)
    ax1.set_ylabel("Percent")
    ax1.plot(date, y10, label="Ten-year yield", color="black", linewidth=2)
    plot_recessions(ax1, (date.iloc[0], date.iloc[-1]), yrange, alpha=0.3)
    ax1.plot(date, pistar, label=r"Trend inflation, $\pi^\ast$", color="red", linewidth=2)
    ax1.plot(date, rstar_mean, label=r"Equilibrium real short rate, $r^\ast$", color="green", linewidth=2)
    ax1.plot(date, istar, label=r"Equilibrium short rate, $i^\ast$", color="steelblue", linewidth=2)
    _configure_time_axis(ax1)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_xlabel("")
    fig1.savefig(output_dir / "data1.pdf")
    plt.close(fig1)

    # Figure 2
    smoothed_names = data.attrs["nms.sm"]
    all_names = data.attrs["nms.all"]
    rstars_sm = data[smoothed_names].to_numpy(dtype=float)
    rstars_all = data[all_names].to_numpy(dtype=float)
    range_min_sm = np.nanmin(rstars_sm, axis=1)
    range_max_sm = np.nanmax(rstars_sm, axis=1)
    range_min_all = np.nanmin(rstars_all, axis=1)
    range_max_all = np.nanmax(rstars_all, axis=1)
    yrange2 = (
        float(np.nanmin(np.concatenate([rstars_sm.flatten(), rstars_all.flatten()]))),
        float(np.nanmax(np.concatenate([rstars_sm.flatten(), rstars_all.flatten()]))),
    )

    fig2, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(7, 5), constrained_layout=True, sharey=True
    )

    ax_left.fill_between(date, range_min_sm, range_max_sm, color="lightblue", alpha=0.5)
    ax_left.plot(date, data["rstar.sm"], color="black", linewidth=2, label="Avg. smoothed r*")
    ax_left.axhline(0, linestyle="--", color="gray")
    ax_left.set_ylabel("Percent")
    ax_left.set_title("Smoothed estimates", fontsize=11)
    plot_recessions(ax_left, (date.iloc[0], date.iloc[-1]), yrange2, alpha=0.3)
    _configure_time_axis(ax_left)
    ax_left.legend(loc="upper right", frameon=True)

    ax_right.fill_between(date, range_min_all, range_max_all, color="lightblue", alpha=0.5)
    ax_right.plot(date, data["rstar.filt"], color="black", linewidth=2, label="Avg. filtered r*")
    ax_right.plot(date, data["rstar.realtime"], color="orange", linewidth=2, label="Avg. real-time r*")
    ax_right.plot(date, data["rr.ewma"], color="green", linewidth=2, label="Moving-average r*")
    ax_right.axhline(0, linestyle="--", color="gray")
    ax_right.set_title("Filtered and real-time estimates", fontsize=11)
    plot_recessions(ax_right, (date.iloc[0], date.iloc[-1]), yrange2, alpha=0.3)
    _configure_time_axis(ax_right)
    ax_right.legend(loc="upper right", frameon=True)

    fig2.savefig(output_dir / "data2.pdf")
    plt.close(fig2)


def generate_cycle_figure(output_dir: Path | str = "figures") -> None:
    """
    Replicate Figure 3 from `plot_cycles.R`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = data_module.load_data()
    data["rstar"] = data["rstar.realtime"]
    data["istar"] = data["pistar.ptr"] + data["rstar"]

    demean = lambda series: series - series.mean()
    data["resid1"] = data["y10"] - data["pistar.ptr"] - data["rstar"]

    # DOLS residual using istar
    p = 4
    data["d_istar"] = data["istar"].diff()
    regressors = ["istar"]
    def _lag_name(lag: int) -> str:
        return f"d_istar_shift_{lag:+d}".replace("+", "p").replace("-", "m")

    for lag in range(-p, p + 1):
        if lag == 0:
            continue
        shifted = data["d_istar"].shift(lag)
        col_name = _lag_name(lag)
        data[col_name] = shifted
        regressors.append(col_name)

    valid = data.dropna(subset=["y10"] + regressors)
    X = np.column_stack([np.ones(len(valid))] + [valid[col].to_numpy() for col in regressors])
    beta = np.linalg.lstsq(X, valid["y10"], rcond=None)[0]
    fitted = X @ beta
    data.loc[valid.index, "resid2"] = valid["y10"] - fitted

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    demean_y10 = demean(data["y10"])
    yrange = (demean_y10.min(), demean_y10.max())
    ax.plot(data["date"], demean_y10, color="black", linewidth=2, label="Ten-year yield, demeaned")
    ax.plot(data["date"], data["resid1"], color="red", linewidth=2, label="Difference between yield and i*")
    ax.plot(data["date"], data["resid2"], color="steelblue", linewidth=2, label="Cointegration residual between yield and i*")
    ax.axhline(0, linestyle="--", color="gray")
    plot_recessions(ax, (data["date"].iloc[0], data["date"].iloc[-1]), yrange, alpha=0.3)
    ax.set_ylabel("Percent")
    ax.set_xlabel("")
    _configure_time_axis(ax)
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(output_dir / "cycles.pdf")
    plt.close(fig)
