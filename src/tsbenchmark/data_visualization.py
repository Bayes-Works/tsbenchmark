import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.transforms import ScaledTranslation

DOUBLE_COL = (6.5, 3.5)


def common_anomalies(result, tolerance):
    """
    Anomalies detected by all methods, allowing a `tolerance` time difference between methods.

    A timestamp is a common anomaly when at least one method flags it and every method has
    an anomaly within `tolerance` of it. Returns a boolean series on the union of the indexes.
    """
    index = result[next(iter(result))].index
    for res in result.values():
        index = index.union(res.index)

    detected = np.zeros(len(index), dtype=bool)
    confirmed = np.ones(len(index), dtype=bool)
    for res in result.values():
        times = res.index[res["value"].to_numpy(dtype=bool)]
        detected |= index.isin(times)
        if len(times) == 0:
            confirmed[:] = False
            continue
        # Distance from each timestamp to the nearest anomaly of this method
        pos = times.searchsorted(index)
        previous = times[np.clip(pos - 1, 0, len(times) - 1)]
        following = times[np.clip(pos, 0, len(times) - 1)]
        distance = np.minimum(np.abs(index - previous), np.abs(index - following))
        confirmed &= distance <= tolerance

    return pd.Series(detected & confirmed, index=index)


def _shade_anomalies(ax, index, anomaly, label):
    """
    Red shaded areas over the full axis height where `anomaly` is True. Each anomalous
    timestamp covers one time step centered on it, so a single-point anomaly stays visible.
    """
    anomaly = np.asarray(anomaly, dtype=bool)
    half_step = pd.Series(index).diff().median() / 2
    # Start and end positions of each run of consecutive anomalous timestamps
    edges = np.flatnonzero(np.diff(np.concatenate(([False], anomaly, [False])).astype(int)))
    for first, end in zip(edges[::2], edges[1::2]):
        ax.axvspan(
            index[first] - half_step,
            index[end - 1] + half_step,
            color="tab:red",
            alpha=0.45,
            linewidth=0,
            label=label,
        )
        label = None  # one legend entry for all the areas


def _legend_below(ax):
    """Legend outside the subplot, centered below the x tick labels."""
    handles, _ = ax.get_legend_handles_labels()
    below = ax.transAxes + ScaledTranslation(0, -20 / 72, ax.figure.dpi_scale_trans)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=below,
        ncol=len(handles),
        fontsize=7,
        frameon=False,
    )


def plot_test_results(
    dataframe,
    result,
    method_params,
    tolerance,
    major_locator=None,
    minor_locator=None,
):
    """
    Plot the anomaly detection results. The top subplot shows the anomalies common to all
    methods, followed by one subplot per method.

    Args:
        dataframe: time series with a "values" column.
        result: dict {method name: dataframe returned by the detector's `anomaly_score`}.
        method_params: dict {method name: dict of settings needed by the plot}.
            Chronos2 needs `quantile_levels` and `horizon`.
        tolerance: time difference allowed between methods for an anomaly to be common.
        major_locator: matplotlib date locator for the major x ticks,
            e.g. `mdates.YearLocator(2)`. Matplotlib's default when None.
        minor_locator: matplotlib date locator for the minor x ticks,
            e.g. `mdates.YearLocator(1)`. No minor ticks when None.

    Returns:
        (figure, axes): the figure and the array of its subplots, top to bottom.
    """
    methods = list(result)
    figure, axes = plt.subplots(
        len(methods) + 1,
        1,
        figsize=(DOUBLE_COL[0], DOUBLE_COL[1] * (len(methods) + 1) / 2),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]

    # Top subplot: data and the anomalies detected by all methods
    common = common_anomalies(result, tolerance)
    ax = axes[0]
    ax.plot(dataframe.index, dataframe["values"], color="tab:red", label="Data")
    _shade_anomalies(ax, common.index, common, "Common anomaly")
    ax.set_ylabel("y")
    ax.set_title(f"Common anomalies (tolerance: {tolerance.days} days)")
    ax.tick_params(labelbottom=True)
    _legend_below(ax)

    for ax, method in zip(axes[1:], methods):
        res = result[method]
        if method == "TimeRCD":
            ax.plot(dataframe.index, dataframe["values"], color="tab:red", label="Data")
            ax.set_ylabel("y")
        elif method == "Chronos2":
            quantile_levels = method_params[method]["quantile_levels"]
            horizon = method_params[method]["horizon"]
            ax.plot(dataframe.index, dataframe["values"], color="tab:red", label="Data")
            ax.plot(res.index, res["median"], color="tab:green", label="Prediction")
            ax.fill_between(
                res.index,
                res["lower"],
                res["upper"],
                color="tab:green",
                alpha=0.3,
                linewidth=0,
                label=f"Quantiles {quantile_levels[0]}--{quantile_levels[-1]}",
            )
            # Dashed lines at the start of each forecasting horizon (same windows as the detector)
            index = res.index
            start = int(res["median"].notna().to_numpy().argmax())
            while start < len(index):
                ax.axvline(index[start], color="k", linestyle="dotted", linewidth=0.5)
                end = int(index.searchsorted(index[start] + horizon))
                start = max(end, start + 1)
            ax.set_ylabel("y")
        elif method == "W1ACAS":
            ax.plot(dataframe.index, dataframe["values"], color="tab:red", label="Data")
            ax.set_ylabel("y")
        _shade_anomalies(ax, res.index, res["value"], "Anomaly")
        ax.set_title(method)
        ax.tick_params(labelbottom=True)
        _legend_below(ax)

    # The subplots share the x axis, so setting the ticks on one applies to all of them
    if major_locator is not None:
        axes[0].xaxis.set_major_locator(major_locator)
        axes[0].xaxis.set_major_formatter(mdates.AutoDateFormatter(major_locator))
    if minor_locator is not None:
        axes[0].xaxis.set_minor_locator(minor_locator)

    return figure, axes
