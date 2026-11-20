import matplotlib.pyplot as plt
import numpy as np


def curve(point_fn, labels, scores, thresholds=None):
    """Points (x, y) of the one-threshold metric point_fn(labels, alarms) at every threshold, from
    the highest (no alarm) to the lowest (all alarm), with alarms = scores > threshold.

    thresholds: None = every distinct score, then -inf.
    """
    if thresholds is None:
        thresholds = np.r_[np.unique(scores), -np.inf]
    points = [point_fn(labels, (scores > t).astype(int)) for t in np.sort(thresholds)[::-1]]
    x, y = np.array(points, dtype=float).T
    return x, y


def area_under_curve(x, y):
    """Trapezoid area under the points (x, y), in the order given by curve."""
    return np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)


def volume_under_surface(point_fn, labels, scores, param_name, param_values):
    """Area under the curve of point_fn for each value of param_name, averaged."""
    pass


def plot_curve(x, y, xlabel, ylabel, title):
    """The curve through the points (x, y), with the area under it shaded."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, lw=1.5, color="tab:blue")
    ax.fill_between(x, y, lw=0, color="tab:blue", alpha=0.15)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal")
    ax.grid(lw=0.5, color="0.9")
    fig.tight_layout()
    plt.show()
