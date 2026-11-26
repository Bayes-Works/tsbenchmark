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


def anomaly_start_and_first_detection(labels, alarms, max_delay=np.inf):
    """Index of the anomaly start (first nonzero label) and of the first alarm at or after it, at
    most max_delay steps after it; None for what does not exist (no anomaly, or no such alarm)."""
    anomaly = np.flatnonzero(labels)
    if len(anomaly) == 0:
        return None, None
    start = anomaly[0]
    delays = np.flatnonzero(alarms[start:])
    return start, (start + delays[0] if len(delays) and delays[0] <= max_delay else None)


def events(labels):
    """Events [start, stop) of the runs of 1 in a 0/1 array, step i being the interval [i, i + 1):
    an array of shape (number of events, 2). Function intended for the F1 affiliation metric."""
    edges = np.diff(np.r_[0, np.asarray(labels) > 0, 0].astype(int))
    return np.c_[np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)]


def plot_vs_magnitude(values, ylabel, ylim=None, std=None):
    """The target values (+/- 1 std) against the magnitude."""
    mags, mean = np.array(list(values)), np.array(list(values.values()))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mags, mean, "o-", color="tab:blue", label="mean" if std else None)
    if std:
        s = np.array([std[mag] for mag in values])
        ax.fill_between(
            mags, mean - s, mean + s, color="tab:blue", alpha=0.2, lw=0, label="± 1 std"
        )
    ax.set(xlabel="anomaly magnitude", ylabel=ylabel, ylim=ylim)
    ax.grid(lw=0.5, color="0.9")
    fig.tight_layout()
    plt.show()


def plot_surface(curves, y, xlabel, ylabel, zlabel, title):
    """Surface through the curves [(x, z), ...]."""
    log_x = [np.log10(1 + x) for x, _ in curves]
    grid = np.linspace(0, max(lx.max() for lx in log_x), 300)
    Z = []
    for lx, (_, z) in zip(log_x, curves):
        lx_unique, last = np.unique(lx[::-1], return_index=True)
        Z.append(np.interp(grid, lx_unique, z[::-1][last]))
    X, Y = np.meshgrid(grid, y)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, np.array(Z), cmap="viridis", rcount=len(y), ccount=len(grid))
    ticks = [t for t in (0, 1, 10, 100, 1000) if np.log10(1 + t) <= grid[-1]]
    ax.set_xticks(np.log10(1 + np.array(ticks)), ticks)
    ax.set(xlabel=f"{xlabel} (log scale)", ylabel=ylabel, zlabel=zlabel, title=title)
    plt.show()


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
