import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_POINTS_PER_YEAR = {"daily": 365, "weekly": 52, "monthly": 12, "quarterly": 4}


def get_num_points_per_year(data_sampling_freq):
    """Number of data points in one year for a data sampling frequency."""
    if data_sampling_freq not in NUM_POINTS_PER_YEAR:
        raise ValueError(
            f"Unknown data_sampling_freq {data_sampling_freq!r}; "
            f"choose one of {tuple(NUM_POINTS_PER_YEAR)}"
        )
    return NUM_POINTS_PER_YEAR[data_sampling_freq]


def apply_recursive(data, fn, **kwargs):
    """
    Apply a function recursively through the dictionary data
    """

    if isinstance(data, (pd.Series, pd.DataFrame)):
        return fn(data, **kwargs)
    return {
        key: apply_recursive(value, fn, **kwargs)
        for key, value in data.items()
    }


def plot_offline(
    y, scores, alarms, threshold=None, anm_labels=None, title="Detection", y_train=None
):
    """A detector's output on y: y with the alarms (red), the scores against the threshold and, if
    anm_labels is given, the true anomaly (gray) to compare with the alarms.

    y_train, if given, is drawn in front of y, unscored; anm_labels then covers y_train then y.
    """
    fig, draw = _detection_figure(y, scores, alarms, threshold, anm_labels, title, y_train)
    fig.suptitle(f"{title}: {draw()} alarms")
    plt.show()


def plot_online(
    y, scores, alarms, threshold=None, anm_labels=None, title="Detection", y_train=None, step=1
):
    """The plot_offline figure drawn step by step from the first step of y: at t, the steps up to t
    only, i.e. what an online detector had seen and output at t. Redrawn every `step` steps;
    closing the window stops it."""
    fig, draw = _detection_figure(y, scores, alarms, threshold, anm_labels, title, y_train)
    start = 0 if y_train is None else len(y_train)
    for t in range(start, start + len(y)):
        if (t - start) % step and t < start + len(y) - 1:
            continue
        if not plt.fignum_exists(fig.number):
            return
        fig.suptitle(f"{title}  t = {t}: {draw(t + 1)} alarms so far")
        plt.pause(0.001)
    plt.show()


def _detection_figure(y, scores, alarms, threshold, anm_labels, title, y_train):
    """Figure of y_train then y, the scores and the alarms, left empty, and draw(end) that shows
    the steps before end (all if None) and returns their number of alarms. Bottom panel: the
    alarms against the true labels."""
    y_train = np.empty(0) if y_train is None else np.asarray(y_train, dtype=float)
    train_len = len(y_train)
    y = np.r_[y_train, np.asarray(y, dtype=float)]
    scores = np.r_[np.full(train_len, np.nan), np.asarray(scores, dtype=float)]
    alarms = np.r_[np.zeros(train_len, dtype=bool), np.asarray(alarms, dtype=bool)]
    steps = np.arange(len(y))

    fig, (ax_y, ax_s, ax_l) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 7), height_ratios=(2, 1, 0.5)
    )
    fig.suptitle(title)
    ax_l.set_yticks([1], ["alarm"])
    if anm_labels is not None:
        labels = np.asarray(anm_labels)
        if len(labels) != len(y):
            raise ValueError(
                f"anm_labels must cover y_train then y: {len(y)} steps, got {len(labels)}"
            )
        runs = np.flatnonzero(np.diff(np.r_[0, labels, 0])).reshape(-1, 2)  # [start, end) of 1s
        for ax in (ax_y, ax_s):
            for start, end in runs:
                ax.axvspan(start - 0.5, end - 0.5, color="0.88", lw=0, label="true anomaly")
        ax_l.plot(np.flatnonzero(labels), np.zeros(labels.sum()), "|", ms=10, color="0.5")
        ax_l.set_yticks([0, 1], ["true", "alarm"])
    (y_line,) = ax_y.plot([], [], lw=0.8, color="0.2", label="y")
    (y_alarms,) = ax_y.plot([], [], "o", ms=4, color="tab:red", label="alarm")
    (score_line,) = ax_s.plot([], [], lw=0.8, color="tab:blue", label="score")
    (score_alarms,) = ax_s.plot([], [], "o", ms=4, color="tab:red", label="alarm")
    if threshold is not None:
        ax_s.axhline(threshold, ls="--", lw=1, color="0.2", label="threshold")
    (lane_alarms,) = ax_l.plot([], [], "|", ms=10, color="tab:red")
    if train_len:
        for ax in (ax_y, ax_s, ax_l):
            ax.axvline(train_len - 0.5, ls=":", lw=1.2, color="0.2", label="end of training")

    def padded_range(values):  # fixed limits, so the axes don't move while drawing
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0, 1
        pad = 0.05 * np.ptp(values) or 0.5
        return values.min() - pad, values.max() + pad

    ax_y.set(ylabel="y", ylim=padded_range(y))
    score_range = np.r_[scores, np.nan if threshold is None else threshold]
    ax_s.set(ylabel="score", ylim=padded_range(score_range))
    ax_l.set(xlabel="time step t", xlim=(-0.5, len(y) - 0.5), ylim=(-0.6, 1.6))
    for ax in (ax_y, ax_s):
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=5, fontsize=8, frameon=False)

    def draw(end=None):
        seen, on = steps[:end], alarms[:end]
        scored = np.isfinite(scores[:end])  # e.g. NaN in the training part or between refits
        y_line.set_data(seen, y[:end])
        y_alarms.set_data(seen[on], y[:end][on])
        score_line.set_data(seen[scored], scores[:end][scored])
        score_alarms.set_data(seen[on], scores[:end][on])
        lane_alarms.set_data(seen[on], np.ones(on.sum()))
        return on.sum()

    return fig, draw
