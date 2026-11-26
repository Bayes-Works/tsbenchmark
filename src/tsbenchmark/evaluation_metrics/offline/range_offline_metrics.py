"""Range x Offline metrics: affiliation F1."""

import numpy as np

from tsbenchmark.evaluation_metrics._utils import events, plot_vs_magnitude


def affiliation_f1(anm_labels_test, anm_labels_pred, anm_mags, plot=False):
    """Affiliation F1 (Huet et al., KDD 2022) for each magnitude: {magnitude: F1}, the harmonic
    mean of the affiliation precision and recall; 0 if there is no prediction. plot: draw it
    against the magnitude.

    *** Need verification. ***
    """
    anm_mags = np.asarray(anm_mags)
    f1s = {}
    for mag in np.unique(anm_mags):
        precisions, recalls = [], []
        for i in np.flatnonzero(anm_mags == mag):
            alarms = np.asarray(anm_labels_pred[i])
            t = (np.arange(4 * len(alarms)) + 0.5) / 4  # quarter-step midpoints
            gt, pred = events(anm_labels_test[i]), events(alarms)
            borders = np.r_[0, (gt[:-1, 1] + gt[1:, 0]) / 2, len(alarms)]  # zones [A, B)
            for (a, b), A, B in zip(gt, borders[:-1], borders[1:]):
                m = min(a - A, B - b)
                x = t[(t >= A) & (t < B) & (alarms[t.astype(int)] > 0)]  # predictions in the zone
                if len(x):
                    d = np.maximum(0, np.maximum(a - x, x - b))
                    survival = 1 - (b - a + np.minimum(d, m) + d) / (B - A)
                    precisions.append(np.mean(np.where(d == 0, 1, survival)))
                starts, stops = np.maximum(pred[:, 0], A), np.minimum(pred[:, 1], B)
                starts, stops = starts[starts < stops], stops[starts < stops]
                if len(starts) == 0:  # no prediction in the zone: infinite distance
                    recalls.append(0.0)
                    continue
                y = t[(t >= a) & (t < b)]
                d = np.maximum(0, np.maximum(starts - y[:, None], y[:, None] - stops)).min(1)
                m_y = np.minimum(y - A, B - y)
                recalls.append(np.mean(1 - (np.minimum(d, m_y) + d) / (B - A)))
        precision = np.mean(precisions) if precisions else np.nan
        recall = np.mean(recalls) if recalls else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s[float(mag)] = float(f1)
    if plot:
        plot_vs_magnitude(f1s, "affiliation F1", ylim=(0, 1.05))
    return f1s
