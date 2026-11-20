"""Gaussian naive detector: an offline detector.

At each step t, a Gaussian is fitted to all the points in the training set defined by the user.
The score of y_t is the confidence level of the narrowest central interval of that Gaussian that
still contains y_t: 2 * Phi(|y_t - mean| / std) - 1, in [0, 1]. The threshold is a confidence
level: with threshold=0.95, an alarm is raised when y_t falls outside the central 95% interval.

Usage:
    detector = GaussianNaiveOfflineDetector(threshold=0.95).fit(y[:train_len])
    labels = detector.detect(y[train_len:])
"""

from math import erf, sqrt

import numpy as np

from tsbenchmark.detector.base_detector import OfflineDetector


class GaussianNaiveOfflineDetector(OfflineDetector):
    """Alarm when y_t is outside the central `threshold` interval of a Gaussian fitted to the
    points in the training set.

    The std is the sample std (divided by n - 1), so at least 2 training points are needed.
    """

    name = "GaussianNaive"
    learning = "semi-supervised"  # fitted once on the anomaly-free training part, never updated

    def __init__(self, threshold=None):
        if threshold is not None and not 0 <= threshold < 1:
            raise ValueError(f"threshold is a confidence level in [0, 1), got {threshold}")
        super().__init__(threshold=threshold)
        self.n, self.mean, self.std = 0, 0.0, 0.0  # count, mean and std of the training part

    def _fit(self, y_train, labels_train=None):
        """Fit the Gaussian to the training part (from scratch at each fit)."""
        self.n = len(y_train)
        self.mean = float(np.mean(y_train)) if self.n > 0 else 0.0
        self.std = float(np.std(y_train, ddof=1)) if self.n > 1 else 0.0  # sample std

    def score(self, y):
        deviation = np.abs(np.asarray(y, dtype=float) - self.mean)
        if self.n < 2:
            return np.full(len(deviation), np.nan)  # too few points for a std
        if self.std == 0:
            return (deviation > 0).astype(float)  # zero-width Gaussian: any deviation is beyond every interval
        return np.vectorize(erf)(deviation / (self.std * sqrt(2)))  # = 2 * Phi(|Z|) - 1 = P(|Z| <= |y_t - mean| / std) = confidence level of the narrowest central interval containing y_t
