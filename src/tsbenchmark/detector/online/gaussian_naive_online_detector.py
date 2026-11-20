"""Gaussian naive detector: an online detector.

At each step t, a Gaussian is fitted to all the points before t: the training part given to fit,
then y[:t] (running mean and std). The score of y_t is the confidence level of the narrowest
central interval of that Gaussian that still contains y_t: 2 * Phi(|y_t - mean| / std) - 1, in
[0, 1]. The threshold is a confidence level: with threshold=0.95, an alarm is raised when y_t
falls outside the central 95% interval.

Usage:
    detector = GaussianNaiveOnlineDetector(threshold=0.95).fit(y[:train_len])
    labels = detector.detect(y[train_len:])
"""

from math import erf, sqrt

import numpy as np

from src.tsbenchmark.detector.base_detector import OnlineDetector


class GaussianNaiveOnlineDetector(OnlineDetector):
    """Alarm when y_t is outside the central `threshold` interval of a Gaussian fitted to the
    points before it: the training part, then y[:t].

    The std is the sample std (divided by n - 1), so at least 2 points are needed to score.
    """

    name = "GaussianNaive"
    learning = "semi-supervised"  # fitted on the anomaly-free training part, then updated online

    def __init__(self, threshold=None):
        if threshold is not None and not 0 <= threshold < 1:
            raise ValueError(f"threshold is a confidence level in [0, 1), got {threshold}")
        super().__init__(threshold=threshold)
        self.n, self.mean, self.m2 = 0, 0.0, 0.0  # count, mean, sum of squared deviations so far

    def _fit(self, y_train, labels_train=None):
        """Start the running mean and std from the training part (from scratch at each fit)."""
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        for y_t in y_train:
            self.update(y_t)

    def score_one(self, y_t):
        if self.n < 2:
            return np.nan  # too few points for a std
        std, deviation = sqrt(self.m2 / (self.n - 1)), abs(y_t - self.mean)
        if std == 0:
            return float(deviation > 0)  # zero-width Gaussian: any deviation is beyond every interval
        return erf(deviation / (std * sqrt(2)))  # = 2 * Phi(|Z|) - 1 = P(|Z| <= |y_t - mean| / std) =confidence level of the narrowest central interval containing y_t

    def update(self, y_t):
        """Welford's update of the running mean and sum of squared deviations."""
        self.n += 1
        delta = y_t - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (y_t - self.mean)
