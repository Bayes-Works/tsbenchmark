from statistics import NormalDist

import numpy as np


# Dummy detector: fit a Gaussian to y_{1:t-1}, alarm if y_t is outside its
# `confidence` interval
def gaussian_point_detector(y, confidence=0.95, min_history=2):
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    alarms = np.zeros(len(y), dtype=int)
    for t in range(min_history, len(y)):
        mu = np.mean(y[:t])
        sigma = np.std(y[:t], ddof=1)
        if abs(y[t] - mu) > z * sigma:
            alarms[t] = 1
    return alarms


# Random detector: alarm with probability alarm_prob, independent of the data
# and of the labels
def random_detector(n, alarm_prob, rng):
    return (rng.random(n) < alarm_prob).astype(int)
