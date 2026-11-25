"""Regime x Online metrics, over a set of test series.

anm_labels_test, anm_labels_pred: one 0/1 array per series, test part only.
Detection: the first alarm at or after the anomaly start, within max_time_to_detection years
(np.inf = no maximum); from then on the new regime is normal.
Rates are pooled (alarms and times summed over the series, then divided); NaN if nothing to divide.
"""

import numpy as np

from tsbenchmark.evaluation_metrics._utils import anomaly_start_and_first_detection
from tsbenchmark.utils import get_num_points_per_year


def time_to_detection(
    anm_labels_test, anm_labels_pred, data_sampling_freq, max_time_to_detection=5
):
    """Mean years from the anomaly start to its detection, over the detected series."""
    num_points_per_year = get_num_points_per_year(data_sampling_freq)
    max_delay = max_time_to_detection * num_points_per_year
    delays = []
    for labels, alarms in zip(anm_labels_test, anm_labels_pred):
        start, first_detection = anomaly_start_and_first_detection(labels, alarms, max_delay)
        if first_detection is not None:
            delays.append(first_detection - start)
    return np.mean(delays) / num_points_per_year if delays else np.nan


def detection_probability(
    anm_labels_test, anm_labels_pred, data_sampling_freq, max_time_to_detection=5
):
    """Proportion of the series with an anomaly where it is detected."""
    max_delay = max_time_to_detection * get_num_points_per_year(data_sampling_freq)
    detected = []
    for labels, alarms in zip(anm_labels_test, anm_labels_pred):
        start, first_detection = anomaly_start_and_first_detection(labels, alarms, max_delay)
        if start is not None:
            detected.append(first_detection is not None)
    return np.mean(detected) if detected else np.nan


def false_alarm_rate(anm_labels_test, anm_labels_pred, data_sampling_freq):
    """False alarms per 10 years: alarms before the anomaly start / test time before it (the
    whole test part if no anomaly)."""
    num_false_alarms, num_steps = 0, 0
    for labels, alarms in zip(anm_labels_test, anm_labels_pred):
        start, _ = anomaly_start_and_first_detection(labels, alarms)
        end = len(labels) if start is None else start
        num_false_alarms += np.sum(alarms[:end])
        num_steps += end
    if num_steps == 0:
        return np.nan
    return num_false_alarms / num_steps * 10 * get_num_points_per_year(data_sampling_freq)


def repeated_alarm_rate(
    anm_labels_test, anm_labels_pred, data_sampling_freq, max_time_to_detection=5
):
    """Repeated alarms per 10 years: alarms after the detection / time from the detection to the
    end, over the detected series."""
    num_points_per_year = get_num_points_per_year(data_sampling_freq)
    max_delay = max_time_to_detection * num_points_per_year
    num_repeated_alarms, num_steps = 0, 0
    for labels, alarms in zip(anm_labels_test, anm_labels_pred):
        _, first_detection = anomaly_start_and_first_detection(labels, alarms, max_delay)
        if first_detection is not None:
            num_repeated_alarms += np.sum(alarms[first_detection + 1:])
            num_steps += len(alarms) - 1 - first_detection
    if num_steps == 0:
        return np.nan
    return num_repeated_alarms / num_steps * 10 * num_points_per_year
