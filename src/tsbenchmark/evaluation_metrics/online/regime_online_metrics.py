"""Regime x Online metrics, over a set of test series: false_alarm_rate, repeated_alarm_rate, 
time_to_detection, detection_probability, vus_amoc."""

import numpy as np

from tsbenchmark.evaluation_metrics._utils import (
    anomaly_start_and_first_detection,
    area_under_curve,
    plot_surface,
    plot_vs_magnitude,
)
from tsbenchmark.utils import get_num_points_per_year


def time_to_detection(
    anm_labels_test,
    anm_labels_pred,
    anm_mags,
    data_sampling_freq,
    max_time_to_detection=5,
    plot=False,
):
    """Years from the anomaly start to its detection."""
    num_points_per_year = get_num_points_per_year(data_sampling_freq)
    max_delay = max_time_to_detection * num_points_per_year
    anm_mags = np.asarray(anm_mags)
    years, years_std = {}, {}
    for mag in np.unique(anm_mags):
        delays = []
        for i in np.flatnonzero(anm_mags == mag):
            start, first_detection = anomaly_start_and_first_detection(
                anm_labels_test[i], anm_labels_pred[i], max_delay
            )
            if first_detection is not None:
                delays.append(first_detection - start)
        years[float(mag)] = float(np.mean(delays)) / num_points_per_year if delays else np.nan
        years_std[float(mag)] = float(np.std(delays)) / num_points_per_year if delays else np.nan
    if plot:
        plot_vs_magnitude(years, "time to detection (years)", ylim=(0, None), std=years_std)
    return years


def detection_probability(
    anm_labels_test,
    anm_labels_pred,
    anm_mags,
    data_sampling_freq,
    max_time_to_detection=5,
    plot=False,
):
    """Proportion of the series where the anomaly is detected."""
    max_delay = max_time_to_detection * get_num_points_per_year(data_sampling_freq)
    anm_mags = np.asarray(anm_mags)
    probabilities = {}
    for mag in np.unique(anm_mags):
        detected = []
        for i in np.flatnonzero(anm_mags == mag):
            start, first_detection = anomaly_start_and_first_detection(
                anm_labels_test[i], anm_labels_pred[i], max_delay
            )
            if start is not None:
                detected.append(first_detection is not None)
        probabilities[float(mag)] = float(np.mean(detected)) if detected else np.nan
    if plot:
        plot_vs_magnitude(probabilities, "detection probability", ylim=(0, 1.05))
    return probabilities


def false_alarm_rate(anm_labels_test, anm_labels_pred, data_sampling_freq):
    """False alarms per 10 years: alarms before the anomaly start / test time before it."""
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
    """Repeated alarms per 10 years: alarms after the detection / time from the detection to the end."""
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


def vus_amoc(
    anm_labels_test,
    anm_scores_pred,
    anm_mags,
    data_sampling_freq,
    max_time_to_detection=5,
    plot=False,
):
    """VUS-AMOC: volume under the surface of mean years to detection over false alarms per 10
    years and anomaly magnitude; lower is better. A miss counts as max_time_to_detection (finite),
    so a detector that misses anomalies gets close to it; the detection probability is not
    reported separately.

    anm_mags: the magnitude of each series, one AMOC curve per distinct value. Each curve sweeps
    the threshold over every distinct score from no alarm to an alarm at every scored step. 

    Limitations:
    - the scores must not depend on the threshold.
    """
    num_points_per_year = get_num_points_per_year(data_sampling_freq)
    max_delay = int(max_time_to_detection * num_points_per_year)
    anm_mags = np.asarray(anm_mags)
    mags = np.unique(anm_mags)
    curves = []
    for mag in mags:
        before_start, windows = [], []
        for i in np.flatnonzero(anm_mags == mag):
            scores = np.where(np.isnan(anm_scores_pred[i]), -np.inf, anm_scores_pred[i])
            start = np.flatnonzero(anm_labels_test[i])[0]
            before_start.append(scores[:start])
            windows.append(scores[start:start + max_delay + 1])  # where a detection counts
        before_start = np.sort(np.concatenate(before_start))
        thresholds = np.unique(np.concatenate([[-np.inf], before_start, *windows]))[::-1]
        num_false_alarms = len(before_start) - np.searchsorted(before_start, thresholds, "right")
        false_alarms = num_false_alarms / len(before_start) * 10 * num_points_per_year
        years = np.zeros(len(thresholds))
        for window in windows:  # first step whose running max exceeds the threshold
            delay = np.searchsorted(np.maximum.accumulate(window), thresholds, "right")
            years += np.where(
                delay < len(window), delay / num_points_per_year, max_time_to_detection
            )
        curves.append((false_alarms, years / len(windows)))
    vus = area_under_curve(mags, np.array([area_under_curve(x, z) for x, z in curves]))
    if plot:
        plot_surface(
            curves,
            mags,
            "false alarms per 10 years",
            "anomaly magnitude",
            "years to detection",
            f"VUS-AMOC = {vus:.3f}",
        )
    return vus
