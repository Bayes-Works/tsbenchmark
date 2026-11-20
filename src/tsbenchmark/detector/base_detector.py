"""Base classes of the anomaly detectors.

Every detector is used the same way: fit(y_train) on the anomaly-free start of a series, then
score(y) gives one anomaly score per time step and detect(y) the 0/1 labels. Three kinds:
    - OfflineDetector: the score at t may use the whole series
    - OnlineDetector: the score at t uses y[0..t] only, one point at a time
    - OfflineToOnlineDetector: an OfflineDetector refitted on y[0..t], with a rule turning its
      successive results into the score at t
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from tsbenchmark.utils import get_num_points_per_year, plot_offline, plot_online


class BaseDetector(ABC):
    """Fit on the training part, then score and detect any series.

    Subclasses set the class attributes, implement score and, if the method learns, _fit.
    """

    name = None
    mode = None  # "online" or "offline"
    learning = None  # "unsupervised", "semi-supervised" or "supervised"
    output = "score"  # "score" (continuous) or "alarm" (0/1, threshold built in)

    def __init__(self, data_sampling_freq=None, threshold=None, **params):
        self.params = params  # hyperparameters, logged with the results
        self.threshold = threshold  # alarm where score > threshold; None = tuned in fit
        self.num_points_per_year = (
            None if data_sampling_freq is None else get_num_points_per_year(data_sampling_freq)
        )
        self.y_train = None  # training part, kept by fit to be drawn in front of y in plots
        self.scores = None  # scores of the last y given to detect, kept for the metrics

    def fit(self, y_train, labels_train=None):
        """Learn from the training part, then tune the threshold if none was given."""
        y_train = np.asarray(y_train, dtype=float)
        self.y_train = y_train
        self._fit(y_train, labels_train)
        if self.output == "score" and self.threshold is None:
            self.threshold = self.tune_threshold(y_train)
        return self

    def _fit(self, y_train, labels_train=None):
        """The method's own learning (labels_train for supervised detectors only); default: none."""

    def tune_threshold(self, y_train):
        """The method's own way to set the threshold from the training part, e.g.
        1.1 * np.nanmax(self.score(y_train)); each method that tunes its threshold overrides it."""
        raise ValueError(f"{self.name}: give a threshold or override tune_threshold")

    @abstractmethod
    def score(self, y):
        """One anomaly score per time step of y, higher = more anomalous, NaN allowed."""

    def detect(self, y, plot=False, anm_labels=None):
        """0/1 per time step: score > threshold, or the 0/1 scores of an "alarm" detector.
        NaN scores give 0.

        plot: also draw the training part from fit (unscored) then y, with the scores and the
        alarms, step by step for an online detector. anm_labels: true labels of the training part
        then y, drawn to compare with the alarms (only used in the plot).
        """
        if self.output == "score" and self.threshold is None:
            raise ValueError(f"{self.name} has no threshold: give one or call fit first")
        scores = np.asarray(self.score(y), dtype=float)
        self.scores = scores
        alarm = scores if self.output == "alarm" else scores > self.threshold
        alarm = np.nan_to_num(alarm).astype(int)
        if plot:
            plot_detection = plot_online if self.mode == "online" else plot_offline
            threshold = self.threshold if self.output == "score" else None
            plot_detection(
                y, scores, alarm, threshold, anm_labels, title=self.name, y_train=self.y_train
            )
        return alarm


class OfflineDetector(BaseDetector):
    """The score at t may use the whole series, including the points after t."""

    mode = "offline"

    def analyze(self, y):
        """Everything the method estimates on y, e.g. {"scores": ..., "changepoints":
        {position: magnitude}, "forecast": ...}; default: the scores only."""
        return {"scores": self.score(y)}


class OnlineDetector(BaseDetector):
    """The score at t uses y[0..t] only: each point is scored, then used to update the state.

    _fit must leave the detector ready to score a series from its first point.
    """

    mode = "online"

    @abstractmethod
    def score_one(self, y_t):
        """Score of the new point y_t, from the state built on the points before it."""

    @abstractmethod
    def update(self, y_t):
        """Take y_t into the state after scoring it: filter step or parameter update."""

    def score(self, y):
        """Score every point in turn from a copy of the fitted state, so the detector itself is
        left unchanged. May be replaced by a one-pass version that gives the same scores."""
        detector = deepcopy(self)
        scores = np.empty(len(y))
        for t, y_t in enumerate(np.asarray(y, dtype=float)):
            scores[t] = detector.score_one(y_t)
            detector.update(y_t)
        return scores

    def analyze(self, y):
        """Everything the method estimates on y, e.g. {"scores": ..., "changepoints":
        {position: magnitude}, "forecast": ...}; default: the scores only."""
        return {"scores": self.score(y)}


class OfflineToOnlineDetector(BaseDetector):
    """An OfflineDetector run online: refitted on y[0..t] (or its last `window` points) every
    refit_every steps, with rule turning its successive analyze() results into the score at t.
    The training part and the steps between refits get NaN.
    """

    mode = "online"

    class Rule(ABC):
        """How successive analyze() results become the online score at t. Has memory, e.g. the
        changepoints of the previous refit, to tell what is new."""

        name = None
        output = "score"  # "score" or "alarm", passed on to the detector

        @abstractmethod
        def reset(self):
            """Clear the memory."""

        @abstractmethod
        def step(self, result, t):
            """Score at t from analyze() on y up to t; positions in result count from
            y[result["start"]]."""

    def __init__(self, offline, rule, refit_every=1, window=None, threshold=None):
        super().__init__(threshold=threshold, refit_every=refit_every, window=window)
        self.offline, self.rule = offline, rule
        self.refit_every, self.window = refit_every, window
        self.name = f"{offline.name}-online[{rule.name}]"
        self.learning, self.output = offline.learning, rule.output
        self.num_points_per_year = offline.num_points_per_year

    def _fit(self, y_train, labels_train=None):
        self.train_len = len(y_train)
        self.offline._fit(y_train, labels_train)  # learning only: the offline threshold isn't used

    def score(self, y):
        y = np.asarray(y, dtype=float)
        scores = np.full(len(y), np.nan)
        self.rule.reset()
        # reference: what is already in the training part isn't new
        self.rule.step(self._analyze_until(y, self.train_len - 1), self.train_len - 1)
        for t in range(self.train_len, len(y), self.refit_every):
            scores[t] = self.rule.step(self._analyze_until(y, t), t)
        return scores

    def _analyze_until(self, y, t):
        """offline.analyze on y[0..t], or on its last `window` points."""
        start = 0 if self.window is None else max(0, t + 1 - self.window)
        return {**self.offline.analyze(y[start:t + 1]), "start": start}
