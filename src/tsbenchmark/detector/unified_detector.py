from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class UnifiedDetector(ABC):
    """
    Unified interface for all detectors.

    Every detector is called the same way:

        detector.detect(data, initial_len, threshold_param, params)

    - data: pd.Series, values are the time series y, index are its timestamps
    - initial_len: int, number of first datapoints used as training / warm-up;
      no alarm is raised there
    - threshold_param: the parameter acting as a threshold; changing it changes
      the detection time and the number of false alarms
    - params: dict, all other hyperparameters the detector requires

    New detectors subclass UnifiedDetector, set `name` and `default_params`,
    and implement `_detect`.
    """

    name: str = None
    default_params: dict = {}

    def detect(
        self,
        data: pd.Series,
        initial_len: int,
        threshold_param,
        params: Optional[dict] = None,
    ) -> pd.Series:
        """
        Run the detector.

        Returns:
            pd.Series of alarms (1 = alarm, 0 = no alarm), indexed by the
            timestamps of `data`.
        """
        y, timestamps = self._check_data(data)
        if not 0 <= initial_len <= len(y):
            raise ValueError(
                f"initial_len must be in [0, {len(y)}], got {initial_len}"
            )

        unknown = set(params or {}) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"Unknown params for {self.name}: {sorted(unknown)}. "
                f"Accepted: {sorted(self.default_params)}"
            )
        params = {**self.default_params, **(params or {})}

        alarms = np.asarray(
            self._detect(y, timestamps, initial_len, threshold_param, params),
            dtype=int,
        )
        if alarms.shape != y.shape:
            raise ValueError(
                f"{self.name}._detect returned shape {alarms.shape}, "
                f"expected {y.shape}"
            )
        alarms[:initial_len] = 0  # no detection during warm-up

        return pd.Series(alarms, index=timestamps, name="alarm")

    @abstractmethod
    def _detect(
        self,
        y: np.ndarray,
        timestamps: pd.Index,
        initial_len: int,
        threshold_param,
        params: dict,
    ) -> np.ndarray:
        """
        Detector-specific logic.

        Returns:
            np.ndarray of 0/1 alarms with the same length as y.
        """

    @staticmethod
    def _check_data(data: pd.Series):
        if not isinstance(data, pd.Series):
            raise TypeError(
                "data must be a pd.Series with y as values and timestamps as "
                f"index, got {type(data).__name__}"
            )
        return data.to_numpy(dtype=float), data.index
