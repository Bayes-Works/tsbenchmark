from abc import ABC, abstractmethod
from src.tsbenchmark.utils import apply_recursive

class BaseDetector(ABC):
    """
    Base detector for anomaly detection
    """

    def __init__(self):
        self._detector_name = None
        self._anom_threshold = None

        self.init_detector()

    @property
    def name(self):
        """str: Name of detector"""
        return self._detector_name
    
    @property
    def anom_threshold(self):
        """float: Anomaly threshold"""
        return self._anom_threshold
    
    @abstractmethod
    def init_detector(self):
        """
        Initialize detector
        """

    @abstractmethod
    def data_process(self, data, **kwargs):
        """
        Data processing for the detector
        """
    
    @abstractmethod
    def anomaly_score(self, data, **kwargs):
        """
        Obtain anomaly score
        """

    def get_anomaly_score(self, data, **kwargs):
        """
        Obtain anomaly score for all time series
        """
        return apply_recursive(data, fn=self.anomaly_score, **kwargs)
