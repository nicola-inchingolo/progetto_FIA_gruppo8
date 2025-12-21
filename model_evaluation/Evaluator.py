from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# CAPISCI SE NECESSARIO DEFINIRE CALCULATE_METRICS E AUC QUI ESSENDO UGUALI PER TUTTI E TRE I TIPI

# da settare come abstract mediante ABC
class Evaluator(ABC):
    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray):

        if not isinstance(datasetToEvaluate, pd.DataFrame):
            raise TypeError("datasetToEvaluate must be a pandas DataFrame")
        if datasetToEvaluate.empty:
            raise ValueError("datasetToEvaluate cannot be empty")
        if not isinstance(metrics, (list, np.ndarray)):
            raise TypeError("metrics must be a list or numpy array")
        if len(metrics) == 0:
            raise ValueError("metrics list cannot be empty")

        self.dataset = datasetToEvaluate
        self.metrics = metrics

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):
        pass

    @abstractmethod
    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series, y_score: pd.Series):
        pass

    @abstractmethod
    def calculate_auc(self, y_test: pd.Series, y_score: pd.Series):
        pass


