from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from metrics import confusion_matrix_binary, metrics


# da settare come abstract mediante ABC
class evaluator(ABC):
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

    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):

        if len(y_test) == 0:
            raise ValueError("y_test is empty")
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty")

        if len(y_test) != len(y_pred):
            raise ValueError("y_test and y_pred must have the same length")

        # making confusion matrix
        cm = confusion_matrix_binary(y_test, y_pred)

        if cm.shape != (2, 2):
            raise ValueError("Confusion matrix must be 2x2")

        print("Confusion Matrix:")

        print(f"TN: {cm[0][0]}")
        print(f"FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}")
        print(f"TP: {cm[1][1]}")

        metrics_calculator = metrics(cm)
        metrics_list = dict()

        valid_metrics = set(range(1, 8))
        if not set(self.metrics).issubset(valid_metrics):
            raise ValueError("Invalid metric index (valid values are 1–7)")

        if 7 in self.metrics:
            metrics_list = metrics_calculator.calculate_all_the_above()
        else:
            if 1 in self.metrics:
                metrics_list["Accuracy"] = metrics_calculator.calculate_accuracy()
            if 2 in self.metrics:
                metrics_list["Error Rate"] = metrics_calculator.calculate_error_rate()
            if 3 in self.metrics:
                metrics_list["Sensitivity"] = metrics_calculator.calculate_sensitivity()
            if 4 in self.metrics:
                metrics_list["Specificity"] = metrics_calculator.calculate_specificity()
            if 5 in self.metrics:
                metrics_list["Geometric Mean"] = metrics_calculator.calculate_geometric_mean()

        return metrics_list, cm

    def calculate_auc(self, y_test: pd.Series, y_score: pd.Series):

        if len(y_test) == 0:
            raise ValueError("y_test is empty")
        if len(y_score) == 0:
            raise ValueError("y_score is empty")

        if np.any((y_score < 0) | (y_score > 1)):
            raise ValueError("y_score values must be in range [0, 1]")

        # Calcolo ROC per più soglie
        # valore soglia potrebbe diventare variabile
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            TP = np.sum((y_score >= thresh) & (y_test == 4))
            FP = np.sum((y_score >= thresh) & (y_test == 2))
            FN = np.sum((y_score < thresh) & (y_test == 4))
            TN = np.sum((y_score < thresh) & (y_test == 2))

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_idx = np.argsort(fpr_array)
        fpr_array = fpr_array[sorted_idx]
        tpr_array = tpr_array[sorted_idx]
        auc = np.trapezoid(tpr_array, fpr_array)
        return auc, tpr_array, fpr_array


