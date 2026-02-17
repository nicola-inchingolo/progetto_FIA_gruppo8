from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from model_evaluation.metrics import confusion_matrix_binary, metrics


"""
Abstract base class for evaluators.

Defines the common interface and shared functionality
for all evaluation strategies.
"""
class evaluator(ABC):

    """
    Initializes the evaluator with a dataset and a list of metrics.

    @:param datasetToEvaluate: Dataset used for model evaluation.
    @:param metrics: List or array of metric identifiers to compute.
    @:param p: distance_coefficients that indicates the strategy

    @:raise TypeError: If datasetToEvaluate is not a pandas DataFrame.
    @:raise ValueError: If datasetToEvaluate is empty.
    @:raise TypeError: If metrics is not a list or numpy array.
    @:raise ValueError: If metrics is empty.
    """
    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray, p: int, k: int):

        if not isinstance(datasetToEvaluate, pd.DataFrame):
            raise TypeError("datasetToEvaluate must be a pandas DataFrame")
        if datasetToEvaluate.empty:
            raise ValueError("datasetToEvaluate cannot be empty")
        if not isinstance(metrics, (list, np.ndarray)):
            raise TypeError("metrics must be a list or numpy array")
        if len(metrics) == 0:
            raise ValueError("metrics list cannot be empty")
        if not isinstance(p, int):
            raise TypeError("parameter p must be an integer")

        self.dataset = datasetToEvaluate
        self.metrics = metrics
        self.distance_strategy = p
        self.k_neighbours = k

    """
    Abstract method that must be implemented by subclasses.

    Executes the evaluation procedure according to the
    selected evaluation strategy.
    """
    @abstractmethod
    def evaluate(self):
        pass

    """
    Computes evaluation metrics and confusion matrix.

    @:param y_test: True class labels.
    @:param y_pred: Predicted class labels.

    @:return: Tuple containing:
        - dictionary of computed metrics
        - confusion matrix (2x2)

    @:raise ValueError: If input vectors are empty.
    @:raise ValueError: If input vectors have different lengths.
    @:raise ValueError: If confusion matrix is not 2x2.
    @:raise ValueError: If invalid metric identifiers are provided.
    """
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

    """
    Computes the Area Under the ROC Curve (AUC).

    @:param y_test: True class labels.
    @:param y_score: Predicted scores or probabilities.

    @:return: Tuple containing:
            - AUC value
            - True Positive Rate (TPR) array
            - False Positive Rate (FPR) array

    @:raise ValueError: If input vectors are empty.
    @:raise ValueError: If y_score values are not in range [0, 1].
    """
    def calculate_auc(self, y_test: pd.Series, y_score: pd.Series):

        if len(y_test) == 0:
            raise ValueError("y_test is empty")
        if len(y_score) == 0:
            raise ValueError("y_score is empty")

        if np.any((y_score < 0) | (y_score > 1)):
            raise ValueError("y_score values must be in range [0, 1]")

        # ROC computation using multiple thresholds
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


