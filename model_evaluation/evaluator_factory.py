import numpy as np
import pandas as pd
from holdout_evaluator import HoldoutEvaluator
from k_fold_evaluator import Kfoldevaluator
from leave_one_out_evaluator import LeaveOneOutEvaluator

"""
Factory class used to instantiate evaluators
according to the selected evaluation strategy.
"""


class EvaluatorFactory:
    """
    Generates and returns an evaluator based on the specified type.
    @:param ev_type: Type of evaluator ("holdout", "k-fold", "loo").
    @:param dataset: Dataset used for evaluation.
    @:param metrics: Array of metrics to compute.
    @:param kwargs: Additional parameters required by specific evaluators (ex. train_percentage for holdout, K_tests for k-fold).
    @:return: Instance of the selected evaluator.

    @:raise ValueError: If the evaluator type is not supported.
    """

    @staticmethod
    def generate_evaluator(ev_type: str, dataset: pd.DataFrame, metrics: np.array, p: int, k: int, **kwargs):
        if ev_type == "holdout":
            return HoldoutEvaluator(dataset, metrics, p, k, kwargs.get("train_percentage"))
        elif ev_type == "k-fold":
            return Kfoldevaluator(dataset, metrics, p, k, kwargs.get("K_tests"))
        elif ev_type == "loo":
            return LeaveOneOutEvaluator(dataset, metrics, p, k)
        else:
            raise ValueError(f"Unknown evaluator type: {ev_type}")
