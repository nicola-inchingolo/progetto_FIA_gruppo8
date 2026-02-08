import numpy as np
import pandas as pd
from holdout_evaluator import holdout_evaluator
from k_fold_evaluator import kFoldEvaluator
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
    def generate_evaluator(ev_type: str, dataset: pd.DataFrame, metrics: np.array, p: int, **kwargs):
        if ev_type == "holdout":
            return holdout_evaluator(dataset, metrics, p, kwargs.get("train_percentage"))
        elif ev_type == "k-fold":
            return kFoldEvaluator(dataset, metrics, p, kwargs.get("K_tests"))
        elif ev_type == "loo":
            return LeaveOneOutEvaluator(dataset, metrics, p)
        else:
            raise ValueError(f"Unknown evaluator type: {ev_type}")
