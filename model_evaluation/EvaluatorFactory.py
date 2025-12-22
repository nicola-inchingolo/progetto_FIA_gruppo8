import numpy as np
import pandas as pd
from holdout_evaluator import holdout_evaluator
from k_fold_evaluator import kFoldEvaluator
from leave_one_out_evaluator import LeaveOneOutEvaluator


class EvaluatorFactory:

    # da rivedere
    @staticmethod
    def generate_evaluator(ev_type: str, dataset: pd.DataFrame, metrics: np.array, **kwargs):
        if ev_type == "holdout":
            return holdout_evaluator(dataset, metrics, kwargs.get("train_percentage"))
        elif ev_type == "k-fold":
            return kFoldEvaluator(dataset, metrics, kwargs.get("K_tests"))
        elif ev_type == "loo":
            return LeaveOneOutEvaluator(dataset, metrics)
        else:
            raise ValueError(f"Unknown evaluator type: {ev_type}")
