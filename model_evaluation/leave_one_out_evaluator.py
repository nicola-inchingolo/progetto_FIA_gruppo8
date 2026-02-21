import numpy as np
import pandas as pd

from model_evaluation.metrics import confusion_matrix_binary
from model_evaluation.metrics import calculate_mean_metrics
from model_evaluation import plot_data
from model_evaluation.evaluator import Evaluator
from model_development import Knn_Algorithm
from data_preprocessing.op6_outlier_remover import run_remove_outlier

"""
The leave-one-out evaluator class implements the leave-one-out
cross-validation strategy.

Each sample is used once as a test set while the remaining samples
are used for training.
"""


class LeaveOneOutEvaluator(Evaluator):
    """
    Initializes the leave-one-out evaluator.

    @:param datasetToEvaluate: Dataset used for evaluation.
    @:param metrics: Array of metric identifiers to compute.
    """

    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.array, p: int, k: int):
        super().__init__(datasetToEvaluate, metrics, p, k)

    """
    Splits the dataset into training and test sets according to the
    leave-one-out strategy.

    @:param size: Index of the sample used as test set.

    @:return: Tuple containing training features, test features,
             training labels and test labels.

    @:raise ValueError: If the index is out of range.
    @:raise KeyError: If required columns are missing from the dataset.
    """

    def split_dataset_with_strategy(self, size: int):

        if size < 0 or size >= len(self.dataset):
            raise ValueError("size index out of range")

        target_col = "classtype_v1"

        if target_col not in self.dataset.columns:
            raise KeyError(f"Dataset must contain column {target_col}")

        test_section = self.dataset.iloc[size:size + 1]
        train_section = pd.concat([
            self.dataset.iloc[:size],
            self.dataset.iloc[size + 1:]
        ])

        print(f"test set:\n{test_section}\n")
        print(f"train set:\n{train_section}\n")

        # Remove outliers from train dataset
        train_section = run_remove_outlier(train_section)
        train_section = train_section.df_output

        x_train = train_section.drop(columns=[target_col])
        y_train = train_section[target_col]

        x_test = test_section.drop(columns=[target_col])
        y_test = test_section[target_col]

        return x_train, x_test, y_train, y_test

    """
    Executes the leave-one-out evaluation procedure.

    Trains and evaluates the model for each sample, computes the mean
    of the selected metrics, calculates AUC if required, and produces
    plots and output files.

    @:raise RuntimeError: If the leave-one-out evaluation process fails.
    """

    def evaluate(self):
        try:
            rows, _ = self.dataset.shape

            size = 0
            metrics_list = dict()

            y_test_all = np.array([])
            y_score_all = np.array([])
            y_pred_all = np.array([])

            for i in range(rows):
                x_train, x_test, y_train, y_test = self.split_dataset_with_strategy(size)

                model = Knn_Algorithm.KNNClassifier(k=self.k_neighbours, p=self.distance_strategy)
                model.fit(x_train, y_train)
                y_score, y_pred = model.predict(x_test, pos_label=4)

                y_test_all = np.append(y_test_all, y_test)
                y_score_all = np.append(y_score_all, y_score)
                y_pred_all = np.append(y_pred_all, y_pred)

                metrics_list[i], _ = self.calculate_metrics(y_test, y_pred)
                size += 1

            mean_metrics = calculate_mean_metrics(metrics_list)

            if 6 in self.metrics or 7 in self.metrics:
                auc, tpr_array, fpr_array = self.calculate_auc(y_test_all, y_score_all)
                plot_data.plot_ROC(fpr_array, tpr_array)
                mean_metrics["AUC"] = auc

            print("\nMean metrics:")
            print(mean_metrics)

            cm_all = confusion_matrix_binary(y_test_all, y_pred_all)
            plot_data.plot_confusion_matrix(cm_all)
            plot_data.save_output_result(mean_metrics)
        except Exception as e:
            raise RuntimeError("Leave-one-out evaluation failed") from e
