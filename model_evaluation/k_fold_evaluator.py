from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from MockKNN import MockKNN
from metrics import metrics
import plot_data
from metrics import confusion_matrix_binary
from metrics import calculate_mean_metrics
from Evaluator import evaluator


# K esperimenti da inserire
class kFoldEvaluator(evaluator):

    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.array, K_tests: int):
        super().__init__(datasetToEvaluate, metrics)

        if not isinstance(K_tests, int):
            raise TypeError("K_tests must be an integer")
        if K_tests < 2 or K_tests > len(datasetToEvaluate):
            raise ValueError("K_tests must be between 2 and the number of samples")
        self.K_tests = K_tests

    def split_dataset_with_strategy(self, folds: np.array, i: int):

        required_cols = {"ID", "Sample code number", "Class"}
        if not required_cols.issubset(self.dataset.columns):
            raise KeyError(f"Dataset must contain columns {required_cols}")

        test_section = folds[i]
        train_section = pd.concat(folds[j] for j in range(self.K_tests) if j != i)

        if test_section.empty or train_section.empty:
            raise ValueError("Train or test fold is empty")

        x_train = train_section.drop(columns=required_cols)
        y_train = train_section["Class"]
        x_test = test_section.drop(columns=required_cols)
        y_test = test_section["Class"]

        return x_train, x_test, y_train, y_test

    def evaluate(self):
        try:
            folds = np.array_split(self.dataset, self.K_tests)

            metrics_list = dict()
            y_test_all = np.array([])
            y_score_all = np.array([])
            y_pred_all = np.array([])

            for i in range(self.K_tests):
                x_train, x_test, y_train, y_test = self.split_dataset_with_strategy(folds, i)

                model = MockKNN(k=5, seed=i, error_rate=0.3)
                model.fit(x_train, y_train)
                y_pred, y_score = model.predict(x_test, y_test=y_test)

                y_test_all = np.append(y_test_all, y_test)
                y_score_all = np.append(y_score_all, y_score)
                y_pred_all = np.append(y_pred_all, y_pred)

                metrics_list[i], _ = self.calculate_metrics(y_test, y_pred)

            print(type(metrics_list))
            mean_metrics = calculate_mean_metrics(metrics_list)
            if 6 in self.metrics or 7 in self.metrics:
                auc, tpr_array, fpr_array = self.calculate_auc(y_test_all, y_score_all)
                mean_metrics["AUC"] = auc
                plot_data.plot_ROC(fpr_array, tpr_array)

            print("\nMedie delle metriche su tutti i fold:")
            print(mean_metrics)

            cm_all = confusion_matrix_binary(y_test_all, y_pred_all)
            plot_data.plot_confusion_matrix(cm_all)
            plot_data.save_output_result(mean_metrics)

        except Exception as e:
            raise RuntimeError("K-fold Evaluation failed, something goes wrong") from e