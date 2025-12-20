from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from MockKNN import MockKNN
from metrics import metrics
import plot_data
from metrics import confusion_matrix_binary
from metrics import calculate_mean_metrics
from holdout_evaluator import evaluator


# K esperimenti da inserire
class kFoldEvaluator(evaluator):

    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.array, K_tests: int):
        super().__init__(datasetToEvaluate, metrics)
        self.K_tests = K_tests

    def split_dataset_with_strategy(self, folds: np.array, i: int, K_split: int):
        test_section = folds[i]
        train_section = pd.concat(folds[j] for j in range(K_split) if j != i)

        x_train = train_section.drop(columns=["ID", "Sample code number", "Class"])
        y_train = train_section["Class"]
        x_test = test_section.drop(columns=["ID", "Sample code number", "Class"])
        y_test = test_section["Class"]

        return x_train, x_test, y_train, y_test

    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):
        cm = confusion_matrix_binary(y_test, y_pred)
        print("Confusion Matrix:")
        print(type(cm))

        print(f"TN: {cm[0][0]}")
        print(f"FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}")
        print(f"TP: {cm[1][1]}")

        metrics_calculator = metrics(cm)
        metrics_list = dict()

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

        return metrics_list

    def calculate_auc(self, y_test: pd.Series, y_score: pd.Series):
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

            # Ordinamento per FPR crescente
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_idx = np.argsort(fpr_array)
        fpr_array = fpr_array[sorted_idx]
        tpr_array = tpr_array[sorted_idx]
        auc = np.trapezoid(tpr_array, fpr_array)
        return auc, tpr_array, fpr_array

    def evaluate(self):
        K_split = int(input("inserisci K: "))

        folds = np.array_split(self.dataset, K_split)

        metrics_list = dict()
        y_test_all = np.array([])
        y_score_all = np.array([])
        y_pred_all = np.array([])

        for i in range(K_split):
            x_train, x_test, y_train, y_test = self.split_dataset_with_strategy(folds, i, K_split)

            model = MockKNN(k=5, seed=i, error_rate=0.3)
            model.fit(x_train, y_train)
            y_pred, y_score = model.predict(x_test, y_test=y_test)

            y_test_all = np.append(y_test_all, y_test)
            y_score_all = np.append(y_score_all, y_score)
            y_pred_all = np.append(y_pred_all, y_pred)

            metrics_list[i] = self.calculate_metrics(y_test, y_pred)

        mean_metrics = calculate_mean_metrics(metrics_list)
        if 6 in self.metrics or 7 in self.metrics:
            auc, tpr_array, fpr_array = self.calculate_auc(y_test_all, y_score_all)
            mean_metrics["AUC"] = auc
            plot_data.plot_ROC(fpr_array, tpr_array)

        print("\nMedie delle metriche su tutti i fold:")
        print(mean_metrics)

        cm_all = confusion_matrix_binary(y_test_all, y_pred_all)
        plot_data.plot_confusion_matrix(cm_all)
        plot_data.save_output_result(mean_metrics, metrics_list)