import random
import pandas as pd
import numpy as np
from MockKNN import MockKNN
from metrics import metrics
from metrics import confusion_matrix_binary
import matplotlib.pyplot as plt
import plot_data
from abc import ABC, abstractmethod


# classe generale evaluator deve avere come parametri il numero di esperimenti K, le metriche da calcolare
# il dataset cleaned la percentuale di divisione nel caso dell'holdout

# da settare come abstract mediante ABC
class evaluator(ABC):
    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray):
        self.dataset = datasetToEvaluate
        self.metrics = metrics

    @abstractmethod
    def evaluate(self):
        pass


class holdout_evaluator(evaluator):
    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray, train_percentage: float):
        super().__init__(datasetToEvaluate, metrics)
        self.train_percentage = train_percentage

    def split_dataset_with_strategy(self):
        # splitting
        # train_percentage = random.uniform(0.6, 0.9)

        split_index = int(len(self.dataset) * self.train_percentage)
        dtr = self.dataset.iloc[:split_index]
        dtst = self.dataset.iloc[split_index:]

        x_train = dtr.drop(columns=["ID", "Sample code number", "Class"])
        y_train = dtr["Class"]

        x_test = dtst.drop(columns=["ID", "Sample code number", "Class"])
        y_test = dtst["Class"]

        return x_train, x_test, y_train, y_test

    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series, y_score: pd.Series):
        # making confusion matrix
        cm = confusion_matrix_binary(y_test, y_pred)
        print("Confusion Matrix:")

        print(f"TN: {cm[0][0]}")
        print(f"FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}")
        print(f"TP: {cm[1][1]}")

        metrics_calculator = metrics(cm)
        metrics_list = dict()

        if 7 in self.metrics:
            metrics_list = metrics_calculator.calculate_all_the_above()
            metrics_list["AUC"] = self.calculate_auc(y_test, y_score)

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
            if 6 in self.metrics:
                metrics_list["AUC"] = self.calculate_auc(y_test, y_score)

        return metrics_list, cm

    def calculate_auc(self, y_test: pd.Series, y_score: pd.Series):
        # Calcolo ROC per più soglie
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

        sorted_idx = np.argsort(fpr_list)
        fpr_list = np.array(fpr_list)[sorted_idx]
        tpr_list = np.array(tpr_list)[sorted_idx]
        auc = np.trapezoid(tpr_list, fpr_list)
        plot_data.plot_ROC(fpr_list, tpr_list)

        return auc

    def evaluate(self):

        x_train, x_test, y_train, y_test = self.split_dataset_with_strategy()

        # calculate prediction
        model = MockKNN(k=5, seed=42, error_rate=0.3)
        model.fit(x_train, y_train)
        y_pred, y_score = model.predict(x_test, y_test=y_test)

        print(y_pred)

        metrics_list, cm = self.calculate_metrics(y_test, y_pred, y_score)

        plot_data.plot_confusion_matrix(cm)
        plot_data.save_output_result(metrics_list)