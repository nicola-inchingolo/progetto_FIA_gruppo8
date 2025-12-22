import random
import pandas as pd
import numpy as np
from MockKNN import MockKNN
from metrics import metrics
from metrics import confusion_matrix_binary
import matplotlib.pyplot as plt
from Evaluator import evaluator
import plot_data


# classe generale evaluator deve avere come parametri il numero di esperimenti K, le metriche da calcolare
# il dataset cleaned la percentuale di divisione nel caso dell'holdout


class holdout_evaluator(evaluator):
    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray, train_percentage: float):
        super().__init__(datasetToEvaluate, metrics)
        # Exception
        if not isinstance(train_percentage, float):
            raise TypeError("train_percentage must be a float")

        if not 0 < train_percentage < 1:
            raise ValueError("train_percentage must be between 0 and 1")

        self.train_percentage = train_percentage

    def split_dataset_with_strategy(self):
        # splitting
        # train_percentage = random.uniform(0.6, 0.9)

        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Dataset must be a pandas DataFrame")

        # da cambiare con le reali
        required_cols = {"ID", "Sample code number", "Class"}
        if not required_cols.issubset(self.dataset.columns):
            raise KeyError(f"Dataset must contain columns {required_cols}")

        split_index = int(len(self.dataset) * self.train_percentage)
        dtr = self.dataset.iloc[:split_index]
        dtst = self.dataset.iloc[split_index:]

        x_train = dtr.drop(columns=required_cols)
        y_train = dtr["Class"]

        x_test = dtst.drop(columns=required_cols)
        y_test = dtst["Class"]

        return x_train, x_test, y_train, y_test

    def evaluate(self):
        try:
            x_train, x_test, y_train, y_test = self.split_dataset_with_strategy()

            # calculate prediction
            model = MockKNN(k=5, seed=42, error_rate=0.3)
            model.fit(x_train, y_train)
            y_pred, y_score = model.predict(x_test, y_test=y_test)

            print(y_pred)

            metrics_list, cm = self.calculate_metrics(y_test, y_pred)

            if 6 in self.metrics or 7 in self.metrics:
                auc, tpr_array, fpr_array = self.calculate_auc(y_test, y_score)
                metrics_list["AUC"] = auc
                plot_data.plot_ROC(fpr_array, tpr_array)

            plot_data.plot_confusion_matrix(cm)
            plot_data.save_output_result(metrics_list)
        except Exception as e:
            raise RuntimeError("Holdout Evaluation failed, something goes wrong") from e
