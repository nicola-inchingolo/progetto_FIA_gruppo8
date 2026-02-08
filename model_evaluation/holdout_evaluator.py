import pandas as pd
import numpy as np
from Evaluator import evaluator
import plot_data
from model_development import Knn_Algorithm

"""
The holdout evaluator class implements the holdout evaluation strategy.

It evaluates a model by splitting the dataset into training and test sets
according to a specified training percentage. This class is a concrete
implementation of the abstract evaluator class.
"""


class holdout_evaluator(evaluator):
    """
    Initializes the holdout evaluator and sets the training percentage.

    @:param datasetToEvaluate: Dataset used for evaluation.
    @:param metrics: Array of metric identifiers to compute.
    @:param train_percentage: Percentage of data used for training.

    @:raise TypeError: If train_percentage is not a float.
    @:raise ValueError: If train_percentage is not between 0 and 1.
    """

    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.ndarray, train_percentage: float, p: int,
                 seed: int = 42):
        # call the constructor of the superclass evaluator
        super().__init__(datasetToEvaluate, metrics, p)

        # these exceptions verify if the train_percentage will be correctly istantiated
        if not isinstance(train_percentage, float):
            raise TypeError("train_percentage must be a float")

        if not 0 < train_percentage < 1:
            raise ValueError("train_percentage must be between 0 and 1")

        self.train_percentage = train_percentage
        self.seed = seed

    """
    Splits the dataset into training and test sets using the holdout strategy.

    @:return: Tuple containing training features, test features,
             training labels and test labels.

    @:raise TypeError: If the dataset is not a pandas DataFrame.
    @:raise KeyError: If required columns are missing from the dataset.
    """

    def split_dataset_with_strategy(self):

        # these exceptions verify if the dataset is instantiated
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Dataset must be a pandas DataFrame")

        required_cols = {"ID", "Sample code number", "Class"}
        if not required_cols.issubset(self.dataset.columns):
            raise KeyError(f"Dataset must contain columns {required_cols}")

        # shuffle del dataset
        shuffled_dataset = self.dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        split_index = int(len(shuffled_dataset) * self.train_percentage)
        dtr = shuffled_dataset.iloc[:split_index]
        dtst = shuffled_dataset.iloc[split_index:]

        x_train = dtr.drop(columns=required_cols)
        y_train = dtr["Class"]

        x_test = dtst.drop(columns=required_cols)
        y_test = dtst["Class"]

        return x_train, x_test, y_train, y_test

    """
    Executes the holdout evaluation procedure.

    Trains the model, computes predictions, evaluates metrics,
    calculates AUC if required, and produces plots and output files.

    @:raise RuntimeError: If the holdout evaluation process fails.
    """

    def evaluate(self):
        try:
            x_train, x_test, y_train, y_test = self.split_dataset_with_strategy()
            # calculate prediction
            model = Knn_Algorithm.KNNClassifier(k=5, p=self.distance_strategy)
            model.fit(x_train, y_train)
            y_score, y_pred = model.predict(x_test, pos_label=4)

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
