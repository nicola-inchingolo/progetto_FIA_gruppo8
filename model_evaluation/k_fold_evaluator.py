import numpy as np
import pandas as pd

import plot_data
from metrics import confusion_matrix_binary
from metrics import calculate_mean_metrics
from Evaluator import evaluator
from model_development import Knn_Algorithm

"""
The k-fold evaluator class implements the k-fold cross-validation strategy.

The dataset is split into K folds, and the evaluation is performed K times,
each time using a different fold as test set and the remaining folds as
training set.
"""


class kFoldEvaluator(evaluator):
    """
    Initializes the k-fold evaluator.

    @:param datasetToEvaluate: Dataset used for evaluation.
    @:param metrics: Array of metric identifiers to compute.
    @:param K_tests: Number of folds to use in k-fold cross-validation.

    @:raise TypeError: If K_tests is not an integer.
    @:raise ValueError: If K_tests is not between 2 and the number of samples.
    """

    def __init__(self, datasetToEvaluate: pd.DataFrame, metrics: np.array, p: int, k: int, K_tests: int,  seed: int = 42):
        super().__init__(datasetToEvaluate, metrics, p, k)

        if not isinstance(K_tests, int):
            raise TypeError("K_tests must be an integer")
        if K_tests < 2 or K_tests > len(datasetToEvaluate):
            raise ValueError("K_tests must be between 2 and the number of samples")
        self.K_tests = K_tests
        self.seed = seed

    """
    Splits the dataset into training and test sets according to the k-fold strategy.

    @:param folds: Array containing the dataset folds.
    @:param i: Index of the fold used as test set.

    @:return: Tuple containing training features, test features,
    training labels and test labels.

    @:raise KeyError: If required columns are missing from the dataset.
    @:raise ValueError: If the training or test fold is empty.
    """

    def split_dataset_with_strategy(self, folds: np.array, i: int):

        target_col = "classtype_v1"

        if target_col not in self.dataset.columns:
            raise KeyError(f"Dataset must contain column {target_col}")

        test_section = folds[i]
        #train_section = pd.concat(
         #   folds[j] for j in range(self.K_tests) if j != i
        #)
        train_section = pd.concat(
            [pd.DataFrame(f) if not isinstance(f, pd.DataFrame) else f for f in folds]
        )

        if test_section.empty or train_section.empty:
            raise ValueError("Train or test fold is empty")

        # X = tutto tranne la label
        x_train = train_section.drop(columns=[target_col])
        y_train = train_section[target_col]

        x_test = test_section.drop(columns=[target_col])
        y_test = test_section[target_col]

        return x_train, x_test, y_train, y_test

    """
    Executes the k-fold cross-validation evaluation.

    Trains and evaluates the model on each fold, computes the mean of
    the selected metrics, calculates AUC if required, and produces plots
    and output files.

    @:raise RuntimeError: If the k-fold evaluation process fails.
    """

    def evaluate(self):
        try:
            shuffled_dataset = (
                self.dataset
                .sample(frac=1, random_state=self.seed)
                .reset_index(drop=True)
            )
            folds = np.array_split(shuffled_dataset, self.K_tests)

            metrics_list = dict()
            y_test_all = np.array([])
            y_score_all = np.array([])
            y_pred_all = np.array([])

            for i in range(self.K_tests):
                x_train, x_test, y_train, y_test = self.split_dataset_with_strategy(folds, i)

                model = Knn_Algorithm.KNNClassifier(k=self.k_neighbours, p=self.distance_strategy)  # MockKNN(k=5, seed=42, error_rate=0.3)
                model.fit(x_train, y_train)
                y_score, y_pred = model.predict(x_test, pos_label=4)

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