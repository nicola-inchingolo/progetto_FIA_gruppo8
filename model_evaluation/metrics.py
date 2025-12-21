import math
import numpy as np


# pos_label = evento di interesse
# neg_label = evento non tipico
def confusion_matrix_binary(y_test, y_pred, pos_label=4, neg_label=2, normalize=True):
    TP = np.sum((y_test == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_test == neg_label) & (y_pred == neg_label))
    FP = np.sum((y_test == neg_label) & (y_pred == pos_label))
    FN = np.sum((y_test == pos_label) & (y_pred == neg_label))

    cm = np.array([[TN, FP],
                   [FN, TP]])
    return cm


def calculate_mean_metrics(metrics_list: dict):
    sum_metrics = {}
    for fold in metrics_list.values():
        for key, value in fold.items():
            if key not in sum_metrics:
                sum_metrics[key] = 0
            sum_metrics[key] += value

    num_folds = len(metrics_list)
    mean_metrics = {key: sum_metrics[key] / num_folds for key in sum_metrics}
    return mean_metrics


class metrics:

    def __init__(self, confusion_matrix: np.ndarray):
        self.TN = confusion_matrix[0][0]
        self.FP = confusion_matrix[0][1]
        self.FN = confusion_matrix[1][0]
        self.TP = confusion_matrix[1][1]

    def calculate_accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def calculate_error_rate(self):
        return (self.FP + self.FN) / (self.TP + self.TN + self.FP + self.FN)

    def calculate_sensitivity(self):
        denom = self.TP + self.FN
        return self.TP / denom if denom != 0 else 0

    def calculate_specificity(self):
        denom = self.TN + self.FP
        return self.TN / denom if denom != 0 else 0

    def calculate_geometric_mean(self):
        sensitivity = self.calculate_sensitivity()
        specificity = self.calculate_specificity()
        return math.sqrt(sensitivity * specificity)

    def calculate_precision(self):
        denom = self.TP + self.FP
        return self.TP / denom if denom != 0 else 0

    def calculate_f1_score(self):
        precision = self.calculate_precision()
        sensitivity = self.calculate_sensitivity()
        denom = precision + sensitivity
        return (2 * precision * sensitivity) / denom if denom != 0 else 0

    def calculate_all_the_above(self):
        return {
            'Accuracy': self.calculate_accuracy(),
            'Error Rate': self.calculate_error_rate(),
            'Sensitivity': self.calculate_sensitivity(),
            'Specificity': self.calculate_specificity(),
            'G-Mean': self.calculate_geometric_mean(),
            'Precision': self.calculate_precision(),
            'F1-Score': self.calculate_f1_score()
        }