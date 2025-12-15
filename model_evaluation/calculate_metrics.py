import math
import numpy as np


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
        return self.TP / (self.TP + self.FN)

    def calculate_specificity(self):
        return self.TN / (self.TN + self.FP)

    def calculate_geometric_mean(self):
        return math.sqrt(self.calculate_sensitivity() * self.calculate_specificity())

    def calculate_precision(self):
        denom = self.TP + self.FP
        return self.TP / denom if denom != 0 else 0

    def calculate_f1_score(self):
        precision = self.calculate_precision()
        sensitivity = self.calculate_sensitivity()
        denom = precision + sensitivity
        return (2 * precision * sensitivity) / denom if denom != 0 else 0

    def calculate_auc(self):
        return self.calculate_sensitivity() / self.calculate_specificity()

    def calculate_all_the_above(self):
        return {
            'Accuracy': self.calculate_accuracy(),
            'Error Rate': self.calculate_error_rate(),
            'Sensitivity': self.calculate_sensitivity(),
            'Specificity': self.calculate_specificity(),
            'G-Mean': self.calculate_geometric_mean(),
            'Precision': self.calculate_precision(),
            'F1-Score': self.calculate_f1_score(),
            'AUC': self.calculate_auc()
        }