import matplotlib

matplotlib.use("TkAgg")
import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

"""
save the plot of the ROC curve given false positive and true positive rates.

@:param fpr_array: Array of false positive rates.
@:param tpr_array: Array of true positive rates.

@:raise ValueError: If arrays are None, empty, or have different lengths.
@:raise RuntimeError: If plotting fails.
"""


def plot_ROC(fpr_array: np.array, tpr_array: np.array, filename: str = "roc_curve.png"):
    if fpr_array is None:
        raise ValueError("false positive rate array cannot be None")
    if tpr_array is None:
        raise ValueError("True Positive Rate cannot be None")
    if len(fpr_array) != len(tpr_array):
        raise ValueError("fpr and tpr array must have the same length")
    if len(fpr_array) == 0:
        raise ValueError("fpr cannot be empty")

    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        plt.figure()
        plt.plot(fpr_array, tpr_array, marker='o', linestyle='-', color='violet')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        raise RuntimeError("Failed saving ROC curve") from e


"""
Save the plot of a 2x2 confusion matrix.

@:param cm: Confusion matrix of shape 2x2.

@:raise ValueError: If cm is None or not 2x2.
@:raise RuntimeError: If plotting fails.
"""


def plot_confusion_matrix(cm: np.array, filename: str = "confusion_matrix.png"):
    if cm is None:
        raise ValueError("Confusion Matrix cannot be None")
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2")

    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        classes = ['2 (Benigno)', '4 (Maligno)']

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}",
                         ha="center",
                         color="black")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        raise RuntimeError("Failed saving Confusion Matrix") from e


"""
Saves evaluation metrics to a CSV file and prints the content.

@:param metrics: Dictionary of metric names and values.

@:raise TypeError: If metrics is not a dictionary.
@:raise RuntimeError: If writing or reading the CSV fails.
"""


def save_output_result(metrics: dict):
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be dict")

    output_dir = "output_result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_result.csv")

    try:

        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])  # header

            for metric, value in metrics.items():
                writer.writerow([metric, value])

        try:
            result_df = pd.read_csv(output_path)
            print(result_df)
        except Exception as e:
            raise RuntimeError(f"Failed reading the output result CSV at {output_path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed writing output result on CSV at {output_path}") from e


"""
def save_output_result(metrics: dict, k_metrics: dict = None):

    Salva le metriche in un CSV.
    - metrics: dizionario con le metriche globali (medie)
    - k_metrics: dizionario con le metriche per fold (opzionale)


    if not isinstance(metrics, dict):
        raise TypeError("metrics must be dict")
    if k_metrics is not None and not isinstance(k_metrics, dict):
        raise TypeError("k_metrics must be dict")



    output_dir = "output_result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_result.csv")

    try:
        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Se ci sono metriche per fold, scrivile
            if k_metrics:
                # Header dinamico
                fold_keys = list(k_metrics.keys())
                metric_keys = list(next(iter(k_metrics.values())).keys())
                header = ["Fold"] + metric_keys
                writer.writerow(header)

                for fold, fold_metrics in k_metrics.items():
                    row = [fold] + [fold_metrics[m] for m in metric_keys]
                    writer.writerow(row)

                # Riga separatrice vuota
                writer.writerow([])

            # Scrivi le metriche globali
            writer.writerow(["Metric", "Value"])
            for metric, value in metrics.items():
                writer.writerow([metric, value])

        try:
            # Leggi e stampa il CSV appena creato
            result_df = pd.read_csv(output_path)
            print(result_df)
        except Exception as e:
            raise RuntimeError(f"Failed reading output result on CSV at {output_path}")
    except OSError as e:
        raise RuntimeError(f"Failed writing output result on CSV at {output_path}")
        """
