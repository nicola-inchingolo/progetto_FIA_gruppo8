import matplotlib
matplotlib.use("TkAgg")
import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#plotta i grafici su un png

def plot_ROC(fpr_array: np.array, tpr_array: np.array):
    if fpr_array is None:
        raise ValueError("false positive rate array cannot be None")
    if tpr_array is None:
        raise ValueError("True Positive Rate cannot be None")

    if len(fpr_array) != len(tpr_array):
        raise ValueError("fpr and tpr array must have the same length")

    if len(fpr_array) == 0:
        raise ValueError("fpr cannot be empty")
    if len(tpr_array) == 0:
        raise ValueError("tpr cannot be empty")

    try:
        # Traccia ROC
        plt.figure()
        plt.plot(fpr_array, tpr_array, marker='o', linestyle='-', color='violet')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()
    except Exception as e:
        raise RuntimeError("Failed plotting ROC curve") from e


def plot_confusion_matrix(cm: np.array):
    if cm is None:
        raise ValueError("Confusione Matrix cannot be NOne")
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2")

    try:
        classes = ['2 (Benigno)', '4 (Maligno)']

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Tick sugli assi
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Scrive i valori nella matrice
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}",
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "black")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise RuntimeError("Failed plotting Confusion Matrix") from e


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