import csv
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ROC(fpr_array: np.array, tpr_array: np.array):
    # --- Traccia ROC ---
    plt.figure()
    plt.plot(fpr_array, tpr_array, marker='o', linestyle='-', color='violet')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def plot_confusion_matrix(cm: np.array):
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


def save_output_result(metrics: dict):
    output_dir = "output_result"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "evaluation_result.csv")

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])  # header

        for metric, value in metrics.items():
            writer.writerow([metric, value])

    result_df = pd.read_csv("output_result/evaluation_result.csv")

    print(result_df)


def save_output_result(metrics: dict, k_metrics: dict = None):
    """
    Salva le metriche in un CSV.
    - metrics: dizionario con le metriche globali (medie)
    - k_metrics: dizionario con le metriche per fold (opzionale)
    """

    output_dir = "output_result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_result.csv")

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

    # Leggi e stampa il CSV appena creato
    result_df = pd.read_csv(output_path)
    print(result_df)
