import re
import numpy as np
import pandas as pd
from holdout_evaluator import holdout_evaluator
from k_fold_evaluator import kFoldEvaluator
from leave_one_out_evaluator import LeaveOneOutEvaluator

if __name__ == "__main__":
    num_samples = 300  # numero di campioni

    df = pd.DataFrame({
        "ID": range(1, num_samples + 1),
        "Sample code number": range(1001, 1001 + num_samples),
        "Clump Thickness": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Uniformity of Cell Size": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Uniformity of Cell Shape": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Marginal Adhesion": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Single Epithelial Cell Size": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Bare Nuclei": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Bland Chromatin": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Normal Nucleoli": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Mitoses": np.round(np.random.uniform(1, 5, num_samples), 1),
        # Alternanza tra benigno (2) e maligno (4) per bilanciare le classi
        "Class": [2, 4] * (num_samples // 2)
    })

    print(df)

VALID_METRICS = set(range(1, 8))  # {1,2,3,4,5,6,7}

while True:
    metrics_input = input(
        "Insert the number of the metrics you want to analyze (comma-separated):\n"
        "1) Accuracy Rate\n"
        "2) Error Rate\n"
        "3) Sensitivity\n"
        "4) Specificity\n"
        "5) Geometric Mean\n"
        "6) Area under the Curve\n"
        "7) All the Above\n"
    )

    #  Controllo: solo cifre e virgole
    if not re.fullmatch(r"\d+(,\d+)*", metrics_input.replace(" ", "")):
        print("Errore: inserisci solo numeri separati da virgole (es. 1,3,6)")
        continue

    #  Parsing
    metrics_array = np.array([int(x) for x in metrics_input.split(",")])

    #  Controllo: numeri ammessi
    if not set(metrics_array).issubset(VALID_METRICS):
        print("Errore: puoi inserire solo numeri da 1 a 7")
        continue

    break  # input valido

print("Metrics selected:", metrics_array)

choice = input(
    "Select the validation to do:\n"
    "1) Holdout\n"
    "2) K-fold\n"
    "3) Leave-one-out\n"
    "4) Exit\n"
)

if choice == "1":
    print("Running Holdout validation...")
    while True:
        try:
            train_percentage = float(input("Inserisci la percentuale di training (tra 0.6 e 0.9): "))
            if 0.6 <= train_percentage <= 0.9:
                break
            else:
                print("Errore: inserisci un valore tra 0.6 e 0.9")
        except ValueError:
            print("Errore: inserisci un numero valido")

    print(f"Percentuale di training selezionata: {train_percentage}")
    he = holdout_evaluator(df, metrics_array, train_percentage)
    he.evaluate()
elif choice == "2":
    print("Running K-Fold validation...")

    kfe = kFoldEvaluator(df, metrics_array, 12)
    kfe.evaluate()
elif choice == "3":
    print("Running Leave-One-Out validation...")
    # chiama la funzione leave_one_out()
    looe = LeaveOneOutEvaluator(df, metrics_array)
    looe.evaluate()
elif choice == "4":
    print("Exiting program...")
    exit()
else:
    print("Invalid choice, please select 1-4.")
