import sys
import os
import re
import numpy as np
import pandas as pd

# Aggiunge le sottocartelle al path per permettere gli import dei moduli
sys.path.append(os.path.join(os.getcwd(), 'data_preprocessing'))
sys.path.append(os.path.join(os.getcwd(), 'model_evaluation'))
sys.path.append(os.path.join(os.getcwd(), 'model_development'))

# Importa la pipeline e la factory di valutazione
# Nota: Potrebbe essere necessario adattare gli import a seconda di come hai strutturato i pacchetti (__init__.py)
from data_preprocessing.pipeline import main as run_pipeline
from model_evaluation.EvaluatorFactory import EvaluatorFactory

if __name__ == "__main__":
    print("--- AVVIO SISTEMA DI CLASSIFICAZIONE TUMORI ---")

    # 1. ESECUZIONE PIPELINE (Caricamento e Pulizia Dati Reali)
    # Assicurati che il percorso del CSV sia corretto
    print("\n[1/2] Esecuzione Data Preprocessing...")
    

    clean_dataset = run_pipeline('data/dati_fia.csv')
    
    # Verifica che il dataset non sia vuoto
    if clean_dataset is None or clean_dataset.empty:
        print("Errore: La pipeline non ha restituito dati validi.")
        exit()
        
    print(f"Dataset pronto: {len(clean_dataset)} campioni caricati.")
    print(f"Colonne disponibili: {list(clean_dataset.columns)}")
    num_samples= len(clean_dataset)
    # 2. CONFIGURAZIONE VALUTAZIONE (Interfaccia Utente)
    print("\n[2/2] Configurazione Valutazione Modello...")
    
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

knn_parameters = input(
    "Select the distance_strategy to apply:\n"
    "1) Manhattan Distance\n"
    "2) Euclidean Distance\n"
)

distance_strategy_parameter : int
while True:
    if knn_parameters == "1":
        distance_strategy_parameter = 1
        break
    elif knn_parameters == "2":
        distance_strategy_parameter = 2
        break
    else:
        print("Invalid choice, please select 1-2.")

k_neighbours = int(input("\nSelect the k_neighbours parameter: "))



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
    # he = holdout_evaluator(df, metrics_array , train_percentage)
    he = EvaluatorFactory.generate_evaluator("holdout", clean_dataset, metrics_array, distance_strategy_parameter, k_neighbours, train_percentage=train_percentage)
    he.evaluate()
elif choice == "2":
    print("Running K-Fold validation...")
    try:
        K_split = int(input("inserisci K: "))
        if (K_split > num_samples):
            raise ValueError
    except ValueError:
        print("Errore: inserisci un numero valido")
        exit(1)
    # kfe = kFoldEvaluator(df, metrics_array , K_split)
    kfe = EvaluatorFactory.generate_evaluator("k-fold", clean_dataset, metrics_array, distance_strategy_parameter, k_neighbours, K_tests=K_split)
    kfe.evaluate()
elif choice == "3":
    print("Running Leave-One-Out validation...")
    # chiama la funzione leave_one_out()
    # looe = LeaveOneOutEvaluator(df, metrics_array)
    looe = EvaluatorFactory.generate_evaluator("loo", clean_dataset, metrics_array, distance_strategy_parameter, k_neighbours)
    looe.evaluate()
elif choice == "4":
    print("Exiting program...")
    exit()
else:
    print("Invalid choice, please select 1-4.")