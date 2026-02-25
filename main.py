import sys
import os
import re
import numpy as np
import pandas as pd

# Adds subfolders to the path to allow module imports
sys.path.append(os.path.join(os.getcwd(), 'data_preprocessing'))
sys.path.append(os.path.join(os.getcwd(), 'model_evaluation'))
sys.path.append(os.path.join(os.getcwd(), 'model_development'))

# Import the pipeline and evaluation factory
# Note: You may need to adapt the imports depending on how you have structured your packages (__init__.py)
from data_preprocessing.pipeline import main as run_pipeline
from model_evaluation.evaluator_factory import EvaluatorFactory

if __name__ == "__main__":
    print("--- START-UP OF THE TUMOUR CLASSIFICATION SYSTEM ---")

    # 1. PIPELINE EXECUTION (Loading and Cleaning Real Data)
    # Ensure that the CSV path is correct
    print("\n[1/2] Esecution Data Preprocessing...")

    clean_dataset = run_pipeline()

    # Verify that the dataset is not empty
    if clean_dataset is None or clean_dataset.empty:
        print("Error: The pipeline did not return valid data.")
        exit()

    print(f"Dataset ready: {len(clean_dataset)} samples loaded.")
    print(f"Available columns: {list(clean_dataset.columns)}")
    num_samples = len(clean_dataset)
    # 2. EVALUATION CONFIGURATION (User Interface)
    print("\n[2/2] Configuration for Model Evaluation...")

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

    # Check: only digits and commas
    if not re.fullmatch(r"\d+(,\d+)*", metrics_input.replace(" ", "")):
        print("Error: insert only numbers separated by commas (e.g. 1,3,6)")
        continue

    #  Parsing
    metrics_array = np.array([int(x) for x in metrics_input.split(",")])

    # Check: permitted numbers
    if not set(metrics_array).issubset(VALID_METRICS):
        print("Error: you can only insert numbers from 1 to 7")
        continue

    break  # input not valid

print("Metrics selected:", metrics_array)


distance_strategy_parameter: int
while True:
    knn_parameters = input(
        "Select the distance_strategy to apply:\n"
        "1) Manhattan Distance\n"
        "2) Euclidean Distance\n"
    )
    if knn_parameters == "1":
        distance_strategy_parameter = 1
        break
    elif knn_parameters == "2":
        distance_strategy_parameter = 2
        break
    else:
        print("Invalid choice, please select 1-2.")

k_neighbours = int(input("\nSelect the k_neighbours parameter: "))

while True:
    choice = input(
        "Select the validation to do:\n"
        "1) Holdout\n"
        "2) K-fold\n"
        "3) Leave-one-out\n"
        "4) Exit\n"
    )

    if choice in ["1", "2", "3", "4"]:
        break
    else:
        print("Invalid choice, please select a number between 1 and 4.")

if choice == "1":
    print("Running Holdout validation...")
    while True:
        try:
            train_percentage = float(input("Insert the training percentage (between 0.6 and 0.9): "))
            if 0.6 <= train_percentage <= 0.9:
                break
            else:
                print("Error: insert a value between 0.6 and 0.9")
        except ValueError:
            print("Error: insert a valid number")

    print(f"Selected training percentage: {train_percentage}")

    he = EvaluatorFactory.generate_evaluator("holdout", clean_dataset, metrics_array, distance_strategy_parameter,
                                             k_neighbours, train_percentage=train_percentage)
    he.evaluate()
elif choice == "2":
    print("Running K-Fold validation...")
    try:
        K_split = int(input("Insert K: "))
        if (K_split > num_samples):
            raise ValueError
    except ValueError:
        print("Error: insert a valid number")
        exit(1)
    kfe = EvaluatorFactory.generate_evaluator("k-fold", clean_dataset, metrics_array, distance_strategy_parameter,
                                              k_neighbours, K_tests=K_split)
    kfe.evaluate()
elif choice == "3":
    print("Running Leave-One-Out validation...")
    # call the function leave_one_out()
    looe = EvaluatorFactory.generate_evaluator("loo", clean_dataset, metrics_array, distance_strategy_parameter,
                                               k_neighbours)
    looe.evaluate()
elif choice == "4":
    print("Exiting program...")
    exit()
else:
    print("Invalid choice, please select 1-4.")
