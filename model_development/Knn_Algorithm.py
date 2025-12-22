import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
from abc import ABC, abstractmethod
from distance_strategy import DistanceFactory


class KNNClassifier:
    """
    Sets up the KNN classifier by defining the number of neighbors (k), 
    the training dataset, and the distance calculation strategy.
    The stored training data serves as the reference for locating 
    the nearest neighbors for new test points,
    while the distance strategy dictates the specific formula 
    used for these computations.
    
    """
    
           
    
    def __init__(self, k: int, p: int, x_data_train: pd.DataFrame = None, y_data_train: pd.Series = None):
        self.x_data_train = x_data_train      # Store the features for the training data.
        self.y_data_train = y_data_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.
        self.p = p  # Number of the strategy to consider.
        self.distance_strategy = DistanceFactory.get_distance_strategy(p)  # Initialize the distance strategy
        
    """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
                
    """
    
    
    def fit(self, x_data_train, y_data_train):
        """
        Memorizza i dati di training (Training step).
        """
        self.x_data_train = np.array(x_data_train)
        self.y_data_train = np.array(y_data_train)     
                       
    def _get_k_neighbors(self, x):
        """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
        """
        # 1. Calcola la distanza tra il punto x_test e TUTTI i punti di training
        distances = []
        distances = self.distance_strategy.compute_distance(self.x_data_train, x)
        
        # 2. Ottieni gli indici che ordinerebbero l'array delle distanze (dal più piccolo al più grande)
        # np.argsort restituisce gli indici, non i valori.
        # [:self.k] prende solo i primi k indici (i più vicini)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Restituisci le LABELS corrispondenti a quegli indici
        return self.y_data_train[k_indices]
        
    def predict(self, X_test, pos_label=4):
        """
        Predicts the labels for a set of test points
        and calculates the PURE probability that the sample is Malignant (Class 4).
        Used to build the ROC curve.
        """
        
        X_test = np.array(X_test)
        predictions = []
        y_proba = []
        
        for x in X_test:
            
            # starting the prediction process for each test point x
            
            # 1. get k nearest neighbors
            neighbors_labels = self._get_k_neighbors(x)
            
            # 2. count the votes for each class
            counts = Counter(neighbors_labels)
            max_votes = max(counts.values())
            
            # 3. find all classes with max votes (to handle ties)
            candidates = [label for label, count in counts.items() if count == max_votes]
            
            # 4. Assign label (Random if tie, otherwise the winner)
            if len(candidates) > 1:
                predictions.append(random.choice(candidates))
            else:
                predictions.append(candidates[0])
                
            # Calculate the raw probability of being Malignant (Class 4)
            
            # 1. Count how many neighbors are Malignant 
            positive_votes = np.sum(neighbors_labels == pos_label)
            
            # 2. Calculate the percentage 
            prob = positive_votes / self.k
            
            # 3. Save the raw data
            y_proba.append(prob)
            
        return np.array(y_proba) , np.array(predictions)
     
       
# --- BLOCCO MAIN PER IL TEST ---
# --- BLOCCO MAIN PER IL TEST ---
if __name__ == "__main__":
    print("--- Test del KNN con Design Patterns (Strategy) ---")

    # 1. Dati di Training Fittizi
    # Gruppo A (Classe 2): Valori bassi
    train_A = [[1.0, 1.1], [1.2, 1.0], [0.9, 0.8], [2.0, 2.1], [1.5, 1.3], [0.8, 1.2]]
    # Gruppo B (Classe 4): Valori alti
    train_B = [[5.0, 5.1], [5.2, 5.3], [6.0, 5.9], [5.5, 5.5], [5.8, 5.2], [4.9, 5.0]]
    # Gruppo C (Classe 2): Alcuni punti intermedi-bassi per creare ambiguità
    train_C = [[2.5, 2.5], [2.8, 2.7]]
    # Gruppo D (Classe 4): Alcuni punti intermedi-alti per creare ambiguità
    train_D = [[3.5, 3.5], [3.2, 3.3]]
    
    x_data_train = np.array(train_A + train_B + train_C + train_D)
    
    # Labels (2 = Benigno/A, 4 = Maligno/B)
    y_data_train = np.array([2] * len(train_A) + [4] * len(train_B) + [2] * len(train_C) + [4] * len(train_D))

    print(f"Dati di Training caricati: {len(x_data_train)} campioni.")

    # 2. Dati di Test Fittizi
    # Punti progettati per testare diverse situazioni:
    X_test = np.array([
        [1.1, 1.1],   # Chiaramente classe 2 (vicino a train_A)
        [5.1, 5.1],   # Chiaramente classe 4 (vicino a train_B)
        [3.0, 3.0],   # PUNTO AMBIGUO: equidistante tra classi - possibile pareggio!
        [2.9, 2.9],   # ALTRO PUNTO AMBIGUO: vicino a confine - possibile pareggio!
        [3.4, 3.4],   # ALTRO PUNTO AMBIGUO: tra i cluster intermedi - possibile pareggio!
        [0.5, 0.5],   # Estremo basso (classe 2)
        [6.5, 6.5]    # Estremo alto (classe 4)
    ])

    try:
        print(f"Dati di Test da classificare:\n{X_test}\n")

        # 3. Creazione e Configurazione Modello
        # La Factory creerà automaticamente la distanza corretta per p=2 (Euclidea)
        # IMPORTANTE: k=4 (pari) per permettere pareggi 2-2 e attivare random.choice()
        knn = KNNClassifier(k=4, p=2)
            
        # 4. Addestramento (Fitting)
        knn.fit(x_data_train, y_data_train)
            
        # 5. Predizione
        # MODIFICA IMPORTANTE: Ora il metodo restituisce DUE array.
        # probs -> Serve per la curva ROC (valori float tra 0.0 e 1.0)
        # predictions -> Serve per la Confusion Matrix (valori 2 o 4)
        probs, predictions = knn.predict(X_test, pos_label=4)
            
        # 6. Stampa Risultati
        print("--- Risultati Predizione ---")
        print(f"{'Punto':<20} | {'Classe Predetta':<15} | {'Prob. Maligno (ROC)':<20}")
        print("-" * 60)
        
        for i, point in enumerate(X_test):
            point_str = str(point)
            # Stampiamo sia la classe decisa che la probabilità calcolata
            print(f"{point_str:<20} | {predictions[i]:<15} | {probs[i]:.4f}")
            
    except Exception as e:
        # Questo cattura errori come problemi con la Factory o dati mancanti
        print(f"Errore durante il test del KNN: {e}")