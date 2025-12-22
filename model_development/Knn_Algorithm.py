import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter


class KNNClassifier:
    """
    Sets up the KNN classifier by defining the number of neighbors (k), 
    the training dataset, and the distance calculation strategy.
    The stored training data serves as the reference for locating 
    the nearest neighbors for new test points,
    while the distance strategy dictates the specific formula 
    used for these computations.
    
    """
    
           
    
    def __init__(self, k: int = 3, x_data_train: pd.DataFrame = None, y_data_train: pd.Series = None, p: int = 2):
        self.x_data_train = x_data_train      # Store the features for the training data.
        self.y_data_train = y_data_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.
        self.p = p  # Number of the strategy to consider.
        
    """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
                
    """
    def _minkowski_distance_vectorized(self, x, p):
        
    # RICORDIAMOCI DI METTERE INPUT(P) NEL FILE MAIN COSì DA SCEGLIERE QUALE STRATEGIA USARE    
    
        distance = np.sum(np.abs(self.x_data_train - x) ** p, axis=1) ** (1 / p)
        
        return distance
    """
        Calcola la distanza di Minkowski (generalizzazione di Euclidea e Manhattan).
        Formula: (sum(|x - y|^p))^(1/p)
        Parametri: 
        Se p=2 -> Distanza Euclidea
        Se p=1 -> Distanza di Manhattan
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
        distances = self._minkowski_distance_vectorized(x, self.p)
        
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
if __name__ == "__main__":
    print("--- Test del KNN con Distanza di Minkowski (p=2) (k=3)---")

    # 1. Dati di Training Fittizi (Features: Altezza, Peso - Esempio astratto)
    # Gruppo A (Classe 2): Valori bassi
    train_A = [[1.0, 1.1], [1.2, 1.0], [0.9, 0.8], [2.0, 2.1]]
    # Gruppo B (Classe 4): Valori alti
    train_B = [[5.0, 5.1], [5.2, 5.3], [6.0, 5.9], [5.5, 5.5]]
    
    x_data_train = np.array(train_A + train_B)
    
    # Labels (2 = Benigno/A, 4 = Maligno/B)
    y_data_train = np.array([2] * len(train_A) + [4] * len(train_B))

    print(f"Dati di Training caricati: {len(x_data_train)} campioni.")

    # 2. Dati di Test Fittizi
    # Punto 1: Vicino al gruppo A [1.1, 1.1] -> Ci aspettiamo 2
    # Punto 2: Vicino al gruppo B [5.1, 5.1] -> Ci aspettiamo 4
    # Punto 3: Punto intermedio/ambiguo [3.0, 3.0] -> Dipende dai vicini
    X_test = np.array([
        [1.1, 1.1], 
        [5.1, 5.1], 
        [4.0, 2.0] 
    ])

    try:
        print(f"Dati di Test da classificare:\n{X_test}\n")

        # 3. Creazione e Configurazione Modello
        # Usiamo p=2 (Euclidea)
        knn = KNNClassifier(k=3, p=2)
            
        # 4. Addestramento (Fitting)
        knn.fit(x_data_train, y_data_train)
            
        # 5. Predizione
        predictions = knn.predict(X_test)
            
        # 6. Stampa Risultati
        print("--- Risultati Predizione ---")
        for i, point in enumerate(X_test):
            point_str = str(point)
            print(f"Punto {point_str} -> Classe Predetta: {predictions[i]}")
    except Exception as e:
        print(f"Errore durante il test del KNN: {e}")