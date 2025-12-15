import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import seaborn as sns 

class KNNClassifier:
    """
    Implementazione di un classificatore K-Nearest Neighbors (K-NN) from scratch.
    
    Attributi:
        k (int): Numero di vicini da considerare.
        X_train (np.array): Features del set di training.
        y_train (np.array): Labels del set di training.
    """

   
        
    def __init__(self, k=3, p=2): 
            
        """
        Inizializza il classificatore con il parametro k.    
        Parametri:
        k (int): Numero di vicini (default=3).
        p (int): Parametro della distanza di Minkowski (default=2, distanza Euclidea).
        """
        self.k = k
        self.p = p 
        self.X_train = None
        self.y_train = None
        

    def fit(self, X, y):
        """
        Memorizza i dati di training.
        
        Parametri:
            X (array-like): Matrice delle features di training.
            y (array-like): Vettore delle label di training.
        
        Ritorna:
            None
        """
        # Conversione in numpy array per supportare operazioni vettoriali
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _minkowski_distance_vectorized(self, x, p):
        """
        Calcola la distanza di Minkowski (generalizzazione di Euclidea e Manhattan).
        Formula: (sum(|x - y|^p))^(1/p)
        
        Se p=2 -> Distanza Euclidea
        Se p=1 -> Distanza di Manhattan
        """
        return np.sum(np.abs(self.X_train - x) ** p, axis=1) ** (1 / p)

    def _predict_single(self, x):
        """
        Prevede la classe per un singolo campione gestendo il pareggio in modo casuale.
        
        Parametri:
            x (np.array): Campione di test.
            
        Ritorna:
            int: La classe predetta.
        """
        # 1. Calcolare la distanza da TUTTI i campioni del set di training
        # Utilizziamo la versione vettorializzata per efficienza
        distances = self._euclidean_distance_vectorized(x)
        
        # 2. Identificare i k campioni più vicini
        # argsort restituisce gli indici che ordinerebbero l'array
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # 3. Classificare scegliendo la label più comune
        counts = Counter(k_nearest_labels)
        max_votes = max(counts.values())
        
        # Lista dei candidati che hanno il massimo dei voti
        candidates = [label for label, count in counts.items() if count == max_votes]
        
        # 4. Se c'è un pareggio, si sceglie in modo casuale 
        if len(candidates) > 1:
            return random.choice(candidates)
        else:
            return candidates[0]

    def predict(self, X_test):
        """
        Prevede le classi per un intero set di dati di test.
        
        Parametri:
            X_test (array-like): Matrice delle features di test.
        
        Ritorna:
            np.array: Vettore con le predizioni per ogni campione.
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
