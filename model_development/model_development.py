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

    def __init__(self, k=3):
        """
        Inizializza il classificatore con il parametro k.
        
        Parametri:
            k (int): Numero di vicini (default=3).
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Memorizza i dati di training (Lazy Learning).
        
        Parametri:
            X (array-like): Matrice delle features di training.
            y (array-like): Vettore delle label di training.
        
        Ritorna:
            None
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, point1, point2):
        """
        Calcola la distanza euclidea tra due punti.
        Richiesto specificamente da Project.pdf.
        
        Parametri:
            point1 (np.array): Primo punto.
            point2 (np.array): Secondo punto.
            
        Ritorna:
            float: La distanza euclidea.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _predict_single(self, x):
        """
        Prevede la classe per un singolo campione gestendo il pareggio in modo casuale.
        
        Parametri:
            x (np.array): Campione di test.
            
        Ritorna:
            int: La classe predetta.
        """
        # 1. Calcolare la distanza da TUTTI i campioni del set di training [cite: 46]
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        # 2. Identificare i k campioni più vicini [cite: 46]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        k_nearest_labels = [label for _, label in k_nearest]
        
        # 3. Classificare scegliendo la label più comune [cite: 48]
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
        Prevede le classi per un intero set di test.
        
        Parametri:
            X_test (array-like): Matrice delle features di test.
            
        Ritorna:
            np.array: Vettore delle predizioni.
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)




