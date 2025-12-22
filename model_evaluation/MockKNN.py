import numpy as np
import random

import pandas as pd


class MockKNN:
    def __init__(self, k=3, seed=None, error_rate=0.3):
        """
        k: numero di vicini (non usato nel mock)
        seed: per riproducibilità
        error_rate: probabilità di predire la classe sbagliata
        """
        self.k = k
        self.error_rate = error_rate
        if seed is not None:
            random.seed(seed)

    def fit(self, X, y):
        # memorizza le classi presenti
        self.classes_ = list(set(y))

    def predict(self, X, y_test=None):
        # y_pred: classi 2 o 4 (come prima)
        # y_score: punteggio continuo per calcolare ROC/AUC
        y_pred = np.random.choice([2, 4], size=len(X))

        # y_score: probabilità simulata di essere 4, continua tra 0 e 1
        y_score = np.random.uniform(0, 1, size=len(X))

        print(y_score)

        return y_pred, y_score
