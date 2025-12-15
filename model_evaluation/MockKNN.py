import numpy as np
import random


class MockKNN:
    def __init__(self, k=3, seed=42):
        self.k = k
        random.seed(seed)

    def fit(self, X, y):
        self.classes_ = list(set(y))

    def predict(self, X):
        # predizioni casuali ma valide
        return np.array([
            random.choice(self.classes_)
            for _ in range(len(X))
        ])
