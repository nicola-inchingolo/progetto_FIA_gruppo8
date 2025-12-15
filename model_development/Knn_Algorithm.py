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
    
    def _minkowski_distance_vectorized(self, x, p):
            """
            Calcola la distanza di Minkowski (generalizzazione di Euclidea e Manhattan).
            Formula: (sum(|x - y|^p))^(1/p)
            
            Se p=2 -> Distanza Euclidea
            Se p=1 -> Distanza di Manhattan
            """
            return np.sum(np.abs(self.X_train - x) ** p, axis=1) ** (1 / p)
    
    def __init__(self, k: int, x_data_train: pd.DataFrame, y_data_train: pd.Series, p: int):
        self.y_data_train = y_data_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.
        self.p = p  # Number of the strategy to consider.
        
        """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
                
        """
        
       