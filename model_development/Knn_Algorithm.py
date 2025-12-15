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
    
    def __init__(self, k: int, x_data_train: pd.DataFrame, y_data_train: pd.Series, distance_strategy: int):
        self.x_data_train = x_data_train      # Store the training data (features).
        self.y_data_train = y_data_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.
        self.distance_strategy = distance_strategy  # Number of the strategy to consider.
        
        """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
                
        """