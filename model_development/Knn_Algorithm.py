import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
from abc import ABC, abstractmethod
from  model_development.distance_strategy import DistanceFactory


class KNNClassifier:
    """
    Sets up the KNN classifier by defining the number of neighbors (k), 
    the training dataset, and the distance calculation strategy.
    The stored training data serves as the reference for locating 
    the nearest neighbors for new test points,
    while the distance strategy dictates the specific formula 
    used for these computations.
    
    """
    
           
    
    def __init__(self, k: int, p: int, x_data_train: pd.DataFrame = None, y_data_train: pd.Series = None, seed: int = 42):
        self.x_data_train = x_data_train      # Store the features for the training data.
        self.y_data_train = y_data_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.
        self.p = p  # Number of the strategy to consider.
        self.distance_strategy = DistanceFactory.get_distance_strategy(p)  # Initialize the distance strategy
        self.seed = seed  # Fixed seed for reproducibility in tie-breaking
        
    """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
                
    """
    
    
    def fit(self, x_data_train, y_data_train):
        """
        Storage of training data (Training step).
        """
        self.x_data_train = np.array(x_data_train)
        self.y_data_train = np.array(y_data_train)     
                       
    def _get_k_neighbors(self, x):
        """
        Identifies the k nearest neighbors by calculating distances against the full training set.
        It returns the labels of the k points with the lowest distance to the test point.
        """
        # 1. Calculate the distance between point x_test and ALL training points.
        distances = []
        distances = self.distance_strategy.compute_distance(self.x_data_train, x)
        
       # 2. Obtain the indices that would sort the array of distances (from smallest to largest)
        # np.argsort returns the indices, not the values.
        # [:self.k] takes only the first k indices (the closest ones)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Return the LABELS corresponding to those indices
        return self.y_data_train[k_indices]
        
    def predict(self, X_test, pos_label=4):
        """
        Predicts the labels for a set of test points
        and calculates the PURE probability that the sample is Malignant (Class 4).
        Used to build the ROC curve.
        """
        
        random.seed(self.seed)
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
