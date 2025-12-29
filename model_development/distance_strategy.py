from abc import ABC, abstractmethod
import numpy as np

class DistanceStrategy(ABC):
    
    '''
 
    Docstring per DistanceStrategy
 
    '''   
    
    @abstractmethod
    def compute_distance(self, x_train, x_test):
        pass
    
class MinkowskiDistanceStrategy(DistanceStrategy):
    
    """
    implementation of minkiwsy distance.
    Euclidea (p=2) and Manhattan (p=1).
    """
    
    def __init__(self, p):
        self.p = p
        
    def compute_distance(self, x_train, x_test):
        
        if x_train is None:
            raise ValueError("Training data not provided.")
        
        distance = np.sum(np.abs(x_train - x_test) ** self.p, axis=1) ** (1 / self.p)
        return distance
    
class DistanceFactory:
    
    @staticmethod
    def get_distance_strategy(p):
        
        if p == 1:
            return MinkowskiDistanceStrategy(p=1)  # Manhattan Distance
        elif p == 2:
            return MinkowskiDistanceStrategy(p=2)  # Euclidean Distance
        else:
            raise ValueError("Unsupported distance strategy. Use p=1 for Manhattan or p=2 for Euclidean.")