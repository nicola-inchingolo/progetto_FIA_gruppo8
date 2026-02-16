import unittest
import numpy as np
import sys
import os
import pandas as pd

# Add the current directory to the path so that we can import local modules and perform a generic test.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Knn_Algorithm import KNNClassifier
from distance_strategy import MinkowskiDistanceStrategy, DistanceFactory


class TestDistanceStrategy(unittest.TestCase):
    """Testing for distance strategies used in KNNClassifier"""
    
    def test_manhattan_distance(self):
        
        """
        Test 1:
        verify the calculation of Manhattan distance (p=1)
        """
        
        strategy = MinkowskiDistanceStrategy(p=1)
        x_train = np.array([[0, 0], [3, 4]])
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 7])  # |0-0| + |0-0| = 0, |3-0| + |4-0| = 7
        np.testing.assert_array_almost_equal(distances, expected)
        
        print("OK Test Manhattan Distance is a success")
        
    
    def test_euclidean_distance(self):
        
        """
        Test 2:
        verify the calculation of Euclidean distance (p=2)
        """
        
        strategy = MinkowskiDistanceStrategy(p=2)
        x_train = np.array([[0, 0], [3, 4]]
                           )
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 5])  # sqrt((3-0)^2 + (4-0)^2) = 5
        np.testing.assert_array_almost_equal(distances, expected)
        
        print("OK Test Euclidean Distance is a success")
        
    
    def test_distance_factory(self):
        
        """
        Test 3: 
        verify that the Factory creates the correct strategy based on the input parameter p.
        """
       
        # Test for Manhattan
        strategy_manhattan = DistanceFactory.get_distance_strategy(1)
        self.assertIsInstance(strategy_manhattan, MinkowskiDistanceStrategy, 
                              "The factory didn't create a Minkowski strategy for p=1")
        self.assertEqual(strategy_manhattan.p, 1, 
                         "The factory didn't set p=1 correctly for the Manhattan strategy")
        
        # Test for Euclidean
        strategy_euclidean = DistanceFactory.get_distance_strategy(2)
        self.assertIsInstance(strategy_euclidean, MinkowskiDistanceStrategy, 
                              "The factory didn't create a Minkowski strategy for p=2")
        self.assertEqual(strategy_euclidean.p, 2, 
                         "The factory didn't set p=2 correctly for the Euclidean strategy")
        
        # Test for unsupported value
        with self.assertRaises(ValueError):
            DistanceFactory.get_distance_strategy(3)
        
        print("OK Test Distance Factory is a success")


class TestKNNClassifier(unittest.TestCase):
    
    """ Testing for KNNClassifier functionality """
    
    def setUp(self):
        
        """
        Preparation of common data for testing.
        Using a simulated pandas DataFrame.
        """
        
        self.x_train = pd.DataFrame({
            'Feature1': [1.0, 1.1, 5.0, 5.1],
            'Feature2': [1.0, 1.0, 5.0, 5.0]
        })
        # Classes: 2 = Benign, 4 = Malignant
        self.y_train = pd.Series([2, 2, 4, 4])
        
    
    def test_knn_prediction_simple(self):
        
        """Test 4: 
        Prediction on points clearly classifiable as class 2 (benign) 
        and verify that the probability of class 4 (malignant) is low.
        """
        
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point near class 2
        X_test = pd.DataFrame([[1.2, 1.2]], columns=['Feature1', 'Feature2'])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 2, "The point should be classified as class 2")
        self.assertLess(y_proba[0], 0.5, "The probability of class 4 should be low")
        
        print(f"OK Test KNN Prediction Simple is a success: predicted={y_pred[0]}, prob={y_proba[0]:.2f}")
        
    
    def test_knn_prediction_opposite_class(self):
        
        """
        Test 5: 
        Prediction on points clearly classifiable as class 4 (malignant) 
        and verify that the probability of class 4 is high.
        """
        
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point near class 4
        X_test = pd.DataFrame([[8.2, 8.2]], columns=['Feature1', 'Feature2'])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 4, "The point should be classified as class 4")
        self.assertGreater(y_proba[0], 0.5, "The probability of class 4 should be high")
        
        print(f"OK Test KNN Opposite Class is a success: predicted={y_pred[0]}, prob={y_proba[0]:.2f}")
        
    
    def test_knn_with_different_k(self):
        
        """
        Test 6: 
        Verify that different values of k work correctly
        """
        
        for k in [1, 3, 5]:
            knn = KNNClassifier(k=k, p=2)
            knn.fit(self.x_train, self.y_train)
            
            X_test = pd.DataFrame([[1.5, 1.5], [8.5, 8.5]], columns=['Feature1', 'Feature2'])
            y_proba, y_pred = knn.predict(X_test, pos_label=4)
            
            self.assertEqual(len(y_pred), 2, f"It should be 2 predictions for k={k}")
            
            print(f"OK Test KNN con k={k} superato: {y_pred}")


    def test_initialization(self):
        
        """
        Test 7: 
        Verify that the parameters (including the seed) are correctly saved.
        """
        
        knn = KNNClassifier(k=3, p=2, seed=123)
        self.assertEqual(knn.k, 3)
        self.assertEqual(knn.p, 2)
        self.assertEqual(knn.seed, 123, "The seed was not correctly saved in init")
        
        print("OK Test Initialization is a success: parameters saved correctly")
        

    def test_reproducibility_tie_break(self):
        
        """
        Test 8: 
        Verify that the Tie-Break (tie) is deterministic thanks to the SEED.
        Without the seed, this test would fail.
        """
        
        
        # Let's create a symmetrical training set to force a tie
        X_train_tie = pd.DataFrame([[1, 1], [5, 5]], columns=['Feature1', 'Feature2']) # Classe 2 e Classe 4
        y_train_tie = pd.Series([2, 4])
        
        # Exactly in the middle (3, 3) -> Equal distance
        X_test_tie = pd.DataFrame([[3, 3]], columns=['Feature1', 'Feature2'])
        
        # Testing both metrics
        for p_val in [1, 2]:
            with self.subTest(p=p_val):
                # Instance A
                knn_a = KNNClassifier(k=2, p=p_val, seed=42)
                knn_a.fit(X_train_tie, y_train_tie)
                y_proba, pred_a = knn_a.predict(X_test_tie)
                
                # Instance B
                knn_b = KNNClassifier(k=2, p=p_val, seed=42)
                knn_b.fit(X_train_tie, y_train_tie)
                y_proba, pred_b = knn_b.predict(X_test_tie)
                
                self.assertEqual(pred_a[0], pred_b[0], 
                                 f"It failed with p={p_val}: the seed did not guarantee determinism.")
        
        print("OK Test Reproducibility Tie-Break is a success: identical results with the same seed for both metrics")
        

    def test_output_format_consistency(self):
        
        """
        Test 9:
        Verify that the predict method always returns a tuple (probabilities, predictions).
        Check different combinations of k (neighbors) and p (distance).
        """
        
        X_test = pd.DataFrame([[2.0, 2.0], [4.0, 4.0]], columns=['Feature1', 'Feature2'])
        
        # We vary both k (neighbours) and p (distance) for robustness
        configs = [
            {'k': 1, 'p': 1}, # 1 neighbor, Manhattan Distance
            {'k': 1, 'p': 2}, # 1 neighbor, Euclidean Distance
            {'k': 3, 'p': 1}, # 3 neighbors, Manhattan Distance
            {'k': 3, 'p': 2}  # 3 neighbors, Euclidean Distance
        ]

        for config in configs:
            k, p = config['k'], config['p']
            
            with self.subTest(k=k, p=p):
                knn = KNNClassifier(k=k, p=p)
                knn.fit(self.x_train, self.y_train)
                
                output = knn.predict(X_test)
                
                # 1. It must be a tuple of length 2
                self.assertIsInstance(output, tuple, f"The output is not a tuple with k={k}, p={p}")
                self.assertEqual(len(output), 2, f"Wrong tuple length with k={k}, p={p}")
                
                probs, preds = output
                
                # 2. MUst be numpy array
                self.assertIsInstance(probs, np.ndarray, f"Probabilities should be a numpy array with k={k}, p={p}")
                self.assertIsInstance(preds, np.ndarray, f"Predictions should be a numpy array with k={k}, p={p}")
                
                # 3. The probabilities must be floats between 0 and 1
                self.assertTrue(np.all((probs >= 0) & (probs <= 1)), f"Probabilities out of range [0,1] with k={k}, p={p}")
                
                # 4. The predictions must be of the correct class (2 or 4)
                self.assertTrue(np.all(np.isin(preds, [2, 4])), f"Predictions contain unknown classes with k={k}, p={p}")
        
        print("OK Test Output Format Consistency is a success: correct format for all configurations")
        

if __name__ == '__main__':
    unittest.main()