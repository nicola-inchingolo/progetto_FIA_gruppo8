import unittest
import numpy as np
from Knn_Algorithm import KNNClassifier
from distance_strategy import MinkowskiDistanceStrategy, DistanceFactory


class TestDistanceStrategy(unittest.TestCase):
    """Test per le strategie di distanza"""
    
    def test_manhattan_distance(self):
        """Test 1: Verifica il calcolo della distanza di Manhattan (p=1)"""
        strategy = MinkowskiDistanceStrategy(p=1)
        x_train = np.array([[0, 0], [3, 4]])
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 7])  # |0-0| + |0-0| = 0, |3-0| + |4-0| = 7
        np.testing.assert_array_almost_equal(distances, expected)
        print("OK Test Manhattan Distance superato")
    
    def test_euclidean_distance(self):
        """Test 2: Verifica il calcolo della distanza Euclidea (p=2)"""
        strategy = MinkowskiDistanceStrategy(p=2)
        x_train = np.array([[0, 0], [3, 4]]
                           )
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 5])  # sqrt((3-0)^2 + (4-0)^2) = 5
        np.testing.assert_array_almost_equal(distances, expected)
        print("OK Test Euclidean Distance superato")
    
    def test_distance_factory(self):
        """Test 3: Verifica che la Factory crei la strategia corretta"""
        # Test per Manhattan
        strategy_manhattan = DistanceFactory.get_distance_strategy(1)
        self.assertIsInstance(strategy_manhattan, MinkowskiDistanceStrategy)
        self.assertEqual(strategy_manhattan.p, 1)
        
        # Test per Euclidea
        strategy_euclidean = DistanceFactory.get_distance_strategy(2)
        self.assertIsInstance(strategy_euclidean, MinkowskiDistanceStrategy)
        self.assertEqual(strategy_euclidean.p, 2)
        
        # Test per valore non supportato
        with self.assertRaises(ValueError):
            DistanceFactory.get_distance_strategy(3)
        
        print("OK Test Distance Factory superato")


class TestKNNClassifier(unittest.TestCase):
    """Test per il classificatore KNN"""
    
    def setUp(self):
        """Prepara i dati di test prima di ogni test"""
        # Dati di training semplici: 2 classi ben separate
        self.x_train = np.array([
            [1, 1], [1.5, 1.5], [2, 2],     # Classe 2
            [8, 8], [8.5, 8.5], [9, 9]      # Classe 4
        ])
        self.y_train = np.array([2, 2, 2, 4, 4, 4])
    
    def test_knn_prediction_simple(self):
        """Test 4: Predizione su punti chiaramente classificabili"""
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point vicino alla classe 2
        X_test = np.array([[1.2, 1.2]])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 2, "Il punto dovrebbe essere classificato come classe 2")
        self.assertLess(y_proba[0], 0.5, "La probabilità di classe 4 dovrebbe essere bassa")
        print(f"OK Test KNN Prediction Simple superato: predetto={y_pred[0]}, prob={y_proba[0]:.2f}")
    
    def test_knn_prediction_opposite_class(self):
        """Test 5: Predizione su punto dell'altra classe"""
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point vicino alla classe 4
        X_test = np.array([[8.2, 8.2]])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 4, "Il punto dovrebbe essere classificato come classe 4")
        self.assertGreater(y_proba[0], 0.5, "La probabilità di classe 4 dovrebbe essere alta")
        print(f"OK Test KNN Opposite Class superato: predetto={y_pred[0]}, prob={y_proba[0]:.2f}")
    
    def test_knn_with_different_k(self):
        """Test 6: Verifica che diversi valori di k funzionino"""
        for k in [1, 3, 5]:
            knn = KNNClassifier(k=k, p=2)
            knn.fit(self.x_train, self.y_train)
            
            X_test = np.array([[1.5, 1.5], [8.5, 8.5]])
            y_proba, y_pred = knn.predict(X_test, pos_label=4)
            
            self.assertEqual(len(y_pred), 2, f"Dovrebbero esserci 2 predizioni per k={k}")
            print(f"OK Test KNN con k={k} superato: {y_pred}")


