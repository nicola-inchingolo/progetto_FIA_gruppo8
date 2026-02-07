import unittest
import numpy as np
import sys
import os
import pandas as pd

# Aggiungiamo la directory corrente al path per poter importare i moduli locali e fare un test generico
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Knn_Algorithm import KNNClassifier
from distance_strategy import MinkowskiDistanceStrategy, DistanceFactory


class TestDistanceStrategy(unittest.TestCase):
    """Test per le strategie di distanza"""
    
    def test_manhattan_distance(self):
        
        """
        Test 1:
        Verifica il calcolo della distanza di Manhattan (p=1)
        """
        
        strategy = MinkowskiDistanceStrategy(p=1)
        x_train = np.array([[0, 0], [3, 4]])
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 7])  # |0-0| + |0-0| = 0, |3-0| + |4-0| = 7
        np.testing.assert_array_almost_equal(distances, expected)
        
        print("OK Test Manhattan Distance superato")
        
    
    def test_euclidean_distance(self):
        
        """
        Test 2:
        Verifica il calcolo della distanza Euclidea (p=2)
        """
        
        strategy = MinkowskiDistanceStrategy(p=2)
        x_train = np.array([[0, 0], [3, 4]]
                           )
        x_test = np.array([0, 0])
        
        distances = strategy.compute_distance(x_train, x_test)
        
        expected = np.array([0, 5])  # sqrt((3-0)^2 + (4-0)^2) = 5
        np.testing.assert_array_almost_equal(distances, expected)
        
        print("OK Test Euclidean Distance superato")
        
    
    def test_distance_factory(self):
        
        """
        Test 3: 
        Verifica che la Factory crei la strategia corretta
        """
       
        # Test per Manhattan
        strategy_manhattan = DistanceFactory.get_distance_strategy(1)
        self.assertIsInstance(strategy_manhattan, MinkowskiDistanceStrategy, 
                              "La factory non ha creato una strategia Minkowski per p=1")
        self.assertEqual(strategy_manhattan.p, 1, 
                         "La factory non ha impostato correttamente p=1 per la strategia Manhattan")
        
        # Test per Euclidea
        strategy_euclidean = DistanceFactory.get_distance_strategy(2)
        self.assertIsInstance(strategy_euclidean, MinkowskiDistanceStrategy, 
                              "La factory non ha creato una strategia Minkowski per p=2")
        self.assertEqual(strategy_euclidean.p, 2, 
                         "La factory non ha impostato correttamente p=2 per la strategia Euclidea")
        
        # Test per valore non supportato
        with self.assertRaises(ValueError):
            DistanceFactory.get_distance_strategy(3)
        
        print("OK Test Distance Factory superato")


class TestKNNClassifier(unittest.TestCase):
    
    """ Test per il classificatore KNN  """
    
    def setUp(self):
        
        """
        Preparazione dati comuni per i test.
        Usiamo un DataFrame pandas simulato.
        """
        
        self.x_train = pd.DataFrame({
            'Feature1': [1.0, 1.1, 5.0, 5.1],
            'Feature2': [1.0, 1.0, 5.0, 5.0]
        })
        # Classi: 2 = Benigno, 4 = Maligno
        self.y_train = pd.Series([2, 2, 4, 4])
        
    
    def test_knn_prediction_simple(self):
        
        """Test 4: 
        Predizione su punti chiaramente classificabili di classe 2 (benigno) 
        e verifica che la probabilità di classe 4 (maligno) sia bassa.
        """
        
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point vicino alla classe 2
        X_test = pd.DataFrame([[1.2, 1.2]], columns=['Feature1', 'Feature2'])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 2, "Il punto dovrebbe essere classificato come classe 2")
        self.assertLess(y_proba[0], 0.5, "La probabilità di classe 4 dovrebbe essere bassa")
        
        print(f"OK Test KNN Prediction Simple superato: predetto={y_pred[0]}, prob={y_proba[0]:.2f}")
        
    
    def test_knn_prediction_opposite_class(self):
        
        """
        Test 5: 
        Predizione su punti chiaramente classificabili di classe 4 (maligno) 
        e verifica che la probabilità di classe 4 sia alta.
        """
        
        knn = KNNClassifier(k=3, p=2)
        knn.fit(self.x_train, self.y_train)
        
        # Test point vicino alla classe 4
        X_test = pd.DataFrame([[8.2, 8.2]], columns=['Feature1', 'Feature2'])
        y_proba, y_pred = knn.predict(X_test, pos_label=4)
        
        self.assertEqual(y_pred[0], 4, "Il punto dovrebbe essere classificato come classe 4")
        self.assertGreater(y_proba[0], 0.5, "La probabilità di classe 4 dovrebbe essere alta")
        
        print(f"OK Test KNN Opposite Class superato: predetto={y_pred[0]}, prob={y_proba[0]:.2f}")
        
    
    def test_knn_with_different_k(self):
        
        """
        Test 6: 
        Verifica che diversi valori di k funzionino
        """
        
        for k in [1, 3, 5]:
            knn = KNNClassifier(k=k, p=2)
            knn.fit(self.x_train, self.y_train)
            
            X_test = pd.DataFrame([[1.5, 1.5], [8.5, 8.5]], columns=['Feature1', 'Feature2'])
            y_proba, y_pred = knn.predict(X_test, pos_label=4)
            
            self.assertEqual(len(y_pred), 2, f"Dovrebbero esserci 2 predizioni per k={k}")
            
            print(f"OK Test KNN con k={k} superato: {y_pred}")


    def test_initialization(self):
        
        """
        Test 7: 
        Verifica che i parametri (incluso il seed) vengano salvati.
        """
        
        knn = KNNClassifier(k=3, p=2, seed=123)
        self.assertEqual(knn.k, 3)
        self.assertEqual(knn.p, 2)
        self.assertEqual(knn.seed, 123, "Il seed non è stato salvato correttamente nell'init")
        
        print("OK Test Initialization superato: parametri salvati correttamente")
        

    def test_reproducibility_tie_break(self):
        
        """
        Test 8: 
        Verifica che il Tie-Break (pareggio) sia deterministico grazie al SEED.
        Senza il seed, questo test fallirebbe.
        """
        
        
        # Creiamo un training set simmetrico per forzare un pareggio
        X_train_tie = pd.DataFrame([[1, 1], [5, 5]], columns=['Feature1', 'Feature2']) # Classe 2 e Classe 4
        y_train_tie = pd.Series([2, 4])
        
        # Punto esattamente nel mezzo (3, 3) -> Distanza uguale
        X_test_tie = pd.DataFrame([[3, 3]], columns=['Feature1', 'Feature2'])
        
        # Testiamo entrambe le metriche
        for p_val in [1, 2]:
            with self.subTest(p=p_val):
                # Istanza A
                knn_a = KNNClassifier(k=2, p=p_val, seed=42)
                knn_a.fit(X_train_tie, y_train_tie)
                y_proba, pred_a = knn_a.predict(X_test_tie)
                
                # Istanza B
                knn_b = KNNClassifier(k=2, p=p_val, seed=42)
                knn_b.fit(X_train_tie, y_train_tie)
                y_proba, pred_b = knn_b.predict(X_test_tie)
                
                self.assertEqual(pred_a[0], pred_b[0], 
                                 f"Fallito con p={p_val}: il seed non ha garantito determinismo.")
        
        print("OK Test Reproducibility Tie-Break superato: risultati identici con lo stesso seed per entrambe le metriche")
        

    def test_output_format_consistency(self):
        
        """
        Test 9:
        Verifica che il metodo predict restituisca sempre una tupla (probabilità, predizioni).
        Controlliamo diverse combinazioni di k (vicini) e p (distanza).
        """
        
        X_test = pd.DataFrame([[2.0, 2.0], [4.0, 4.0]], columns=['Feature1', 'Feature2'])
        
        # Variamo sia k (vicini) che p (distanza) per robustezza
        configs = [
            {'k': 1, 'p': 1}, # 1 vicino, Distanza Manhattan
            {'k': 1, 'p': 2}, # 1 vicino, Distanza Euclidea
            {'k': 3, 'p': 1}, # 3 vicini, Distanza Manhattan
            {'k': 3, 'p': 2}  # 3 vicini, Distanza Euclidea
        ]

        for config in configs:
            k, p = config['k'], config['p']
            
            with self.subTest(k=k, p=p):
                knn = KNNClassifier(k=k, p=p)
                knn.fit(self.x_train, self.y_train)
                
                output = knn.predict(X_test)
                
                # 1. Deve essere una tupla di lunghezza 2
                self.assertIsInstance(output, tuple, f"L'output non è una tupla con k={k}, p={p}")
                self.assertEqual(len(output), 2, f"Lunghezza tupla errata con k={k}, p={p}")
                
                probs, preds = output
                
                # 2. Devono essere numpy array
                self.assertIsInstance(probs, np.ndarray, "Le probabilità devono essere array numpy")
                self.assertIsInstance(preds, np.ndarray, "Le predizioni devono essere array numpy")
                
                # 3. Le probabilità devono essere float tra 0 e 1
                self.assertTrue(np.all((probs >= 0) & (probs <= 1)), "Probabilità fuori range [0,1]")
                
                # 4. Le predizioni devono essere della classe corretta (2 o 4)
                self.assertTrue(np.all(np.isin(preds, [2, 4])), "Le predizioni contengono classi sconosciute")
        
        print("OK Test Output Format Consistency superato: formato corretto per tutte le configurazioni")
        

if __name__ == '__main__':
    unittest.main()