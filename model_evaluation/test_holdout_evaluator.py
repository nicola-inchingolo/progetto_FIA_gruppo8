import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from model_evaluation.holdout_evaluator import holdout_evaluator
from model_development import Knn_Algorithm


class TestHoldoutEvaluator(unittest.TestCase):

    """defining needed attributes"""
    def setUp(self):
        self.df = pd.DataFrame({
            "ID": [1, 2, 3, 4],
            "Sample code number": [10, 11, 12, 13],
            "Feature1": [1.0, 2.0, 3.0, 4.0],
            "Feature2": [0.1, 0.2, 0.3, 0.4],
            "Class": [2, 4, 2, 4]
        })
        self.metrics = np.array([1])

    """Test that holdout_evaluator initializes correctly with valid parameters."""
    def test_init_valid(self):
        ev = holdout_evaluator(self.df, np.array([1]), 2, 2, 0.5)
        self.assertEqual(ev.train_percentage, 0.5)
        self.assertEqual(ev.seed, 42)

    """Test that non-float train_percentage raises TypeError."""
    def test_init_invalid_percentage_type(self):
        with self.assertRaises(TypeError):
            holdout_evaluator(self.df, np.array([1]), 2, 2, "0.5")

    """Test that train_percentage outside (0,1) raises ValueError."""
    def test_init_invalid_percentage_value(self):
        with self.assertRaises(ValueError):
            holdout_evaluator(self.df, np.array([1]), 1, 2, 1.2)

    """Test that the dataset splits into correct training/test sizes."""
    def test_split_dataset_sizes(self):
        ev = holdout_evaluator(self.df, np.array([1]), 2, 2, 0.6)
        x_train, x_test, y_train, y_test = ev.split_dataset_with_strategy()
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(x_test), 2)

    """Test that split_dataset removes 'Class' column from feature sets."""
    def test_split_dataset_removes_class_column(self):
        ev = holdout_evaluator(self.df, np.array([1]), 2, 2, 0.5)
        x_train, _, _, _ = ev.split_dataset_with_strategy()
        self.assertNotIn("Class", x_train.columns)

    """Test that missing required columns raises KeyError."""
    def test_split_dataset_missing_columns(self):
        df = self.df.drop(columns=["Class"])
        ev = holdout_evaluator(df, np.array([1]), 2, 2, 0.5)
        with self.assertRaises(KeyError):
            ev.split_dataset_with_strategy()

    """methods used to test the flow of the holdout_evalutaion"""
    def test_holdout_evaluation(self):

        train_percentage = 0.7
        features = np.array([6])

        evaluation = holdout_evaluator(self.df,features,  2, 3, train_percentage)

        x_train, x_test, y_train, y_test = evaluation.split_dataset_with_strategy()
        self.assertEqual(len(x_train), int(len(
            features) * train_percentage))
        self.assertEqual(len(x_test), len(features) - len(x_train))
        self.assertEqual(len(y_train), len(x_train))
        self.assertEqual(len(y_test), len(x_test))

        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_test), 0)

        model = Knn_Algorithm.KNNClassifier(k=evaluation.k_neighbours, p=evaluation.distance_strategy)
        model.fit(x_train, y_train)
        y_score, y_pred = model.predict(x_test, pos_label=4)

        self.assertEqual(len(y_pred), len(y_test))

        evaluation.evaluate()

        self.assertTrue(True)  # if the code reaches this line, the evaluation has run with no problems

