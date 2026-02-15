import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from model_evaluation.k_fold_evaluator import kFoldEvaluator
from model_development import Knn_Algorithm


class TestKFoldEvaluator(unittest.TestCase):
    """defining needed variables"""

    def setUp(self):
        self.df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Sample code number": [10, 11, 12, 13, 14],
            "Feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "classtype_v1": [2, 4, 2, 4, 2]
        })

    """Test that kFoldEvaluator initializes correctly with valid parameters."""

    def test_init_valid(self):
        ev = kFoldEvaluator(self.df, np.array([1]), 3, 1, 3)
        self.assertEqual(ev.K_tests, 3)
        self.assertEqual(ev.seed, 42)

    """Test that non-integer K_tests raises TypeError."""

    def test_init_invalid_K_type(self):
        with self.assertRaises(TypeError):
            kFoldEvaluator(self.df, np.array([1]), 2, 2, 2.5)

    """Test that K_tests outside valid range raises ValueError."""

    def test_init_invalid_K_value(self):
        with self.assertRaises(ValueError):
            kFoldEvaluator(self.df, np.array([1]), 1, 2, 1)  # less than 2
        with self.assertRaises(ValueError):
            kFoldEvaluator(self.df, np.array([1]), 10, 2, 10)  # more than dataset size

    """Test that split_dataset_with_strategy returns correct shapes."""

    def test_split_dataset_valid(self):
        ev = kFoldEvaluator(self.df, np.array([1, 2]), 3, 2, 3)
        folds = np.array_split(ev.dataset, ev.K_tests)

        x_train, x_test, y_train, y_test = ev.split_dataset_with_strategy(folds, 0)

        #excluding the target ones
        expected_num_features = self.df.shape[1] - 1

        self.assertEqual(x_train.shape[1], expected_num_features)
        self.assertEqual(x_test.shape[1], expected_num_features)

        self.assertEqual(sorted(y_train.unique().tolist()), [2, 4])

    """Test that missing required columns raises KeyError."""

    def test_split_dataset_missing_columns(self):
        df = self.df.drop(columns=["classtype_v1"])
        ev = kFoldEvaluator(df, np.array([1]), 2, 2, 2)
        folds = np.array_split(ev.dataset, ev.K_tests)
        with self.assertRaises(KeyError):
            ev.split_dataset_with_strategy(folds, 0)

    """Test that an empty fold raises ValueError."""

    def test_split_dataset_empty_fold(self):
        ev = kFoldEvaluator(self.df, np.array([1]), 5, 2, 5)
        folds = [pd.DataFrame()] * 5
        with self.assertRaises(ValueError):
            ev.split_dataset_with_strategy(folds, 0)

    """methods used to test the flow of the k_fold_evaluation"""

    def test_k_fold_evaluation(self):
        K_test = 4
        features = np.array([6])
        p = 2
        k = 3

        evaluation = kFoldEvaluator(self.df, features, p, k, K_test)

        evaluation.evaluate()

        self.assertTrue(True)  # if the code reaches this line, the evaluation has run with no problems

    def test_k_fold_evaluation_with_K_tests_out_of_range(self):
        K_test = 65
        features = np.array([6])
        p = 2
        k=3

        with self.assertRaises(ValueError):
            evaluation = kFoldEvaluator(self.df, features, p,k , K_test)  # Check prediction length
            evaluation.evaluate()
