import unittest
import pandas as pd
import numpy as np
from model_evaluation.evaluator import Evaluator


"""MockEvaluator used to emulate an evaluator and simuling the abstract class methods"""
class MockEvaluator(Evaluator):
    def evaluate(self):
        return "ok"

"""class used to test the abstract class Evaluator"""
class TestEvaluator(unittest.TestCase):

    """defining the variables needed"""
    def setUp(self):
        self.df = pd.DataFrame({
            "feature": [1, 2, 3, 4]
        })

        self.valid_metrics = np.array([1])  # Accuracy

        self.y_test = pd.Series([4, 4, 2, 2])
        self.y_pred = pd.Series([4, 2, 2, 2])
        self.y_score = pd.Series([0.9, 0.7, 0.4, 0.1])

        self.ev = MockEvaluator(self.df, self.valid_metrics, 2, 3)

    """
    Test that the Evaluator is correctly instantiated
    when valid parameters are provided.
    """
    def test_init_valid(self):
        ev = MockEvaluator(self.df, self.valid_metrics, 2, 3)
        self.assertIsNotNone(ev)

    """
    Test that a TypeError is raised if the dataset
    is not a pandas DataFrame.
    """
    def test_init_dataset_not_dataframe(self):
        with self.assertRaises(TypeError):
            MockEvaluator([1, 2, 3], self.valid_metrics, 2, 3)

    """
    Test that a ValueError is raised if the dataset is empty.
    """
    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            MockEvaluator(pd.DataFrame(), self.valid_metrics,2, 3)

    """
    Test that a TypeError is raised if metrics are not
    provided as a numpy array.
    """
    def test_init_metrics_wrong_type(self):
        with self.assertRaises(TypeError):
            MockEvaluator(self.df, "metrics",2, 3)

    """
    Test that a ValueError is raised if the metrics array is empty.
    """
    def test_init_empty_metrics(self):
        with self.assertRaises(ValueError):
            MockEvaluator(self.df, [], 2, 3)

    """
    Test correct computation of metrics and confusion matrix
    when valid inputs are provided.
    """
    def test_calculate_metrics_valid(self):
        metrics, cm = self.ev.calculate_metrics(self.y_test, self.y_pred)

        self.assertIsInstance(metrics, dict)
        self.assertEqual(cm.shape, (2, 2))

    """
    Test that a ValueError is raised when y_test is empty.
    """
    def test_calculate_metrics_empty_y_test(self):
        with self.assertRaises(ValueError):
            self.ev.calculate_metrics(pd.Series([]), self.y_pred)

    """
    Test that a ValueError is raised when y_pred is empty.
    """
    def test_calculate_metrics_empty_y_pred(self):
        with self.assertRaises(ValueError):
            self.ev.calculate_metrics(self.y_test, pd.Series([]))

    """
    Test that a ValueError is raised when y_test and y_pred
    have different lengths.
    """
    def test_calculate_metrics_different_lengths(self):
        with self.assertRaises(ValueError):
            self.ev.calculate_metrics(self.y_test, pd.Series([4, 2]))

    """
    Test that a ValueError is raised when an invalid
    metric index is provided.
    """
    def test_calculate_metrics_invalid_metric_index(self):
        ev = MockEvaluator(self.df, np.array([9]),2, 3)
        with self.assertRaises(ValueError):
            ev.calculate_metrics(self.y_test, self.y_pred)

    """
    Test correct AUC computation when valid inputs are provided.
    """
    def test_calculate_auc_valid(self):
        auc, tpr, fpr = self.ev.calculate_auc(self.y_test, self.y_score)

        self.assertTrue(0 <= auc <= 1)
        self.assertEqual(len(tpr), len(fpr))

    """
    Test that a ValueError is raised when y_test is empty
    during AUC calculation.
    """
    def test_calculate_auc_empty_y_test(self):
        with self.assertRaises(ValueError):
            self.ev.calculate_auc(pd.Series([]), self.y_score)

    """
    Test that a ValueError is raised when y_score is empty
    during AUC calculation.
    """
    def test_calculate_auc_empty_y_score(self):
        with self.assertRaises(ValueError):
            self.ev.calculate_auc(self.y_test, pd.Series([]))

    """
    Test that a ValueError is raised when y_score contains
    values outside the valid range [0, 1].
    """
    def test_calculate_auc_score_out_of_range(self):
        y_score_invalid = pd.Series([1.2, -0.3, 0.5, 0.1])
        with self.assertRaises(ValueError):
            self.ev.calculate_auc(self.y_test, y_score_invalid)