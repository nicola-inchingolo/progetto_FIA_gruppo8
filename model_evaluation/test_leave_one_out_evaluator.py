import unittest
import pandas as pd
import numpy as np
from model_evaluation.leave_one_out_evaluator import LeaveOneOutEvaluator

class TestLeaveOneOutEvaluator(unittest.TestCase):

    def setUp(self):
        """defining needed variables"""
        self.df = pd.DataFrame({
            "ID": [1, 2, 3, 4],
            "Sample code number": [10, 11, 12, 13],
            "Feature1": [1.0, 2.0, 3.0, 4.0],
            "Feature2": [0.1, 0.2, 0.3, 0.4],
            "Class": [2, 4, 2, 4]
        })
        self.metrics = np.array([1])  # only accuracy for simplicity

    def test_init_valid(self):
        """Test that the evaluator initializes correctly."""
        ev = LeaveOneOutEvaluator(self.df, self.metrics, 2)
        self.assertEqual(ev.dataset.shape[0], 4)
        self.assertTrue(np.array_equal(ev.metrics, self.metrics))

    def test_split_dataset_basic(self):
        """Test leave-one-out split for first sample."""
        ev = LeaveOneOutEvaluator(self.df, self.metrics, 2)
        x_train, x_test, y_train, y_test = ev.split_dataset_with_strategy(0)

        self.assertEqual(len(x_train), 3)
        self.assertEqual(len(x_test), 1)

        self.assertNotIn("Class", x_train.columns)
        self.assertNotIn("Class", x_test.columns)

        self.assertEqual(y_test.iloc[0], 2)

    def test_split_dataset_out_of_range(self):
        """Test that index out of range raises ValueError."""
        ev = LeaveOneOutEvaluator(self.df, self.metrics, 2)
        with self.assertRaises(ValueError):
            ev.split_dataset_with_strategy(-1)
        with self.assertRaises(ValueError):
            ev.split_dataset_with_strategy(10)

    def test_split_dataset_missing_columns(self):
        """Test that missing required columns raises KeyError."""
        df_missing = self.df.drop(columns=["Class"])
        ev = LeaveOneOutEvaluator(df_missing, self.metrics, 2)
        with self.assertRaises(KeyError):
            ev.split_dataset_with_strategy(0)

    def test_leave_one_out_evaluation(self):
        features = np.array([6])

        evaluation = LeaveOneOutEvaluator(self.df, features, 2)

        evaluation.evaluate()

        self.assertTrue(True)  # if the code reaches this line, the evaluation has run with no problems
