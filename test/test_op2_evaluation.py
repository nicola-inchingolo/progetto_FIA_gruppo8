import unittest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from data_preprocessing.op2_data_evaluation import run_evaluation, EvaluationOutputs

class TestRunEvaluation(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up the test data before each test method.
        Creates a DataFrame with mixed data types (numeric, object) and missing values.
        """
        self.df_mixed = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Anna', 'Bob', 'Charlie', np.nan],  # Object column with nulls
            'age': [25, np.nan, 30, 22],                  # Numeric column with nulls
            'city': ['Rome', 'Milan', 'Naples', 'Turin']  # Object column without nulls
        })

    def test_return_structure(self):
        """
        Verifies that the function returns the correct data structure and objects.
        Checks if the output matches the expected NamedTuple fields.
        """
        # Suppress stdout to keep the test console output clean
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(self.df_mixed)
        
        # Verify that the returned dataframe matches the input
        pd.testing.assert_frame_equal(result.df_output, self.df_mixed)
        
        # Verify that columns with nulls were detected
        self.assertFalse(result.columns_with_nulls_output.empty)
        
        # Verify that object columns were identified correctly
        self.assertIn('name', result.object_columns_output)
        self.assertIn('city', result.object_columns_output)
        # Ensure numeric columns are not in the object list
        self.assertNotIn('age', result.object_columns_output)

    def test_missing_values_detection(self):
        """
        Verifies that missing values are correctly identified and counted.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(self.df_mixed)
            
        # We expect 'name' and 'age' to be in the nulls list
        null_cols = result.columns_with_nulls_output
        self.assertIn('name', null_cols)
        self.assertIn('age', null_cols)
        self.assertNotIn('city', null_cols)
        
        # Verify the exact count of nulls (1 for name, 1 for age)
        self.assertEqual(null_cols['name'], 1)
        self.assertEqual(null_cols['age'], 1)

    def test_no_missing_values(self):
        """
        Tests the behavior when the dataset has no missing values.
        """
        df_clean = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(df_clean)
            
        # The output series for nulls should be empty
        self.assertTrue(result.columns_with_nulls_output.empty)

    def test_no_object_columns(self):
        """
        Tests the behavior when the dataset contains only numeric columns.
        """
        df_numeric = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3]
        })
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(df_numeric)
            
        # The index for object columns should be empty
        self.assertTrue(result.object_columns_output.empty)

    def test_output_integrity(self):
        """
        Verifies that the input df is not mutated during the process
        and the output is reliable.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(self.df_mixed)
            
        # Ensure the output df is exactly the same as the input df
        pd.testing.assert_frame_equal(result.df_output, self.df_mixed)

if __name__ == '__main__':
    # Running the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)