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
        Verifies that the function returns the correct data structure.
        Since the function only returns df_output now, we verify that specifically.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_evaluation(self.df_mixed)
        
        # Verify that the return is an instance of EvaluationOutputs
        self.assertIsInstance(result, EvaluationOutputs)

        # Verify that the returned dataframe matches the input
        pd.testing.assert_frame_equal(result.df_output, self.df_mixed)
        
        # Checks for columns_with_nulls_output and object_columns_output
        # as they are no longer part of the return signature.

    def test_missing_values_print_logic(self):
        """
        Verifies that missing values are correctly identified by checking the printed output.
        """
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            run_evaluation(self.df_mixed)
            output = fake_out.getvalue()

        # Check if the "Missing Values Analysis" section was printed
        self.assertIn("Missing Values Analysis:", output)
        
        # Check if the columns with nulls were mentioned in the output
        self.assertIn("name", output)
        self.assertIn("age", output)
        
        # Ensure columns without nulls are not listed in the missing values section
        # 'city' might appear in other sections (like unique values), 
        # so we rely on the specific message "Columns with missing values:" check roughly.
        # A simple check is to ensure the specific logic for "No missing values" is not present.
        self.assertNotIn("No missing values were found in the dataset.", output)

    def test_no_missing_values_print_logic(self):
        """
        Tests the printed output when the dataset has no missing values.
        """
        df_clean = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            run_evaluation(df_clean)
            output = fake_out.getvalue()
            
        # We expect the specific success message
        self.assertIn("No missing values were found in the dataset.", output)

    def test_object_columns_print_logic(self):
        """
        Tests that object columns are correctly detected and unique values are printed.
        """
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            run_evaluation(self.df_mixed)
            output = fake_out.getvalue()

        # Check that the analysis found the object columns
        self.assertIn("Object-type Columns Analysis:", output)
        
        # The list of object columns should be printed
        self.assertIn("name", output)
        self.assertIn("city", output)
        
        # Check that unique values are being inspected
        self.assertIn("Unique values within Object columns:", output)

    def test_no_object_columns_print_logic(self):
        """
        Tests the behavior when the dataset contains only numeric columns.
        """
        df_numeric = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3]
        })
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            run_evaluation(df_numeric)
            output = fake_out.getvalue()
            
        # We expect the specific message for no object columns
        self.assertIn("No object-type (string) columns found in the dataset.", output)

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
    unittest.main(argv=['first-arg-is-ignored'], exit=False)