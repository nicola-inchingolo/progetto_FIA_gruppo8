import unittest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from data_preprocessing.op3_data_cleaning import run_cleaning, CleaningOutputs

class TestRunCleaning(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up the test data before each test method.
        Creates a DataFrame with messy strings simulating typical data issues.
        """
        self.df_messy = pd.DataFrame({
            'price': ['10,50', '20.50', '1000', 'NotAvailable'], # Mixed formats + text
            'weight': ['1,2', '5,5', '0,5', np.nan],            # European decimal format
            'category': ['A', 'B', 'C', 'D']                    # Column that should NOT be touched
        })

    def test_comma_replacement(self):
        """
        Verifies that commas are correctly replaced by dots and converted to floats.
        """
        # We only target the 'price' column
        target_cols = pd.Index(['price'])
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy(), target_cols)
        
        # Check specific values
        cleaned_price = result.df_output['price']
        
        # '10,50' -> 10.5
        self.assertEqual(cleaned_price[0], 10.5)
        # '20.50' -> 20.5 (should remain correct)
        self.assertEqual(cleaned_price[1], 20.5)
        # '1000' -> 1000.0
        self.assertEqual(cleaned_price[2], 1000.0)

    def test_non_numeric_conversion(self):
        """
        Verifies that non-numeric strings are coerced to NaN.
        """
        target_cols = pd.Index(['price'])
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy(), target_cols)
            
        # The 4th value was 'NotAvailable', should be NaN
        self.assertTrue(pd.isna(result.df_output['price'][3]))

    def test_multiple_columns(self):
        """
        Verifies that the function can handle multiple columns at once.
        """
        target_cols = pd.Index(['price', 'weight'])
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy(), target_cols)
            
        # Check both columns are now float type
        self.assertTrue(pd.api.types.is_float_dtype(result.df_output['price']))
        self.assertTrue(pd.api.types.is_float_dtype(result.df_output['weight']))
        
        # Check a value in the second column ('1,2' -> 1.2)
        self.assertEqual(result.df_output['weight'][0], 1.2)

    def test_ignore_unlisted_columns(self):
        """
        Verifies that columns not in the object_columns list are left unchanged.
        """
        target_cols = pd.Index(['price'])
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy(), target_cols)
            
        # 'category' was not in target_cols, so it should remain 'object' (string)
        self.assertEqual(result.df_output['category'][0], 'A')
        # Ensure it wasn't converted to NaN or numbers
        self.assertFalse(pd.api.types.is_numeric_dtype(result.df_output['category']))

    def test_error_resilience(self):
        """
        Verifies that the function handles errors without crashing, thanks to the try-except block.
        """
        # 'non_existent' column does not exist in df
        target_cols = pd.Index(['price', 'non_existent'])
        
        # This should not raise a KeyError because the exception is catched
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            result = run_cleaning(self.df_messy.copy(), target_cols)
            
            # Verify 'price' was still processed correctly
            self.assertEqual(result.df_output['price'][0], 10.5)

    def test_return_structure(self):
        """
        Verifies the output is strictly the expected NamedTuple.
        """
        target_cols = pd.Index(['price'])
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy(), target_cols)
            
        self.assertIsInstance(result, CleaningOutputs)
        self.assertIsInstance(result.df_output, pd.DataFrame)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)