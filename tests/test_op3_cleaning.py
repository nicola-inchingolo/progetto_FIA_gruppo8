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
        Creates a DataFrame with mixed issues:
        1. Columns that should be cleaned (commas, strings representing numbers).
        2. Columns that should be dropped (IDs).
        3. Columns that are text and will become NaN.
        """
        self.df_messy = pd.DataFrame({
            'Sample code number': [1001, 1002, 1003, 1004], # Should be dropped
            'price': ['10,50', '20.50', '1000', 'NotAvailable'], # Mixed formats
            'weight': ['1,2', '5,5', '0,5', np.nan],            # European decimal format
            'category': ['A', 'B', 'C', 'D']                    # Text column (will become NaN)
        })

    def test_column_dropping(self):
        """
        Verifies that specific irrelevant columns are dropped from the dataframe.
        """
        with patch('sys.stdout', new=io.StringIO()):
            # Run cleaning
            result = run_cleaning(self.df_messy.copy())
            
        # Check that 'Sample code number' is not in the columns
        self.assertNotIn('Sample code number', result.df_output.columns)
        
        # Ensure other columns remain
        self.assertIn('price', result.df_output.columns)

    def test_comma_replacement_and_conversion(self):
        """
        Verifies that commas are replaced by dots and object columns 
        are correctly converted to floats.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy())
        
        cleaned_price = result.df_output['price']
        cleaned_weight = result.df_output['weight']
        
        # Check 'price' conversion: '10,50' -> 10.5
        self.assertEqual(cleaned_price[0], 10.5)
        # '20.50' -> 20.5
        self.assertEqual(cleaned_price[1], 20.5)
        
        # Check 'weight' conversion: '1,2' -> 1.2
        self.assertEqual(cleaned_weight[0], 1.2)
        
        # Verify dtypes are now float
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_price))
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_weight))

    def test_non_numeric_coercion(self):
        """
        Verifies that non-numeric strings (like 'NotAvailable' or 'A') 
        are coerced to NaN.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy())
            
        # 'NotAvailable' in price should be NaN
        self.assertTrue(pd.isna(result.df_output['price'][3]))
        
        # The 'category' column contained only text.
        # Since the function converts all object columns, these should all be NaN.
        self.assertTrue(result.df_output['category'].isna().all())

    def test_exception_handling(self):
        """
        Verifies that the function handles unexpected errors during conversion
        without crashing.
        We mock pd.to_numeric to raise an exception for a specific column.
        """
        # We create a specific dataframe for this test
        df_test = pd.DataFrame({'col1': ['1,2'], 'col2': ['3,4']})
        
        # We patch pandas to_numeric just for this test block
        with patch('pandas.to_numeric') as mock_to_numeric:
            # Configure the mock: raise ValueError for the first call, work for the second
            mock_to_numeric.side_effect = [ValueError("Simulated Error"), 1.2]
            
            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                run_cleaning(df_test)
                output = fake_out.getvalue()
                
            # Verify that the error message was printed
            self.assertIn("Error during conversion of column", output)
            self.assertIn("Simulated Error", output)

    def test_return_structure(self):
        """
        Verifies the output is strictly the expected NamedTuple.
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_cleaning(self.df_messy.copy())
            
        self.assertIsInstance(result, CleaningOutputs)
        self.assertIsInstance(result.df_output, pd.DataFrame)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)