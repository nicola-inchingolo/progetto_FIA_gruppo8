import unittest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from data_preprocessing.op6_outlier_remover import run_remove_outlier, OutlierOutputs

class TestOutlierRemoval(unittest.TestCase):

    def test_basic_outlier_removal(self):
        """
        Test if simple high/low outliers are removed within a single class.
        """
        data = {
            'classtype_v1': ['A'] * 7,
            'value': [10, 11, 10, 12, 10, 1000, -500] # 1000 and -500 are obvious outliers
        }
        df = pd.DataFrame(data)

        # Run the function
        result = run_remove_outlier(df)

        # Assertions
        # We expect 5 rows to remain (the values around 10-12)
        self.assertEqual(len(result.df_output), 5)
        
        # Ensure the extreme values are gone
        self.assertNotIn(1000, result.df_output['value'].values)
        self.assertNotIn(-500, result.df_output['value'].values)

    def test_group_specific_logic(self):
        """
        This ensures outliers are calculated per group/target.
        Value 100 is an outlier for Group 'Small', but normal for Group 'Big'.
        The function must remove 100 from 'Small' but keep it in 'Big'.
        """
        # Group 'Small': Centered around 10. 100 is a huge outlier.
        small_group = pd.DataFrame({
            'classtype_v1': ['Small'] * 10,
            'val': [10, 10, 11, 9, 10, 10, 11, 9, 10, 100]
        })

        # Group 'Big': Centered around 100. 100 is the median/normal.
        big_group = pd.DataFrame({
            'classtype_v1': ['Big'] * 10,
            'val': [100, 100, 101, 99, 100, 100, 101, 99, 100, 100]
        })

        # Combine them
        df = pd.concat([small_group, big_group], ignore_index=True)
        
        # Run function
        result = run_remove_outlier(df)
        df_out = result.df_output

        # Check Group 'Small'
        small_results = df_out[df_out['classtype_v1'] == 'Small']
        # 100 should be gone from Small
        self.assertNotIn(100, small_results['val'].values) 
        
        # Check Group 'Big'
        big_results = df_out[df_out['classtype_v1'] == 'Big']
        # 100 should still exist in Big
        self.assertIn(100, big_results['val'].values)

    def test_preserves_non_numeric_columns(self):
        """
        Ensure string/object columns are not dropped and don't cause crashes.
        """
        df = pd.DataFrame({
            'classtype_v1': ['A', 'A', 'A', 'A', 'A'],
            'numeric_val': [1, 2, 1, 2, 100], # 100 is outlier
            'description': ['keep', 'me', 'please', 'safe', 'delete_me']
        })

        result = run_remove_outlier(df)
        
        # Check strictly that the 'description' column still exists
        self.assertIn('description', result.df_output.columns)
        
        # The row with 100 should be removed, so the description 'delete_me' should also be gone
        self.assertNotIn('delete_me', result.df_output['description'].values)
        self.assertIn('keep', result.df_output['description'].values)

    def test_return_type(self):
        """
        Ensure the function returns the correct NamedTuple structure.
        """
        df = pd.DataFrame({'classtype_v1': ['A'], 'val': [1]})
        result = run_remove_outlier(df)
        
        self.assertIsInstance(result, OutlierOutputs)
        self.assertIsInstance(result.df_output, pd.DataFrame)

    def test_no_outliers(self):
        """
        Ensure that if data is clean, no rows are dropped.
        """
        df = pd.DataFrame({
            'classtype_v1': ['A'] * 5,
            'val': [10, 11, 10, 11, 10]
        })
        result = run_remove_outlier(df)
        
        # Length should be identical
        self.assertEqual(len(df), len(result.df_output))
        # df should be equal
        pd.testing.assert_frame_equal(df, result.df_output)

if __name__ == '__main__':
    unittest.main()