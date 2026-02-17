import unittest
import pandas as pd
import numpy as np
import io
import os
from unittest.mock import patch, MagicMock
from data_preprocessing.op4_handle_missing_values import run_handling_missing_value, ImputationOutputs

class TestRunHandlingMissingValue(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up the test data before each test method.
        Creates a DataFrame with specific missing value patterns.
        """
        self.df = pd.DataFrame({
            # 'classtype_v1' is required because the function explicitly drops nulls from it
            'classtype_v1': [1, 1, 1, np.nan], 
            
            # 'age': 20, 30, NaN. After dropping row 3 (index 3), 
            # we have [20, 30, NaN]. The mean of (20, 30) is 25.
            'age': [20.0, 30.0, np.nan, 40.0],  
            
            # 'salary': 100, NaN, 200. After dropping row 3,
            # we have [100, NaN, 200]. The mean of (100, 200) is 150.
            'salary': [100.0, np.nan, 200.0, 300.0] 
        })

        # The function expects 'columns_with_nulls' to be a Series representing columns that need imputation.
        self.null_cols_input = pd.Series([1, 1], index=['age', 'salary'])

    @patch('pandas.Series.plot')
    @patch('matplotlib.pyplot.savefig') # Prevent saving files to disk
    @patch('matplotlib.pyplot.figure')  # Prevent creating figure windows
    @patch('matplotlib.pyplot.close')   # Prevent closing errors
    @patch('random.choice')             # Control the random column selection
    def test_imputation_logic(self, mock_choice, mock_close, mock_figure, mock_savefig, mock_pandas_plot):
        """
        Verifies that missing values are correctly replaced by the mean.
        We mock the plotting libraries to avoid file system errors.
        """
        # Force random.choice to pick 'age' so we know which column is being plotted
        mock_choice.return_value = 'age'
        
        # Suppress print output
        with patch('sys.stdout', new=io.StringIO()):
            result = run_handling_missing_value(self.df.copy(), self.null_cols_input)

        # Verify rows with null 'classtype_v1' are dropped
        # Original size 4, 1 null in classtype -> expect size 3
        self.assertEqual(len(result.df_output), 3)

        # Verify Mean Imputation on 'age'
        # Remaining valid values for age before imputation: 20, 30. Mean = 25.
        # The NaN value should be replaced by 25.
        filled_age_values = result.df_output['age'].values
        self.assertIn(25.0, filled_age_values)
        self.assertFalse(np.isnan(filled_age_values).any())

        # Verify Mean Imputation on 'salary'
        # Remaining valid values: 100, 200. Mean = 150.
        filled_salary_values = result.df_output['salary'].values
        self.assertIn(150.0, filled_salary_values)
        self.assertFalse(np.isnan(filled_salary_values).any())

    @patch('pandas.Series.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('random.choice')
    def test_plotting_calls(self, mock_choice, mock_close, mock_figure, mock_savefig, mock_pandas_plot):
        """
        Verifies that the code attempts to save the plots (histograms).
        """
        mock_choice.return_value = 'age'
        
        with patch('sys.stdout', new=io.StringIO()):
            run_handling_missing_value(self.df.copy(), self.null_cols_input)
        
        # Verify savefig was called exactly twice (before and after imputation)
        self.assertEqual(mock_savefig.call_count, 2)
        
        # Verify it tried to save to the correct paths
        # We check the arguments passed to the first call
        args, _ = mock_savefig.call_args_list[0]
        self.assertIn('hist_col_before_imputation.png', args[0])

    def test_missing_mandatory_column(self):
        """
        Verifies behavior if 'classtype_v1' is missing from the DataFrame.
        """
        df_broken = pd.DataFrame({'age': [20, 30]})
        
        with self.assertRaises(KeyError):
             run_handling_missing_value(df_broken, self.null_cols_input)

    @patch('pandas.Series.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('random.choice')
    def test_no_nulls_remaining(self, mock_choice, mock_close, mock_figure, mock_savefig, mock_pandas_plot):
        """
        Verifies the final check ensuring the output df has 0 nulls.
        """
        mock_choice.return_value = 'age'
        
        with patch('sys.stdout', new=io.StringIO()):
            result = run_handling_missing_value(self.df.copy(), self.null_cols_input)
            
        self.assertEqual(result.df_output.isnull().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)