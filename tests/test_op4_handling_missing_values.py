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
        """
        self.df = pd.DataFrame({
            # 'classtype_v1' is required. Row index 3 has a null here, so it will be dropped.
            'classtype_v1': [1, 1, 1, np.nan], 
            
            # 'age': 20, 30, NaN, 40. 
            # After dropping row 3 (where classtype is nan), we have [20, 30, NaN]. 
            # The mean of (20, 30) is 25.
            'age': [20.0, 30.0, np.nan, 40.0],  
            
            # 'salary': 100, NaN, 200, 300. 
            # After dropping row 3, we have [100, NaN, 200]. 
            # The mean of (100, 200) is 150.
            'salary': [100.0, np.nan, 200.0, 300.0] 
        })

    @patch('pandas.Series.plot')
    @patch('matplotlib.pyplot.savefig') # Prevent saving files to disk
    @patch('matplotlib.pyplot.figure')  # Prevent creating figure windows
    @patch('matplotlib.pyplot.close')   # Prevent closing errors
    @patch('random.choice')             # Control the random column selection
    def test_imputation_logic(self, mock_choice, mock_close, mock_figure, mock_savefig, mock_pandas_plot):
        """
        Verifies that missing values are correctly replaced by the mean
        after the rows with null 'classtype_v1' have been dropped.
        """
        # Force random.choice to pick 'age' to ensure plotting logic works without error
        mock_choice.return_value = 'age'
        
        # Suppress print output
        with patch('sys.stdout', new=io.StringIO()):
            # Removed the second argument (columns_with_nulls)
            result = run_handling_missing_value(self.df.copy())

        # Verify rows with null 'classtype_v1' are dropped
        self.assertEqual(len(result.df_output), 3)

        # Verify Mean Imputation on 'age'
        filled_age_values = result.df_output['age'].values
        self.assertIn(25.0, filled_age_values)
        self.assertFalse(np.isnan(filled_age_values).any())

        # Verify Mean Imputation on 'salary'
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
            run_handling_missing_value(self.df.copy())
        
        # Verify savefig was called exactly twice (before and after imputation)
        self.assertEqual(mock_savefig.call_count, 2)
        
        # Verify it tried to save to the correct paths
        # We check the arguments passed to the first call
        args, _ = mock_savefig.call_args_list[0]
        # Depending on OS, path might contain backslash or slash, checking for filename part is safer
        self.assertIn('hist_col_before_imputation.png', str(args[0]))

    def test_missing_mandatory_column(self):
        """
        Verifies behavior if 'classtype_v1' is missing from the DataFrame.
        The function tries to dropna subset=['classtype_v1'], so this should raise KeyError.
        """
        df_broken = pd.DataFrame({'age': [20, 30, np.nan]})
        
        with self.assertRaises(KeyError):
             run_handling_missing_value(df_broken)

    @patch('pandas.Series.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('random.choice')
    def test_return_structure(self, mock_choice, mock_close, mock_figure, mock_savefig, mock_pandas_plot):
        """
        Verifies the output object type.
        """
        mock_choice.return_value = 'age'
        with patch('sys.stdout', new=io.StringIO()):
            result = run_handling_missing_value(self.df.copy())
            
        self.assertIsInstance(result, ImputationOutputs)
        self.assertIsInstance(result.df_output, pd.DataFrame)
        # Verify no nulls remain in the output
        self.assertEqual(result.df_output.isnull().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)