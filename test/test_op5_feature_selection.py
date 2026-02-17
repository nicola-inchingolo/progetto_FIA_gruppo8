import unittest
import pandas as pd
import numpy as np
import io
import os
from unittest.mock import patch, MagicMock
from data_preprocessing.op5_feature_selection import run_feature_selection, FeatureSelectionOutputs

class TestFeatureSelection(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up the test data.
        We create a DataFrame with specific correlation patterns.
        """
        self.df = pd.DataFrame({
            # Feature A: Base data
            'feat_A': [1, 2, 3, 4, 5],
            
            # Feature B: Perfectly correlated with A (Corr = 1.0)
            # This should be dropped.
            'feat_B': [2, 4, 6, 8, 10], 
            
            # Feature C: Not correlated (random-ish)
            # This should remain.
            'feat_C': [5, 1, 4, 2, 3],
            
            # Feature D: High negative correlation with A (Corr = -1.0)
            # Since code uses .abs(), this should be dropped too.
            'feat_D': [-1, -2, -3, -4, -5],

            # Target Column: The function explicitly looks for 'classtype_v1'
            # This must be preserved.
            'classtype_v1': [0, 1, 0, 1, 0]
        })

    @patch('matplotlib.pyplot.savefig') # Mock saving file
    @patch('matplotlib.pyplot.show')    # Mock showing plot
    @patch('matplotlib.pyplot.figure')  # Mock creating figure
    @patch('seaborn.heatmap')           # Mock seaborn heatmap
    @patch('os.makedirs')               # Mock directory creation
    def test_high_correlation_removal(self, mock_makedirs, mock_sns, mock_fig, mock_show, mock_savefig):
        """
        Verifies that columns with correlation > 0.8 are removed.
        """
        # Suppress print output
        with patch('sys.stdout', new=io.StringIO()):
            result = run_feature_selection(self.df.copy())
            
        df_out = result.df_output
        columns = df_out.columns

        # 'feat_A' should exist
        self.assertIn('feat_A', columns)
        
        # 'feat_B' should be dropped (Corr(A,B) = 1.0 > 0.8)
        self.assertNotIn('feat_B', columns)

        # 'feat_C' should exist (Low correlation)
        self.assertIn('feat_C', columns)

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.figure')
    def test_negative_correlation_removal(self, mock_fig, mock_sns, mock_save):
        """
        Verifies that high negative correlation is also handled 
        (because the code uses .abs()).
        """
        with patch('sys.stdout', new=io.StringIO()):
            result = run_feature_selection(self.df.copy())
            
        df_out = result.df_output
        
        # 'feat_D' is -1.0 correlated with 'feat_A'. 
        # Since abs(-1.0) = 1.0 > 0.8, it should be dropped.
        self.assertNotIn('feat_D', df_out.columns)

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.figure')
    def test_target_preservation(self, mock_fig, mock_sns, mock_save):
        """
        Ensures the target column 'classtype_v1' is never dropped,
        regardless of its correlation with features.
        """
        # Create a df where a feature is identical to the target
        df_risky = pd.DataFrame({
            'feat_X': [0, 1, 0, 1],
            'classtype_v1': [0, 1, 0, 1]
        })

        with patch('sys.stdout', new=io.StringIO()):
            result = run_feature_selection(df_risky)
            
        # The target must remain because it is excluded from the correlation matrix calculation
        self.assertIn('classtype_v1', result.df_output.columns)
        # feat_X should also remain because it has nothing else to correlate with in the matrix
        self.assertIn('feat_X', result.df_output.columns)

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.figure')
    def test_no_redundancy(self, mock_fig, mock_sns, mock_save):
        """
        Verifies that if no correlations exceed 0.8, no columns are dropped.
        """
        df_clean = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 2, 5, 1], # Low correlation
            'classtype_v1': [0, 0, 1, 1]
        })

        with patch('sys.stdout', new=io.StringIO()):
            result = run_feature_selection(df_clean)

        # Output shape should match input shape
        pd.testing.assert_frame_equal(result.df_output, df_clean)

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.figure')
    def test_plot_generation(self, mock_fig, mock_sns, mock_savefig):
        """
        Verifies that the code attempts to generate and save the heatmap.
        """
        with patch('sys.stdout', new=io.StringIO()):
            run_feature_selection(self.df.copy())
            
        # Check if heatmap was called
        self.assertTrue(mock_sns.called)
        
        # Check if savefig was called
        self.assertTrue(mock_savefig.called)
        
        # Check arguments for savefig
        args, _ = mock_savefig.call_args
        self.assertIn('correlation_heatmap.png', args[0])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)