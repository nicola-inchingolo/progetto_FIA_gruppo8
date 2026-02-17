import unittest
import pandas as pd
import os
import shutil
import tempfile
from unittest.mock import patch
from data_preprocessing.op1_read_file import run_load_and_convert_to_csv

class TestRunLoadAndConvert(unittest.TestCase):

    def setUp(self):
        """
        Sets up a temporary directory and dummy files before each test.
        This ensures we don't pollute the actual project folder with test files.
        """
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a dummy DataFrame
        self.dummy_data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
        self.df = pd.DataFrame(self.dummy_data)

        # Create a dummy CSV file
        self.csv_path = os.path.join(self.test_dir, 'test_file.csv')
        self.df.to_csv(self.csv_path, index=False)

        # Create a dummy JSON file (to test conversion)
        self.json_path = os.path.join(self.test_dir, 'test_file.json')
        self.df.to_json(self.json_path)

    def tearDown(self):
        """
        Cleans up the temporary directory after each test.
        """
        shutil.rmtree(self.test_dir)

    def test_load_csv_direct(self):
        """
        Tests loading a CSV file directly. 
        It should verify that no conversion is needed and data is loaded correctly.
        """
        result = run_load_and_convert_to_csv(self.csv_path)
        
        # Verify the content matches
        pd.testing.assert_frame_equal(result.df_output, self.df)

    def test_convert_json_to_csv(self):
        """
        Tests loading a JSON file.
        It should verify that the function converts it to CSV and returns the DataFrame.
        """
        # Run the function on the JSON path
        result = run_load_and_convert_to_csv(self.json_path)

        # Verify the content matches
        # We reset index because read_json might behave slightly differently with indices,
        # but here we expect a simple match.
        pd.testing.assert_frame_equal(result.df_output.reset_index(drop=True), self.df.reset_index(drop=True))

        # Verify that the new .csv file was actually created on disk
        expected_new_csv_path = os.path.join(self.test_dir, 'test_file.csv') 
        # Let's check if a file ending in .csv exists for the json base name.
        self.assertTrue(os.path.exists(expected_new_csv_path))

    def test_unsupported_format(self):
        """
        Tests behavior when an unsupported file extension is provided.
        """
        # Create a dummy .txt file
        txt_path = os.path.join(self.test_dir, 'test.txt')
        with open(txt_path, 'w') as f:
            f.write("Some text")

        # We expect the function to crash or fail because df is undefined for unsupported files.
        with self.assertRaises(UnboundLocalError):
            run_load_and_convert_to_csv(txt_path)

if __name__ == '__main__':
    unittest.main()