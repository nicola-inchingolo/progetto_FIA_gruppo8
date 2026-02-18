from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_data_cleaning import run_cleaning
from op4_handle_missing_values import run_handling_missing_value
from op5_feature_selection import run_feature_selection
from op6_outlier_remover import run_remove_outlier
import pandas as pd

"""Executes the full data preprocessing pipeline.

    This function coordinates the sequential execution of four main operations:
    1. Read File: Detects the file format, converts it to CSV (if needed), and loads it into a DataFrame.
    2. Data Evaluation: Analyzes the raw structure and identifies null/object columns.
    3. Data Cleaning: Converts object-type columns (with commas) to numeric floats.
    4. Missing Value Handling: Imputes missing values using the mean of each column.
    5. Feature Selection: Removes redundant features based on correlation analysis.
    6. Outlier Removal: Removes anomalous records using Interquartile Range (IQR).

    Args:
        file_path (str): The relative or absolute path to the input CSV file. 
            Defaults to 'data/dati_fia.csv'.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame, cleaned and filtered, 
            ready for model development.
"""

def main() -> pd.DataFrame:

    user_input = input('Enter the path (leave empty for default): ')
    file_path = user_input if user_input else 'data/dati_fia.csv'
        
    print("\n----- STARTING PIPELINE -----")

    # Operation 1: Read File
    uploaded_data = run_load_and_convert_to_csv(file_path)

    # Operation 2: Evaluation
    evaluated_data = run_evaluation(
        uploaded_data.df_output
        )

    # Operation 3: Data Cleaning (Conversion from string to float)
    cleaned_data = run_cleaning(
        uploaded_data.df_output, 
        )

    # Operation 4: Handle Missing Values (Mean Imputation)
    data_without_missing_values = run_handling_missing_value(
        cleaned_data.df_output,
    )

    # Operation 5: Feature Selection (Correlation Heatmap and Redundancy removal)
    selected_features = run_feature_selection(
        data_without_missing_values.df_output,
    )

    print("Final Dataset ready:")
    final_dataset = selected_features.df_output
    print(final_dataset.head)

    print("\n----- PIPELINE COMPLETED SUCCESSFULLY -----")
    
    return final_dataset
    
if __name__ == "__main__":
    main()