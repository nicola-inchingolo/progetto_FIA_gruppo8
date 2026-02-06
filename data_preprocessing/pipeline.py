from op1_data_evaluation import run_evaluation
from op2_data_cleaning import run_cleaning
from op3_handle_missing_values import run_handling_missing_value
from op4_feature_selection import run_feature_selection
from op5_outlier_remover import run_remove_outlier
import pandas as pd

"""Executes the full data preprocessing pipeline.

    This function coordinates the sequential execution of four main operations:
    1. Data Evaluation: Analyzes the raw structure and identifies null/object columns.
    2. Data Cleaning: Converts object-type columns (with commas) to numeric floats.
    3. Missing Value Handling: Imputes missing values using the mean of each column.
    4. Feature Selection: Removes redundant features based on correlation analysis.
    5. Outlier Removal: Removes anomalous records using Interquartile Range (IQR).

    Args:
        file_path (str): The relative or absolute path to the input CSV file. 
            Defaults to 'data/dati_fia.csv'.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame, cleaned and filtered, 
            ready for model development.
"""

def main(file_path: str = 'data/dati_fia.csv') -> pd.DataFrame:
    
    # Load the initial dataset
    df = pd.read_csv(file_path)
    
    print("\n----- STARTING PIPELINE -----")

    # Operation 1: Evaluation
    evaluated_data = run_evaluation(df)

    # Operation 2: Data Cleaning (Conversion from string to float)
    cleaned_data = run_cleaning(
        df, 
        evaluated_data.object_columns_output,
        )

    # Operation 3: Handle Missing Values (Mean Imputation)
    data_without_missing_values = run_handling_missing_value(
        cleaned_data.df_output,
        evaluated_data.columns_with_nulls_output,
    )

    # Operation 4: Feature Selection (Correlation Heatmap and Redundancy removal)
    selected_features = run_feature_selection(
        data_without_missing_values.df_output,
    )

    # Operation 5: Outlier Removal
    selected_rows = run_remove_outlier(
        selected_features.df_output,
    )

    print("Final Dataset ready:")
    final_dataset = selected_rows.df_output
    print(final_dataset.head)

    print("\n----- PIPELINE COMPLETED SUCCESSFULLY -----")
    
    return final_dataset
    
if __name__ == "__main__":
    main()