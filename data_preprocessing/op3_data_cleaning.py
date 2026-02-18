from typing import NamedTuple
import pandas as pd

"""Cleans categorical columns by converting them to numeric format.

    This function iterates through specified object-type columns, handles 
    decimal formatting (replacing commas with dots), and attempts to convert 
    the data to float. Values that cannot be converted are transformed into NaN.

    Args:
        df (pd.DataFrame): The input DataFrame.
        object_columns (pd.Index): Index of column names to target for conversion.

    Returns:
        df_output (pd.DataFrame): The cleaned DataFrame with converted numeric columns.
"""

class CleaningOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_cleaning(
        df: pd.DataFrame, 
        ) -> CleaningOutputs :
    
    # 'Sample code number' is a patient ID and holds no predictive value
    # The other columns are not relevant to cell tumor prediction
    cols_to_drop = ['Sample code number','Blood Pressure', 'Heart Rate']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    

    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        try:
            # Replace comma with dot for decimal consistency
            # Using astype(str) to ensure string methods work
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            
            # Convert to numeric (float)
            # errors='coerce' turns values that cannot be converted into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Successfully processed column: '{col}'")
            
        except Exception as e:
            print(f"Error during conversion of column '{col}': {e}")

    # Check how many numeric columns we have after conversion to evaluate the result of latest conversion
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(f"\nCleaning complete. Current numeric columns: {numeric_cols.size}/{df.columns.size}")
    
    print("=" * 50)

    return CleaningOutputs(
        df_output = df
    )