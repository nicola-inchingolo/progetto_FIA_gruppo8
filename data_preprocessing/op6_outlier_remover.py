from typing import NamedTuple
import pandas as pd

"""Removes outliers using a Class-Specific IQR method with safeguards.

    Instead of applying a global filter, this function handles outliers separately 
    for each class. This is crucial for medical datasets where 'disease' cases 
    naturally have extreme values that are not errors but signals.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df_output (pd.DataFrame): The DataFrame with outliers removed intelligently.
"""

class OutlierOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_remove_outlier(
        df: pd.DataFrame,
        ) -> OutlierOutputs :
    
    target_col = 'classtype_v1'
    original_rows = len(df)
    
    # Container for the processed data
    cleaned_data_chunks = []

    print(f"Starting Stratified Outlier Removal (Total rows: {original_rows})...")

    # Iterate through each class (e.g., 2.0 and 4.0) independently
    # This ensures Malignant cases are compared only to other Malignant cases
    classes = df[target_col].unique()
    
    for class_value in classes:
        # Select subset of data belonging to the current class
        df_class = df[df[target_col] == class_value].copy()
        
        # Select numeric columns for outlier detection, excluding the target itself
        numeric_cols = df_class.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        # Initialize a mask of False (keep all rows by default)
        outliers_mask = pd.Series(False, index=df_class.index)
        
        for col in numeric_cols:
            Q1 = df_class[col].quantile(0.25)
            Q3 = df_class[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # SAFEGUARD: If IQR is 0 (all values are identical, e.g., all 1.0),
            # any variation would be flagged as outlier. We SKIP outlier removal 
            # for this column in this class to preserve data.
            if IQR == 0:
                continue
            
            # Define bounds (standard 1.5 multiplier)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers in this specific column
            col_outliers = (df_class[col] < lower_bound) | (df_class[col] > upper_bound)
            
            # Update the global mask for this class
            outliers_mask = outliers_mask | col_outliers

        # Keep only rows that are NOT outliers
        df_class_cleaned = df_class[~outliers_mask]
        cleaned_data_chunks.append(df_class_cleaned)
        
        removed_in_class = len(df_class) - len(df_class_cleaned)
        print(f" - Class {class_value}: removed {removed_in_class} outliers out of {len(df_class)} rows.")

    # Reassemble the dataset
    df_final = pd.concat(cleaned_data_chunks).sort_index()
    
    total_removed = original_rows - len(df_final)
    print(f"Outlier handling complete. Total rows removed: {total_removed}")
    print("\n" + "="*50 + "\n")

    return OutlierOutputs(
        df_output = df_final
    )