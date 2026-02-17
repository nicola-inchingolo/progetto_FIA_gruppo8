from typing import NamedTuple
import pandas as pd

"""
    Removes outliers from the DataFrame using the IQR method, applied per class.

    This function groups the data by the 'classtype_v1' column and calculates the 
    Interquartile Range (IQR) for numeric columns specific to each group. Rows falling 
    outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR] within their respective group are removed.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric features and 
            the 'classtype_v1' target column.

    Returns:
        df_output: A named tuple containing the cleaned DataFrame.
    """

class OutlierOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_remove_outlier(
        df: pd.DataFrame,
        ) -> OutlierOutputs :
    
    target_col = 'classtype_v1'
    
    # Identify numeric columns for outlier calculation without considering the target column
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    cleaned_group_list = []
    total_rows_removed = 0

    # Iterate over each class (group) present in the target column
    # We use '_' because we don't need the group name inside the loop, only the data (group_df)
    for _, group_df in df.groupby(target_col):
        
        # Calculate specific IQR for this group (Class)
        # Q1 and Q3 are calculated only on data from the current class
        Q1 = group_df[numeric_cols].quantile(0.25)
        Q3 = group_df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outlier rows only within this group
        outliers_mask = ((group_df[numeric_cols] < lower_bound) | 
                         (group_df[numeric_cols] > upper_bound)).any(axis=1)
        
        # Filter the group
        group_cleaned = group_df[~outliers_mask]
        
        # Track how many rows are removed
        rows_removed_in_group = len(group_df) - len(group_cleaned)
        total_rows_removed += rows_removed_in_group
        
        # Add the cleaned group to the list
        cleaned_group_list.append(group_cleaned)

    # Reconstruct the full DataFrame by concatenating the cleaned groups
    # sort_index() restores the original row order
    df_final = pd.concat(cleaned_group_list).sort_index()

    # Log results
    if total_rows_removed > 0:
        print(f"Outlier handling complete (Group-wise). Total rows removed: {total_rows_removed}")
    else:
        print("No outliers detected based on the group-wise 1.5 * IQR rule.")

    print("\n" + "="*50 + "\n")

    return OutlierOutputs(
        df_output = df_final
    )