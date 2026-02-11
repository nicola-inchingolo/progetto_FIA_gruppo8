from typing import NamedTuple
import pandas as pd

"""Removes outliers from the dataset using the Interquartile Range (IQR) method.

    This function identifies numeric columns and calculates the first (Q1) and 
    third (Q3) quartiles. It defines the acceptable range as [Q1 - 1.5 * IQR, 
    Q3 + 1.5 * IQR]. Any row containing at least one value outside these bounds 
    in any numeric column is removed.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df_output (pd.DataFrame): The DataFrame with outliers removed.
"""

class OutlierOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_remove_outlier(
        df: pd.DataFrame,
        ) -> OutlierOutputs :
    
    # Calculate the Interquartile Range (IQR)
    # The IQR is the distance between the 25th percentile (Q1) and the 75th percentile (Q3).
    # It represents the range where the central 50% of the data points lie.
    numeric_df = df.select_dtypes(include=['number'])
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outlier detection
    # Any value outside this range is statistically considered an anomaly (outlier) 
    # because it deviates significantly from the central distribution of the dataset.
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify rows that contain at least one outlier in numeric columns
    outliers_mask = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)
    
    # Filter the DataFrame: the '~' operator keeps rows where the mask is False
    df_cleaned = df[~outliers_mask]
    
    rows_removed = len(df) - len(df_cleaned)

    # Check if any outliers were found and removed
    if rows_removed > 0:
        print(f"Outlier handling complete. Rows removed: {rows_removed}")
    else:
        print("No outliers detected based on the 1.5 * IQR rule. The dataset remains unchanged.")

    print("\n" + "="*50 + "\n")

    return OutlierOutputs(
        df_output = df_cleaned
    )