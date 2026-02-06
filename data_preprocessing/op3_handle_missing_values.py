from typing import NamedTuple
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

"""Handles missing values in the specified columns using mean imputation.

    This function iterates through columns identified as having null values and 
    replaces the missing entries with the calculated mean of each respective 
    column. It then performs a final verification to ensure no null values remain.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_with_nulls (pd.Index): Index of column names that require imputation.

    Returns:
        df_output (pd.DataFrame): The DataFrame with missing values imputed.
"""

class ImputationOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_handling_missing_value(
        df: pd.DataFrame,
        columns_with_nulls: pd.Index 
        ) -> ImputationOutputs :
    
    random_col = random.choice(columns_with_nulls.index.tolist())
    
    plt.figure(figsize=(10, 6))
    df[random_col].plot(kind='hist', title=f"{random_col} before imputation")
    before_imp_path = os.path.join('data/plot', 'hist_col_before_imputation.png')
    plt.savefig(before_imp_path)
    plt.close()

    col_mean = df[random_col].mean()
    print(f"The value of {random_col} mean is {col_mean:.2f}")

    print("\n" + "="*50 + "\n")

    for column in columns_with_nulls.index:
        # Fill missing values with the mean of the column
        # Using the mean preserve the overall average of the distribution 
        # without introducing significant bias.
        df[column] = df[column].fillna(df[column].mean())

    # We expect to see an increase in frequency (a taller bar) at the mean value
    # Comparing this plot with the 'before' version confirms that the 
    # imputation was successful and shows how the data distribution has shifted
    plt.figure(figsize=(10, 6))
    df[random_col].plot(kind='hist', title=f"{random_col} after imputation")
    after_imp = os.path.join('data/plot', 'hist_col_after_imputation.png')
    plt.savefig(after_imp)
    plt.close()

    # Verify that there are no more missing values in the DataFrame
    print("Checking for remaining missing values in the DataFrame:")
    print(df.isnull().sum())

    if df.isnull().sum().sum() == 0:
        print("\nSuccess: No missing values remaining!")
    else:
        print("\nWarning: Some missing values could not be filled.")

    print("\n" + "="*50 + "\n")

    return ImputationOutputs(
        df_output = df
    )