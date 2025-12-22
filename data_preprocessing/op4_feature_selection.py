from typing import NamedTuple
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

"""Identifies and removes redundant features based on Pearson correlation.

    This function calculates a correlation matrix and visualizes it using a 
    heatmap. It then identifies pairs of features with a correlation coefficient 
    higher than 0.8 and drops one column from each pair to reduce 
    multicollinearity and improve model performance.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df_output (pd.DataFrame): The DataFrame after removing redundant (highly correlated) features.
"""

class FeatureSelectionOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_feature_selection(
        df: pd.DataFrame,
        ) -> FeatureSelectionOutputs :
    
    # Calculate the correlation matrix to identify linear relationships between features
    # values close to 1 or -1 indicate a strong relationship, while 0 indicates no relationship
    correlation_matrix = df.corr()

    # Generate a Heatmap to visually represent the correlation matrix
    # We use a heatmap because it allows for immediate identification of patterns 
    # and highly correlated features through color-coding
    # Warmer colors (red) indicate strong positive correlation, while 
    # cooler colors (blue) indicate strong negative correlation
    print("Correlation Matrix Heatmap:")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True,
        fmt='.2f',
        cmap='coolwarm', 
        vmax=1, vmin=-1
    )
    plt.title("Feature Correlation Heatmap")

    # Save the heatmap plot as a PNG image in the 'data' directory
    save_path = os.path.join('data/plot', 'correlation_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # Close the plot to free up memory
    plt.close()

    # Consider absolute values to detect both strong positive and negative correlations
    # (both contribute to redundancy/multicollinearity)
    correlation_matrix_abs = correlation_matrix.abs()

    # Select the upper triangle of the correlation matrix
    # k=1 ignores the main diagonal (correlations of 1.0 with themselves)
    # np.triu ensures we only check M[i][j] and ignore the symmetric M[j][i]
    upper_triangle = correlation_matrix_abs.where(
        np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool)
    )

    # Identify columns to drop: creates a list of columns with correlation > 0.8
    columns_to_drop_corr = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > 0.8)
    ]

    if not columns_to_drop_corr:
        print("No redundant features found (all correlations <= 0.8).")
    else:
        print("The following redundant columns were removed: ", columns_to_drop_corr)
        # Drop the highly correlated columns from the DataFrame
        df = df.drop(columns=columns_to_drop_corr)

    print("\n" + "="*50 + "\n")

    return FeatureSelectionOutputs(
        df_output = df
    )