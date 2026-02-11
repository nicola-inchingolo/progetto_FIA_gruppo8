from typing import NamedTuple
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

"""Identifies and removes redundant features based on Pearson correlation.

    This function calculates a correlation matrix ONLY on input features and visualizes it.
    It identifies pairs of features with a correlation coefficient higher than 0.8 
    and drops one column from each pair. The target variable is excluded from this 
    process to prevent removing good predictors.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df_output (pd.DataFrame): The DataFrame after removing redundant features.
"""

class FeatureSelectionOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_feature_selection(
        df: pd.DataFrame,
        ) -> FeatureSelectionOutputs :
    
    target_column = 'classtype_v1'
    
    # Crea un DataFrame temporaneo con le sole feature (escludendo il target)
    # Vogliamo evitare di calcolare la correlazione tra feature e target qui,
    # perché una forte correlazione col target è positiva e non va eliminata.
    df_features = df.drop(columns=[target_column], errors='ignore')
    
    # Calcola la matrice di correlazione sulle sole feature di input
    correlation_matrix = df_features.corr()

    # Generate a Heatmap
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

    # Save the heatmap plot
    if not os.path.exists('data/plot'):
        os.makedirs('data/plot')
    save_path = os.path.join('data/plot', 'correlation_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Consider absolute values
    correlation_matrix_abs = correlation_matrix.abs()

    # Select the upper triangle of the correlation matrix
    upper_triangle = correlation_matrix_abs.where(
        np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool)
    )

    # Identify columns to drop: creates a list of columns with correlation > 0.8
    # Il target non è presente in 'upper_triangle', quindi non rischiamo di rimuoverlo
    columns_to_drop_corr = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > 0.8)
    ]

    if not columns_to_drop_corr:
        print("No redundant features found (all correlations <= 0.8).")
    else:
        print("The following redundant columns were removed: ", columns_to_drop_corr)
        # Drop the highly correlated columns from the ORIGINAL DataFrame
        df = df.drop(columns=columns_to_drop_corr)

    print("\n" + "="*50 + "\n")

    return FeatureSelectionOutputs(
        df_output = df
    )