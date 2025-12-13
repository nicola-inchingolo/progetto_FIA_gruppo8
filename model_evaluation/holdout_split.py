import numpy as np
import pandas as pd
import random


def houldout_split(dataset: pd.DataFrame):
    train_percentage = random.uniform(0.6, 0.9)  # range sensato

    split_index = int(len(dataset) * train_percentage)
    train_section = dataset.iloc[:split_index]
    test_section = dataset.iloc[split_index:]

    return train_section, test_section
