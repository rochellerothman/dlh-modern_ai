#!/usr/bin/env python3
"""Module for visualizing missing values in a DataFrame."""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a DataFrame using a scatter plot."""
    plt.figure(figsize=(12, 8))

    missing = df.isnull().to_numpy()
    rows, columns = np.where(missing)

    plt.scatter(rows, columns, marker='|')
    plt.yticks(np.arange(len(df.columns)), df.columns)

    plt.tight_layout()
    plt.show()
