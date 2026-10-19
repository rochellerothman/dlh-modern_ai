
#!/usr/bin/env python3
"""Module for visualizing missing values in a DataFrame."""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a DataFrame."""
    plt.figure(figsize=(12, 8))

    missing = df.isnull().to_numpy()
    rows, cols = np.where(missing)

    plt.scatter(rows, cols, marker='|')

    plt.yticks(
        np.arange(len(df.columns)),
        df.columns
    )

    plt.tight_layout()
    plt.show()
