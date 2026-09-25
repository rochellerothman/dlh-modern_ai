#!/usr/bin/env python3
"""Module for visualizing missing values in a DataFrame."""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a DataFrame."""
    plt.figure(figsize=(12, 8))

    missing = df.isnull()

    for i, column in enumerate(df.columns):
        rows = np.where(missing[column])[0]
        plt.scatter(
            rows,
            np.full(len(rows), i),
            marker='|'
        )

    plt.yticks(
        np.arange(len(df.columns)),
        df.columns
    )

    plt.tight_layout()
    plt.show()
