#!/usr/bin/env python3
"""Visualize correlations between numerical features."""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """Display an annotated correlation heatmap."""
    plt.figure(figsize=(6, 5))

    correlation_matrix = df.select_dtypes(
        include=['number']
    ).corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    plt.title('Correlation Matrix')
    plt.savefig('Task_9.png')
    plt.show()
