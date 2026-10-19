#!/usr/bin/env python3
"""Plot the distributions of categorical features."""

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """Display categorical feature distributions in a grid."""
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(
            include=['object']
        ).columns.drop('Churn', errors='ignore').tolist()
    else:
        columns_to_plot = list(columns_to_plot)

    n_cols = 3
    n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        df[column].value_counts().plot(
            kind='bar',
            ax=axes[i]
        )
        axes[i].set_title(column)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelrotation=45)

    for i in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
