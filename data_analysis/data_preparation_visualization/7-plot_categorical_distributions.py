#!/usr/bin/env python3
"""Plot distributions of categorical variables."""

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """Plot categorical features in a grid of bar charts."""
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

    for i, col in enumerate(columns_to_plot):
        df[col].value_counts().plot(
            kind='bar',
            ax=axes[i],
            color='skyblue'
        )
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)

    for i in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
