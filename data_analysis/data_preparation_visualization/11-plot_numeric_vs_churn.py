#!/usr/bin/env python3
"""Plot numeric feature distributions grouped by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Display grouped histograms for churn categories."""
    no_churn = df[df['Churn'] == 'No'][col].dropna()
    yes_churn = df[df['Churn'] == 'Yes'][col].dropna()

    plt.figure(figsize=(12, 8))

    plt.hist(
        [no_churn, yes_churn],
        bins=30,
        label=['No', 'Yes']
    )

    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')
    plt.show()
