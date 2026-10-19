#!/usr/bin/env python3
"""Visualize numerical feature distributions by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Plot overlapping histograms grouped by churn status."""
    plt.figure(figsize=(12, 8))

    for churn_value in ['No', 'Yes']:
        data = df[df['Churn'] == churn_value][col]
        plt.hist(data, bins=30, alpha=0.7, label=churn_value)

    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')
    plt.show()
