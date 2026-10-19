#!/usr/bin/env python3
"""Visualize churn rates across categorical features."""

import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Plot the proportion of churned customers by category."""
    churn_rates = df.groupby(col)['Churn'].apply(
        lambda values: (values == 'Yes').mean()
    )

    plt.figure(figsize=(12, 8))
    churn_rates.plot(kind='bar')

    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Task_10.png')
    plt.show()
