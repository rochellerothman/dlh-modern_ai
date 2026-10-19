#!/usr/bin/env python3
"""Plot churn rates for categorical features."""

import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Display the churn rate for each category."""
    churn_rates = df.groupby(col)['Churn'].apply(
        lambda values: (values == 'Yes').mean()
    )

    plt.figure(figsize=(12, 8))
    plt.bar(churn_rates.index, churn_rates.values)
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.show()
