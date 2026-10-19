#!/usr/bin/env python3
"""Plot the distribution of customer churn."""

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Display a bar chart of churn counts."""
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()
    counts = counts.reindex(['No', 'Yes'])

    plt.bar(counts.index, counts.values,
            color=['skyblue', 'salmon'])

    plt.title('Churn Distribution')
    plt.ylabel('Count')
    plt.show()
