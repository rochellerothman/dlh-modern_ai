#!/usr/bin/env python3
"""Perform chi-square tests on categorical features."""

import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Return chi-square p-values for categorical features."""
    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns.drop('Churn', errors='ignore')

    p_values = {}

    for column in categorical_columns:
        contingency_table = pd.crosstab(df[column], df['Churn'])
        _, p_value, _, _ = stats.chi2_contingency(contingency_table)
        p_values[column] = p_value

    return p_values
