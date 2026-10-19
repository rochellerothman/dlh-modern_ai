#!/usr/bin/env python3
"""Perform Welch's t-tests on numerical features."""

from scipy import stats


def ttest_numeric(df):
    """Return Welch's t-test p-values for numerical columns."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    p_values = {}

    for column in numeric_columns:
        churn_yes = df[df['Churn'] == 'Yes'][column].dropna()
        churn_no = df[df['Churn'] == 'No'][column].dropna()

        _, p_value = stats.ttest_ind(
            churn_yes,
            churn_no,
            equal_var=False
        )

        p_values[column] = p_value

    return p_values
