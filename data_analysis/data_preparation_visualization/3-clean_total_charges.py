
# !/usr/bin/env python3
"""Handle missing values in the TotalCharges column."""


def clean_total_charges(df, method='drop'):
    """Clean missing TotalCharges using the selected method."""
    df = df.copy()

    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        median = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median)
    elif method == 'impute':
        missing = df['TotalCharges'].isna()
        df.loc[missing, 'TotalCharges'] = (
            df.loc[missing, 'MonthlyCharges']
            * df.loc[missing, 'tenure']
        )

    return df
