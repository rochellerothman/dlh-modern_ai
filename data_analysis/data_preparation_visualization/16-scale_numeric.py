#!/usr/bin/env python3
"""Standardize numeric features for machine learning."""

from sklearn import preprocessing


def scale_numeric(df):
    """Standardize MonthlyCharges and TotalCharges."""
    df = df.copy()

    numeric_columns = ['MonthlyCharges', 'TotalCharges']

    scaler = preprocessing.StandardScaler()

    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )

    return df
