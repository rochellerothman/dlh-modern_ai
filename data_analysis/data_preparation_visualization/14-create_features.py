#!/usr/bin/env python3
"""Create engineered features for customer churn analysis."""

import pandas as pd


def create_features(df):
    """Create service count and tenure group features."""
    df = df.copy()

    service_columns = [
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    df['NumServices'] = (
        df[service_columns].eq('Yes').sum(axis=1)
        + df['InternetService'].isin(['DSL', 'Fiber optic']).astype(int)
    )

    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-48', '49-60', '60+']
    )

    df.drop(
        columns=service_columns + ['InternetService', 'tenure'],
        inplace=True
    )

    return df
