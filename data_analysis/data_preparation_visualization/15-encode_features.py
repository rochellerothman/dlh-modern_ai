#!/usr/bin/env python3
"""Encode categorical features for machine learning."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode churn, binary, tenure, and nominal features."""
    df = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df['Churn'] = churn_le.fit_transform(df['Churn'])

    binary_columns = [
        'Partner',
        'Dependents',
        'PaperlessBilling',
        'SeniorCitizen'
    ]

    binary_oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']]
    )

    for column in binary_columns:
        df[column] = binary_oe.fit_transform(
            df[[column]]
        ).astype(int)

    tenure_oe = preprocessing.OrdinalEncoder()
    df['TenureGroup'] = tenure_oe.fit_transform(
        df[['TenureGroup']].astype(str)
    ).astype(int)

    df = pd.get_dummies(
        df,
        columns=['Contract', 'PaymentMethod'],
        drop_first=True,
        dtype=int
    )

    return df, churn_le, binary_oe, tenure_oe
