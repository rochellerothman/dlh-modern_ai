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
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'PaperlessBilling'
    ]

    binary_oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']] * len(binary_columns)
    )

    df[binary_columns] = binary_oe.fit_transform(
        df[binary_columns]
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
