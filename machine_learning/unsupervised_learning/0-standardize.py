#!/usr/bin/env python3
"""Module for standardizing tabular data using Scikit-learn."""

from sklearn import preprocessing


def Standardize(X):
    """Standardize features to have zero mean and unit variance."""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
