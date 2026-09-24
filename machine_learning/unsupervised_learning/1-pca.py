#!/usr/bin/env python3
"""Module for performing Principal Component Analysis."""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Apply PCA to data and return transformed data and fitted PCA."""
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
