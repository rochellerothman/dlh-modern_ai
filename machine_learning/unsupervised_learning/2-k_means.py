#!/usr/bin/env python3
"""Module for performing K-Means clustering using Scikit-learn."""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Create and fit a K-Means clustering model."""
    model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    model.fit(X)
    return model
