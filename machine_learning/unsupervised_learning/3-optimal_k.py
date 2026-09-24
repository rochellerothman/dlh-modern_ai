#!/usr/bin/env python3
"""Module for evaluating the optimal number of K-Means clusters."""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means models using inertia and silhouette scores."""
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(
            X,
            n_clusters=k,
            random_state=random_state
        )

        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_values.append(
            metrics.silhouette_score(X, model.labels_)
        )

    return ks, inertia_values, silhouette_values
