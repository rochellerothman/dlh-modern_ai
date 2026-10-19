#!/usr/bin/env python3
"""Create a Random Forest classifier using Scikit-learn."""

from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Return a Random Forest classifier with the given parameters."""
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    return model
