#!/usr/bin/env python3
"""Generate predictions using a trained classifier."""


def generate_predictions(clf, X):
    """Return predicted class labels for the input features."""
    return clf.predict(X)
