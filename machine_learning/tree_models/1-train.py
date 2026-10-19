#!/usr/bin/env python3
"""Train a tree-based classifier using Scikit-learn."""


def train_tree(clf, X, y):
    """Fit a classifier to the supplied features and labels."""
    clf.fit(X, y)
