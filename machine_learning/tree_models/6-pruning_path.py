#!/usr/bin/env python3
"""Retrieve the cost-complexity pruning path of a decision tree."""


def get_pruning_path(clf, X, y):
    """Return pruning alpha values and corresponding leaf impurities."""
    path = clf.cost_complexity_pruning_path(X, y)

    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    return ccp_alphas, impurities
