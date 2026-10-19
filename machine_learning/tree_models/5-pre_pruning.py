#!/usr/bin/env python3
"""Find optimal decision tree pre-pruning hyperparameters."""

from sklearn import model_selection


def prepruning(X, y, clf):
    """Return the best decision tree parameters using grid search."""
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(2, 5),
        'min_samples_leaf': range(2, 5),
        'min_samples_split': range(2, 5)
    }

    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5
    )

    grid_search.fit(X, y)

    return grid_search.best_params_
