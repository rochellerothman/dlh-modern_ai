#!/usr/bin/env python3
"""Display the decision rules of a trained decision tree."""

from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print a readable text representation of a decision tree."""
    rules = tree.export_text(
        clf,
        feature_names=feature_names
    )

    for index, name in enumerate(class_names):
        rules = rules.replace(
            'class: {}'.format(index),
            'class: {}'.format(name)
        )

    print(rules, end='')
