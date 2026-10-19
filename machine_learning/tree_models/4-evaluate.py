#!/usr/bin/env python3
"""Evaluate classification model performance."""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Return a classification report for predicted labels."""
    report = metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

    return report
