#!/usr/bin/env python3
"""Create SVM classifiers with different kernels."""

from sklearn import svm


def get_SVM_model(name, random_state):
    """Return an untrained SVM classifier with the specified kernel."""
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
