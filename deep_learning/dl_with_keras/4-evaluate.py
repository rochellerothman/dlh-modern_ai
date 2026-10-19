#!/usr/bin/env python3
"""Evaluate the performance of a trained Keras model."""


def evaluate_model(model, X, Y, verbose=0):
    """Return the loss and accuracy of a trained model."""
    loss, accuracy = model.evaluate(X, Y, verbose=verbose)
    return loss, accuracy
