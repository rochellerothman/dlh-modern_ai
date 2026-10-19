#!/usr/bin/env python3
"""Train a Keras neural network model."""


def train_model(model, X, Y, epochs, verbose=1):
    """Train the model using the supplied data and epochs."""
    model.fit(X, Y, epochs=epochs, verbose=verbose)
