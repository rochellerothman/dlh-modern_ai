#!/usr/bin/env python3
"""Save and load the weights of a Keras model."""


def save_model_weights(model, filepath):
    """Save the weights of a Keras model to a file."""
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """Load saved weights into an existing Keras model."""
    model.load_weights(filepath)
