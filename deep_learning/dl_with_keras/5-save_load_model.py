#!/usr/bin/env python3
"""Save and load Keras models."""

from tensorflow import keras


def save_model(model, filepath):
    """Save a Keras model to the specified file path."""
    model.save(filepath)


def load_model(filepath):
    """Load and return a previously saved Keras model."""
    model = keras.models.load_model(filepath)
    return model
