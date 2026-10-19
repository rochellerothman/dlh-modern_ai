#!/usr/bin/env python3
"""Generate class predictions using a trained Keras model."""

import tensorflow as tf


def predict(model, X, verbose=0):
    """Return predicted class labels for the input data."""
    predictions = model.predict(X, verbose=verbose)
    return tf.argmax(predictions, axis=1)
