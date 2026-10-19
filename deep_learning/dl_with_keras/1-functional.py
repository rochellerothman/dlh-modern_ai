#!/usr/bin/env python3
"""Build a shallow neural network using the Keras Functional API."""

from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Create a functional model for multiclass classification."""
    inputs = keras.Input(shape=(input_dim,))

    hidden = keras.layers.Dense(
        neurons_h,
        activation='sigmoid'
    )(inputs)

    outputs = keras.layers.Dense(
        10,
        activation='softmax'
    )(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
