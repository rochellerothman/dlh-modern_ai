#!/usr/bin/env python3
"""Build a shallow neural network using Keras Sequential."""

from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Return a Sequential model with one hidden layer."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(
            neurons_h,
            activation='sigmoid'
        ),
        keras.layers.Dense(
            10,
            activation='softmax'
        )
    ])

    return model
