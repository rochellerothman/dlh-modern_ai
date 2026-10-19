#!/usr/bin/env python3
"""Build a deep neural network using Keras Sequential."""

from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """Build a deep neural network for multiclass classification."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for neurons in hidden_layers:
        model.add(keras.layers.Dense(
            neurons,
            activation='relu'
        ))

    model.add(keras.layers.Dense(
        10,
        activation='softmax'
    ))

    return model
