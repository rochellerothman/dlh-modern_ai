#!/usr/bin/env python3
"""Build a neural network with L2-regularized hidden layers."""

from tensorflow import keras


def build_model_with_L2_regularization(
    input_dim, hidden_units, n_layers, lambda_l2
):
    """Return a model with L2 regularization on hidden kernel weights.

    Args:
        input_dim: Number of input features.
        hidden_units: Number of neurons in each hidden layer.
        n_layers: Number of hidden layers.
        lambda_l2: Strength of L2 regularization.

    Returns:
        A Keras model with ReLU hidden layers and ten softmax outputs.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    for _ in range(n_layers):
        model.add(keras.layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(lambda_l2)
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
