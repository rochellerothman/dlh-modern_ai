#!/usr/bin/env python3
"""Build classification models with activation-specific initializers."""

from tensorflow import keras


def build_model_initializer_by_activation(
    input_dim, hidden_units, activation
):
    """Return a compiled model with an appropriate hidden initializer.

    Args:
        input_dim: Number of input features.
        hidden_units: Number of hidden neurons.
        activation: 'sigmoid', 'tanh', 'relu', or 'leaky_relu'.

    Returns:
        A compiled Keras model with ten softmax outputs.

    Raises:
        ValueError: If the activation is unsupported.
    """
    if activation in ('sigmoid', 'tanh'):
        initializer = keras.initializers.GlorotUniform()
    elif activation in ('relu', 'leaky_relu'):
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError("Unsupported activation: " + activation)

    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_initializer=initializer
        ),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
