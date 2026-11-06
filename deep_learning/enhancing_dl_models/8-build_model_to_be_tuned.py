#!/usr/bin/env python3
"""Build a tunable model for MNIST classification."""

from tensorflow import keras


def build_model(hp):
    """Return a compiled model using Keras Tuner hyperparameters.

    Args:
        hp: A Keras Tuner HyperParameters instance.

    Returns:
        A compiled Keras Sequential classification model.
    """
    num_layers = hp.Int('num_layers', min_value=1, max_value=2)
    units = hp.Int('units', min_value=4, max_value=12, step=4)
    activation = hp.Choice('activation', values=['relu', 'sigmoid'])
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(784,)))

    for _ in range(num_layers):
        model.add(keras.layers.Dense(units, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
