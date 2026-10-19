#!/usr/bin/env python3
"""Compile a Keras model for training."""

from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """Compile a model using SGD and binary cross-entropy loss."""
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
