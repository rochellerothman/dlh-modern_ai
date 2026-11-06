#!/usr/bin/env python3
"""Create an early stopping callback that restores the best weights."""

from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """Return a configured Keras EarlyStopping callback.

    Args:
        patience: Number of epochs to wait without improvement.
        monitor: Training or validation metric to monitor.
        verbose: Verbosity mode for callback messages.

    Returns:
        An EarlyStopping callback that restores the best model weights.
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        restore_best_weights=True
    )
