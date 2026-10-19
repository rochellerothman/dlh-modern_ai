#!/usr/bin/env python3
"""Train a Keras model and log metrics to TensorBoard."""

from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """Train a model and record training metrics in TensorBoard."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir + "/" + timestamp

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_path,
        histogram_freq=1
    )

    model.fit(
        X,
        Y,
        epochs=epochs,
        verbose=verbose,
        callbacks=[tensorboard_callback]
    )
