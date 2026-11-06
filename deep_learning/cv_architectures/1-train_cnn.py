#!/usr/bin/env python3
"""Compile and train a convolutional neural network."""

from tensorflow import keras


def compile_and_train_cnn(
    model, epochs, batch_size, x_train, y_train, x_val, y_val,
    optimizer_name='adam', optimizer_params=None
):
    """Compile and train a CNN with the selected optimizer.

    Args:
        model: Keras CNN model to train.
        epochs: Number of training epochs.
        batch_size: Number of samples per training batch.
        x_train: Training input images.
        y_train: One-hot encoded training labels.
        x_val: Validation input images.
        y_val: One-hot encoded validation labels.
        optimizer_name: Name of the Keras optimizer.
        optimizer_params: Optional optimizer configuration dictionary.

    Returns:
        A tuple containing the trained model and its History object.
    """
    if optimizer_params is None:
        optimizer_params = {}

    optimizer = keras.optimizers.get({
        'class_name': optimizer_name,
        'config': optimizer_params
    })

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val)
    )

    return model, history
