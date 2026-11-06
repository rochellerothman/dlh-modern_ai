#!/usr/bin/env python3
"""Create a configurable convolutional neural network."""

from tensorflow import keras


def create_cnn_model(
    input_shape, filters, kernel_sizes, activations, pooling_type='max'
):
    """Return a compiled CNN for ten-class image classification.

    Args:
        input_shape: Input image shape, excluding the batch dimension.
        filters: Number of filters in each convolutional layer.
        kernel_sizes: Kernel size for each convolutional layer.
        activations: Activation for each convolutional layer.
        pooling_type: Pooling method, either 'max' or 'avg'.

    Returns:
        A compiled Keras Sequential model.

    Raises:
        ValueError: If pooling is unsupported or list lengths differ.
    """
    if not len(filters) == len(kernel_sizes) == len(activations):
        raise ValueError("Convolution configuration lengths must match")

    if pooling_type == 'max':
        pooling_layer = keras.layers.MaxPooling2D
    elif pooling_type == 'avg':
        pooling_layer = keras.layers.AveragePooling2D
    else:
        raise ValueError("pooling_type must be 'max' or 'avg'")

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for index in range(len(filters)):
        model.add(keras.layers.Conv2D(
            filters=filters[index],
            kernel_size=kernel_sizes[index],
            activation=activations[index]
        ))
        model.add(pooling_layer(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
