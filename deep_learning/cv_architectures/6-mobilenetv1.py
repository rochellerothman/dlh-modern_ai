#!/usr/bin/env python3
"""Build the MobileNetV1 classification architecture."""

from tensorflow import keras

mobilenet_backbone = __import__('5-mobilenet_backbone').mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """Return a MobileNetV1 classification model.

    Args:
        input_shape: Input image shape, excluding the batch dimension.
        num_classes: Number of output classes.

    Returns:
        A Keras Model implementing MobileNetV1.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = mobilenet_backbone(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        num_classes, activation='softmax'
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='MobileNetV1'
    )
