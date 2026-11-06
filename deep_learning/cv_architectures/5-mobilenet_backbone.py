#!/usr/bin/env python3
"""Build the MobileNetV1 feature extraction backbone."""

from tensorflow import keras

depthwise_separable_conv = __import__(
    '4-depthwise_separable_conv'
).depthwise_separable_conv


def mobilenet_backbone(inputs):
    """Return MobileNetV1 feature maps before classification.

    Args:
        inputs: Input image tensor.

    Returns:
        The output tensor of the MobileNetV1 backbone.
    """
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding='same',
        use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = depthwise_separable_conv(x, 64)
    x = depthwise_separable_conv(x, 128, stride=2)
    x = depthwise_separable_conv(x, 128)
    x = depthwise_separable_conv(x, 256, stride=2)
    x = depthwise_separable_conv(x, 256)
    x = depthwise_separable_conv(x, 512, stride=2)

    for _ in range(5):
        x = depthwise_separable_conv(x, 512)

    x = depthwise_separable_conv(x, 1024, stride=2)
    x = depthwise_separable_conv(x, 1024)

    return x
