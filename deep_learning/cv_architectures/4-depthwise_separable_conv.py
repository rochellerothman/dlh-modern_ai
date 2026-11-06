#!/usr/bin/env python3
"""Implement a MobileNetV1 depthwise separable convolution block."""

from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Return the output of a depthwise separable convolution block.

    Args:
        X: Input tensor.
        filters: Number of output channels in the pointwise convolution.
        stride: Stride applied to the depthwise convolution.

    Returns:
        The output tensor after depthwise and pointwise convolutions.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        use_bias=False
    )(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
