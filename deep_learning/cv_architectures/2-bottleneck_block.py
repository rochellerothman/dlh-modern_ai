#!/usr/bin/env python3
"""Implement a ResNet bottleneck residual block."""

from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Return the output tensor of a bottleneck residual block.

    Args:
        x: Input tensor.
        filters: Number of filters in the reduced representation.
        stride: Stride for the first convolution and projection.
        downsample: Whether to use a projection shortcut.
        name: Optional prefix for layer names.

    Returns:
        The output tensor after residual addition and ReLU.
    """
    prefix = name + '_' if name else ''
    shortcut = x

    y = keras.layers.Conv2D(
        filters, (1, 1), strides=stride, use_bias=False,
        name=prefix + 'conv1' if name else None
    )(x)
    y = keras.layers.BatchNormalization(
        name=prefix + 'bn1' if name else None
    )(y)
    y = keras.layers.ReLU(
        name=prefix + 'relu1' if name else None
    )(y)

    y = keras.layers.Conv2D(
        filters, (3, 3), padding='same', use_bias=False,
        name=prefix + 'conv2' if name else None
    )(y)
    y = keras.layers.BatchNormalization(
        name=prefix + 'bn2' if name else None
    )(y)
    y = keras.layers.ReLU(
        name=prefix + 'relu2' if name else None
    )(y)

    y = keras.layers.Conv2D(
        filters * 4, (1, 1), use_bias=False,
        name=prefix + 'conv3' if name else None
    )(y)
    y = keras.layers.BatchNormalization(
        name=prefix + 'bn3' if name else None
    )(y)

    if downsample:
        shortcut = keras.layers.Conv2D(
            filters * 4, (1, 1), strides=stride, use_bias=False,
            name=prefix + 'shortcut_conv' if name else None
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            name=prefix + 'shortcut_bn' if name else None
        )(shortcut)

    y = keras.layers.Add(
        name=prefix + 'add' if name else None
    )([y, shortcut])

    return keras.layers.ReLU(
        name=prefix + 'out' if name else None
    )(y)
