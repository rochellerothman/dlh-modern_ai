#!/usr/bin/env python3
"""Configure gradient descent optimizers and their batch sizes."""

from tensorflow import keras


def train_with_gradient_descent_variant(variant, learning_rate,
                                       x_train, batch_size):
    """Return an SGD optimizer and batch size for the selected variant.

    Args:
        variant: 'batch', 'stochastic', or 'mini_batch'.
        learning_rate: Learning rate for the optimizer.
        x_train: Training input dataset.
        batch_size: Batch size for mini-batch gradient descent.

    Returns:
        A tuple containing the optimizer and selected batch size.

    Raises:
        ValueError: If the variant is unsupported.
    """
    if variant == 'batch':
        bs = len(x_train)
    elif variant == 'stochastic':
        bs = 1
    elif variant == 'mini_batch':
        bs = batch_size
    else:
        raise ValueError(
            "variant must be 'batch', 'stochastic', or 'mini_batch'"
        )

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    return optimizer, bs
