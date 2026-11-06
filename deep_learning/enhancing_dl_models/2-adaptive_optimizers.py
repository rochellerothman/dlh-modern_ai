#!/usr/bin/env python3
"""Configure SGD, Adam, and RMSprop optimizers."""

from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """Return the selected optimizer with its corresponding parameters.

    Args:
        name: Optimizer name: 'sgd', 'adam', or 'rmsprop'.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum factor used only by SGD.
        beta_1: First moment decay rate used only by Adam.
        beta_2: Second moment decay rate used only by Adam.
        rho: Squared gradient decay rate used only by RMSprop.

    Returns:
        A configured Keras optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    if name == 'sgd':
        return keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum
        )
    elif name == 'adam':
        return keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(
            learning_rate=learning_rate, rho=rho
        )
    else:
        raise ValueError("Unsupported optimizer: " + name)
