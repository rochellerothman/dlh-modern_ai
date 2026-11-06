#!/usr/bin/env python3
"""Configure standard, momentum, and Nesterov SGD optimizers."""

from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """Return an SGD optimizer configured for the selected variant.

    Args:
        name: SGD variant name.
        lr: Learning rate.
        momentum: Momentum factor.
        nesterov: Whether to use Nesterov for the Nesterov variant.

    Returns:
        A configured Keras SGD optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    if name == 'SGD':
        return keras.optimizers.SGD(learning_rate=lr)
    elif name == 'SGD+Momentum':
        return keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum
        )
    elif name == 'SGD+Momentum+Nesterov':
        return keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum, nesterov=nesterov
        )
    else:
        raise ValueError("Unsupported SGD optimizer variant: " + name)
