#!/usr/bin/env python3
"""Configure SGD with stepwise learning rate decay schedules."""

from tensorflow import keras


def get_optimizer_SGD_with_schedule(
    schedule_type, initial_lr, decay_steps, decay_rate, momentum
):
    """Return an SGD optimizer and its learning rate schedule.

    Args:
        schedule_type: 'exponential' or 'inverse_time'.
        initial_lr: Initial learning rate.
        decay_steps: Number of optimizer steps per decay interval.
        decay_rate: Decay rate for the selected schedule.
        momentum: Momentum factor for SGD.

    Returns:
        A tuple containing the SGD optimizer and learning rate schedule.

    Raises:
        ValueError: If the schedule type is unsupported.
    """
    if schedule_type == 'exponential':
        schedule_class = keras.optimizers.schedules.ExponentialDecay
    elif schedule_type == 'inverse_time':
        schedule_class = keras.optimizers.schedules.InverseTimeDecay
    else:
        raise ValueError("Unsupported schedule type: " + schedule_type)

    lr_schedule = schedule_class(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum
    )

    return optimizer, lr_schedule
