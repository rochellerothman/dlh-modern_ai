#!/usr/bin/env python3
"""Initialize a Keras Tuner for hyperparameter optimization."""

import keras_tuner


def initiate_tuner(
    tuner_type, build_model, seed,
    hyperband_iterations, max_trials, objective
):
    """Return a configured Hyperband, random, or Bayesian tuner.

    Args:
        tuner_type: Hyperband, RandomSearch, or BayesianOptimization.
        build_model: Function that builds a compiled Keras model.
        seed: Random seed for the tuner.
        hyperband_iterations: Number of Hyperband iterations.
        max_trials: Maximum trials for random or Bayesian search.
        objective: Metric to optimize, such as 'val_accuracy'.

    Returns:
        A configured Keras Tuner instance.

    Raises:
        ValueError: If the tuner type is unsupported.
    """
    if tuner_type == 'Hyperband':
        return keras_tuner.Hyperband(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            hyperband_iterations=hyperband_iterations,
            overwrite=True
        )
    elif tuner_type == 'RandomSearch':
        return keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=True
        )
    elif tuner_type == 'BayesianOptimization':
        return keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=True
        )
    else:
        raise ValueError("Unsupported tuner type: " + tuner_type)
