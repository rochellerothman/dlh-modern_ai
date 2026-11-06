#!/usr/bin/env python3
"""Run hyperparameter tuning and retrieve the best configuration."""


def search_and_return_best_model(
    tuner, x_train, y_train, epochs, validation_split, verbose=0
):
    """Run the tuner search and return its best hyperparameters.

    Args:
        tuner: A configured Keras Tuner instance.
        x_train: Training input data.
        y_train: Training target data.
        epochs: Number of training epochs per trial.
        validation_split: Fraction of data reserved for validation.
        verbose: Verbosity level for the search.

    Returns:
        The best HyperParameters object found by the tuner.
    """
    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
    )

    return tuner.get_best_hyperparameters(num_trials=1)[0]
