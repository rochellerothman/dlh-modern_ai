#!/usr/bin/env python3
"""Create a linear regression model using Scikit-learn."""

from sklearn import linear_model


def Linear_Regression():
    """Return an untrained ordinary least squares regression model."""
    model = linear_model.LinearRegression()
    return model
