#!/usr/bin/env python3
"""Retrieve and sort Random Forest feature importances."""

import numpy as np


def feature_importance(rf):
    """Return feature importances and ascending sorted indices."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    return importances, indices
