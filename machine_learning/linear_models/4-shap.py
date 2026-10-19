#!/usr/bin/env python3
"""Generate SHAP explanations for trained regression models."""

import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Return a SHAP explainer and SHAP values for test samples."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
