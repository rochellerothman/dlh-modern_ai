#!/usr/bin/env python3
"""Select the best cost-complexity pruning alpha."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Return the best pruning alpha and its trained classifier."""
    best_index = min(
        range(len(clfs)),
        key=lambda i: (
            -test_scores[i],
            abs(train_scores[i] - test_scores[i]),
            -ccp_alphas[i]
        )
    )

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
