import numpy as np
from sklearn.calibration import calibration_curve


def reliability_table(y_true, y_prob, n_bins=10):
    """
    Returns a simple calibration table:
    predicted vs observed frequencies.
    """
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile",
    )

    rows = []
    for i in range(len(prob_true)):
        rows.append({
            "bin": i,
            "avg_predicted": float(prob_pred[i]),
            "actual_rate": float(prob_true[i]),
            "gap": float(prob_pred[i] - prob_true[i]),
        })

    return rows
