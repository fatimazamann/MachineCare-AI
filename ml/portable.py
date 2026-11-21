"""Version-independent copy of the trained model.

A joblib/pickle file only loads reliably with the same scikit-learn and pandas versions that saved it.
The best model is a Logistic Regression pipeline (median imputer -> standard scaler -> logistic
regression), so we also store its learned numbers in plain JSON and redo the maths with NumPy."""

from __future__ import annotations

import json

import numpy as np


def export_pipeline(pipe, features, name, path, threshold=0.5, smooth="15min") -> bool:
    """Save a fitted imputer->scaler->LogisticRegression pipeline as JSON. Returns False for other models."""
    from sklearn.linear_model import LogisticRegression
    model = pipe.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        return False
    num = pipe.named_steps["prep"].named_transformers_["num"]
    imp, sc = num.named_steps["impute"], num.named_steps.get("scale")
    data = {"name": name, "type": "logistic_regression", "features": list(features), "threshold": threshold, "smooth": smooth,
            "impute_median": imp.statistics_.tolist(),
            "scale_mean": sc.mean_.tolist() if sc is not None else [0.0] * len(features),
            "scale_std": sc.scale_.tolist() if sc is not None else [1.0] * len(features),
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return True


class PortableLogit:
    """Loads the JSON above and gives the same probabilities as the scikit-learn pipeline."""

    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.name, self.features, self.threshold = d["name"], d["features"], d["threshold"]
        self.med = np.array(d["impute_median"])
        self.mean = np.array(d["scale_mean"])
        self.std = np.array(d["scale_std"])
        self.coef = np.array(d["coef"])
        self.b = d["intercept"]

    def predict_proba(self, X):
        x = np.asarray(X[self.features] if hasattr(X, "columns") else X, dtype=float)
        x = np.where(np.isnan(x), self.med, x)
        z = ((x - self.mean) / self.std) @ self.coef + self.b
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])
