"""Verify that saved artifacts reproduce metrics.csv without retraining."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# Main verification loop for all saved models.
def Main() -> None:
    testData = pd.read_csv(ROOT / "test_data.csv")
    X = testData.drop(columns="letter")
    y = testData["letter"]
    expected = pd.read_csv(ROOT / "metrics.csv").set_index("ML Model Name")

    for name, fileName in MODEL_FILES.items():
        model = joblib.load(ROOT / "models" / fileName)
        predicted = model.predict(X)
        probabilities = model.predict_proba(X)
        actual = {
            "Accuracy": accuracy_score(y, predicted),
            "AUC": roc_auc_score(y, probabilities, labels=list(model.classes_), multi_class="ovr", average="macro"),
            "Precision": precision_score(y, predicted, average="macro", zero_division=0),
            "Recall": recall_score(y, predicted, average="macro", zero_division=0),
            "F1": f1_score(y, predicted, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y, predicted),
        }
        for metric, value in actual.items():
            if abs(value - expected.loc[name, metric]) > 0.00006:
                raise AssertionError(f"{name}: {metric} does not match metrics.csv")
        print(f"PASS - {name}")

    print("All five saved models reproduce the recorded held-out metrics. No model was retrained.")


if __name__ == "__main__":
    Main()
