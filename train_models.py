"""Train five saved classifiers for the ML Assignment 2 project.

The project uses UCI's Letter Recognition data: 20,000 rows, 16 numeric
features, and 26 target classes. The test split becomes test_data.csv which is split as per 80(train)-20(test).
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
RAW_DATA_PATH = ROOT / "letter_recognition_raw.csv"
TARGET_COLUMN = "letter"
FEATURE_COLUMNS = [
    "x_box",
    "y_box",
    "width",
    "height",
    "onpix",
    "x_bar",
    "y_bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x_edge",
    "xegvy",
    "y_edge",
    "yegvx",
]

# Functions for loading dataset, evaluating models, and main training loop.
def LoadDataset() -> pd.DataFrame:
    """Download the public UCI source once, then return a labelled DataFrame."""
    if not RAW_DATA_PATH.exists():
        print("Downloading the UCI Letter Recognition dataset...")
        urlretrieve(DATA_URL, RAW_DATA_PATH)
    return pd.read_csv(RAW_DATA_PATH, header=None, names=[TARGET_COLUMN, *FEATURE_COLUMNS])

# Function for evaluating a trained model on test data.
def EvaluateModel(classifierModel, testFeatures: pd.DataFrame, testLabels: pd.Series) -> dict[str, float]:
    """Calculate required metrics using macro averaging for the 26 classes."""
    predicted = classifierModel.predict(testFeatures)
    probabilities = classifierModel.predict_proba(testFeatures)
    classeLabels = list(classifierModel.classes_)
    return {
        "Accuracy": float(accuracy_score(testLabels, predicted)),
        "AUC": float(roc_auc_score(testLabels, probabilities, labels=classeLabels, multi_class="ovr", average="macro")),
        "Precision": float(precision_score(testLabels, predicted, average="macro", zero_division=0)),
        "Recall": float(recall_score(testLabels, predicted, average="macro", zero_division=0)),
        "F1": float(f1_score(testLabels, predicted, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(testLabels, predicted)),
    }

# Main training loop for all classifiers.
def Main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    data = LoadDataset()
    featureMatrix = data[FEATURE_COLUMNS]
    targetLabels = data[TARGET_COLUMN]

    trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(
        featureMatrix, targetLabels, test_size=0.20, stratify=targetLabels, random_state=RANDOM_STATE
    )

    # Pipelines retain the scaler fitted solely on training rows, so the app can
    # apply identical preprocessing to new test data without retraining.
    classifierModels = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE))]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", C=8.0, probability=True, random_state=RANDOM_STATE, cache_size=1000))]
        ),
    }

    modelPerformanceMetrics: list[dict[str, float | str]] = []
    for displayName, model in classifierModels.items():
        print(f"Training {displayName}...")
        model.fit(trainingFeatures, trainingLabels)
        metrics = EvaluateModel(model, testFeatures, testLabels)
        modelPerformanceMetrics.append({"ML Model Name": displayName, **metrics})
        artifact_name = displayName.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / artifact_name)

    metricsFrame = pd.DataFrame(modelPerformanceMetrics)
    metricsFrame.to_csv(ROOT / "metrics.csv", index=False, float_format="%.4f")

    testDataWithLabels = testFeatures.copy()
    testDataWithLabels[TARGET_COLUMN] = testLabels
    testDataWithLabels.to_csv(ROOT / "test_data.csv", index=False)

    metadata = {
        "dataset_name": "Letter Recognition",
        "dataset_source": "https://archive.ics.uci.edu/dataset/59/letter+recognition",
        "instances": int(len(data)),
        "features": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": sorted(targetLabels.unique().tolist()),
        "train_rows": int(len(trainingFeatures)),
        "test_rows": int(len(testFeatures)),
        "random_state": RANDOM_STATE,
        "metric_averaging": "macro; AUC is one-vs-rest (OvR) macro average",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTraining completed. Metrics written to metrics.csv")
    print(metricsFrame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    Main()
