"""Interactive evaluator for the five saved Letter Recognition classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_data
def LoadMetadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def LoadModel(model_file: str):
    return joblib.load(MODEL_DIR / model_file)


def CalculateMetrics(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], pd.Series, pd.DataFrame]:
    """Evaluate an already fitted model; this function deliberately never calls fit."""
    predictions = pd.Series(model.predict(X), index=X.index, name="predicted_letter")
    class_labels = list(model.classes_)
    probability_frame = pd.DataFrame(model.predict_proba(X), columns=class_labels, index=X.index)
    present_labels = sorted(y.unique().tolist())

    if len(present_labels) < 2:
        raise ValueError("AUC requires at least two target classes in the uploaded CSV.")
    if len(present_labels) == 2:
        positive_label = present_labels[1]
        auc = roc_auc_score((y == positive_label).astype(int), probability_frame[positive_label])
    else:
        auc = roc_auc_score(
            y,
            probability_frame[present_labels],
            labels=present_labels,
            multi_class="ovr",
            average="macro",
        )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC (OvR macro)": auc,
        "Precision (macro)": precision_score(y, predictions, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, predictions, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return metrics, predictions, probability_frame

def ApplyClinicalStyling() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #FFFFFF;
        }

        [data-testid="stSidebar"] {
            background: #F3F4F4;
            border-right: 1px solid #D9D9D9;
        }

        [data-testid="stSidebar"] * {
            color: #1A1A1A;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border-top: 4px solid #E66B00;
            border-radius: 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            padding: 18px;
        }

        [data-testid="stMetricLabel"] {
            color: #4A4A4A;
            font-size: 0.9rem;
        }

        [data-testid="stMetricValue"] {
            color: #111111;
            font-weight: 700;
        }

        .clinical-nav {
            background: #050505;
            color: #FFFFFF;
            padding: 20px 30px;
            margin: -4rem -4rem 2.5rem -4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .clinical-title {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .clinical-title span {
            color: #E66B00;
        }

        .clinical-meta {
            color: #FFFFFF;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
        }

        .hero-label {
            color: #008C95;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
        }

        .hero-title {
            color: #171717;
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0.35rem 0 0.5rem 0;
        }

        .hero-description {
            color: #555555;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        h2, h3 {
            color: #1A1A1A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def Main() -> None:
    st.set_page_config(page_title="Letter Recognition Evaluator", page_icon="✉️", layout="wide")
    ApplyClinicalStyling()
    metadata = LoadMetadata()

    st.markdown(
    """
    <div class="clinical-nav">
        <div class="clinical-title">
            <span>Letter Recognition Evaluator</span>
        </div>
        <div class="clinical-meta">
            MACHINE LEARNING ASSIGNMENT 2
        </div>
    </div>

    <div class="hero-label">MODEL EVALUATION WORKSPACE</div>
    <div class="hero-title">Understand every letter prediction.</div>
    <div class="hero-description">
        Compare saved classification models using reproducible held-out test data.
    </div>
    """,
    unsafe_allow_html=True,
    )
    st.info(
        f"Dataset: {metadata['dataset_name']} - {metadata['instances']:,} rows, "
        f"{metadata['features']} input features, and {len(metadata['class_labels'])} letter classes (A-Z)."
    )

    with st.sidebar:
        st.header("Evaluation settings")
        selected_model_name = st.selectbox("Choose a saved classifier", list(MODEL_FILES))
        use_sample = st.checkbox("Use included held-out test data", value=True)
        uploaded_file = st.file_uploader("Or upload a labelled test CSV", type=["csv"], disabled=use_sample)

    if use_sample:
        data = pd.read_csv(ROOT / "test_data.csv")
        st.success(f"Using the included held-out test set ({len(data):,} rows).")
    elif uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Choose the included test set or upload a labelled CSV to evaluate a model.")
        st.stop()

    target_column = metadata["target_column"]
    required_features = metadata["feature_columns"]
    required_columns = [*required_features, target_column]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        st.error("The test CSV is missing required column(s): " + ", ".join(missing_columns))
        st.stop()

    X_test = data[required_features].copy()
    y_test = data[target_column].astype(str).str.strip().str.upper()
    valid_labels = set(metadata["class_labels"])
    unknown_labels = sorted(set(y_test) - valid_labels)
    if unknown_labels:
        st.error("The target column has unknown label(s): " + ", ".join(unknown_labels))
        st.stop()
    if X_test.isna().any().any():
        st.error("The test features contain missing values. Please upload a complete CSV.")
        st.stop()

    model = LoadModel(MODEL_FILES[selected_model_name])
    try:
        metrics, predictions, probabilities = CalculateMetrics(model, X_test, y_test)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.subheader(f"Results: {selected_model_name}")
    metric_columns = st.columns(6)
    for column, (name, value) in zip(metric_columns, metrics.items()):
        column.metric(name, f"{value:.4f}")

    left, right = st.columns([1.25, 1])
    class_labels = metadata["class_labels"]
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(y_test, predictions, labels=class_labels)
        figure, axis = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            matrix,
            cmap="Blues",
            cbar=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axis,
            square=True,
        )
        axis.set_xlabel("Predicted letter")
        axis.set_ylabel("Actual letter")
        st.pyplot(figure, clear_figure=True)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_test,
            predictions,
            labels=class_labels,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True, height=700)

    st.subheader("Evaluation data preview")
    preview = data.copy()
    preview["predicted_letter"] = predictions
    preview["prediction_confidence"] = probabilities.max(axis=1).round(4)
    st.dataframe(preview.head(30), use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        preview.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    Main()
