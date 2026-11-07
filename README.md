# ML Assignment 2 - Letter Recognition Classifier

## Objective

This project compares five machine learning classifiers on the UCI Letter Recognition dataset.

The objective is to predict the correct capital letter, from `A` to `Z`, using 16 numeric shape features extracted from black-and-white character images.

The implemented classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

## Dataset

This project uses the UCI Letter Recognition dataset.

- Total instances: 20,000
- Input features: 16 numeric features
- Target classes: 26 capital letters (`A` to `Z`)
- Target column: `letter`
- Missing values: None

Each row represents one letter image. The original image is converted into numeric shape features, including bounding-box position, width, height, active-pixel characteristics, statistical moments, and edge-count information.

The source data used by this project is:

```text
letter_recognition_raw.csv
```

## Data Preparation

The dataset is divided into:

```text
80% training data
20% held-out test data
```

A stratified split is used so that all 26 letters remain proportionally represented in both training and test sets.

The random state is fixed at `42` to make the experiment reproducible.

## Model Preprocessing

Logistic Regression, K-Nearest Neighbors, and Support Vector Machine use `StandardScaler` inside a scikit-learn pipeline.

This ensures that scaling is fitted only on training data and that the same preprocessing is applied when making predictions on test data.

Decision Tree and Gaussian Naive Bayes are trained directly on the numeric features.

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- AUC Score - One-vs-Rest macro average
- Precision - Macro average
- Recall - Macro average
- F1 Score - Macro average
- Matthews Correlation Coefficient (MCC)

Macro averaging gives equal importance to every letter class rather than allowing more common letters to dominate the score.

## Model Comparison Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7742 | 0.9805 | 0.7748 | 0.7730 | 0.7728 | 0.7653 |
| Decision Tree | 0.8562 | 0.9544 | 0.8592 | 0.8561 | 0.8563 | 0.8506 |
| K-Nearest Neighbors | 0.9500 | 0.9965 | 0.9506 | 0.9498 | 0.9499 | 0.9480 |
| Gaussian Naive Bayes | 0.6522 | 0.9573 | 0.6641 | 0.6512 | 0.6479 | 0.6391 |
| Support Vector Machine | 0.9748 | 0.9995 | 0.9748 | 0.9746 | 0.9746 | 0.9737 |

## Observations

- Logistic Regression provides a useful linear baseline but cannot fully capture the nonlinear differences between visually similar letters.
- Decision Tree improves performance by learning nonlinear decision rules, although its threshold-based splits are less flexible than distance-based or kernel-based approaches.
- K-Nearest Neighbors performs strongly because similar letter shapes tend to occupy nearby regions in the scaled feature space.
- Gaussian Naive Bayes performs weakest because its assumption that all input features are conditionally independent is not realistic for correlated image-shape features.
- Support Vector Machine is the best-performing model across all reported metrics. The RBF kernel effectively separates complex and nonlinear letter-shape patterns.

## Saved Training Outputs

After running `train_models.py`, the project creates:

```text
models/
    decision_tree.joblib
    gaussian_naive_bayes.joblib
    k_nearest_neighbors.joblib
    logistic_regression.joblib
    support_vector_machine.joblib
    metadata.json

metrics.csv
test_data.csv
```

The `.joblib` files contain the trained classifiers. For Logistic Regression, KNN, and SVM, the saved files also contain the fitted feature scaler.

## Project Structure

```text
ml-a2-letter-recog/
│
├── .gitignore
├── README.md
├── requirements.txt
├── letter_recognition_raw.csv
├── app.py
├── train_models.py
├── verify_project.py
├── metrics.csv
├── test_data.csv
├── streamlit/
│   ├── config.toml # Light clinical visual theme
│
└── models/
    ├── decision_tree.joblib
    ├── gaussian_naive_bayes.joblib
    ├── k_nearest_neighbors.joblib
    ├── logistic_regression.joblib
    ├── support_vector_machine.joblib
    └── metadata.json
```

## Local Setup

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the five classifiers:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

## Reproducibility Verification

Run:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```

This script loads each saved trained model and confirms that its predictions on `test_data.csv` reproduce the values in `metrics.csv`. It does not retrain any model.

## Python Version

This project has been tested locally with Python 3.14.

Final training and validation will also be performed in the BITS Virtual Lab environment using Python 3.12.

## Streamlit application

The Streamlit application evaluates already saved classifiers. It does not train or refit a model when a CSV is uploaded.

Features:

- Select any of the five saved classifiers.
- Evaluate the included 4,000-row held-out test dataset.
- Upload a labelled CSV containing the `letter` column and all 16 feature columns.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display a confusion matrix and classification report.
- Load saved `.joblib` model artifacts from the `models/` folder.