# ML Assignment 2 – Letter Recognition Classifier

## 1. Problem Statement

The objective of this project is to implement, evaluate, and compare five supervised machine learning classification algorithms using the UCI Letter Recognition dataset.

The classification task is to predict the correct English capital letter (A–Z) from 16 numerical shape features extracted from black-and-white character images.

The five implemented classification algorithms are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

All five models are trained and evaluated using the same dataset and held-out test data to ensure a consistent comparison.

An interactive Streamlit web application is developed to evaluate saved classifiers, visualize their predictions, compare model performance, and identify the best-performing model.

## 2. Dataset Description

**Dataset:** UCI Letter Recognition Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/59/letter+recognition)

| Property | Description |
|---|---|
| Total instances | 20,000 |
| Input features | 16 numerical features |
| Target classes | 26 English capital letters (A–Z) |
| Target column | `letter` |
| Missing values | None |
| Training instances | 16,000 (80%) |
| Test instances | 4,000 (20%) |

Each dataset record represents one capital letter. The original character image is converted into numerical shape features describing bounding-box position, width, height, active-pixel characteristics, statistical moments, and edge-count information.

The source dataset is stored in:

`letter_recognition_raw.csv`

The dataset satisfies the assignment requirements of at least 12 input features and 500 instances.

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/MayankManish2026/ml-a2-letter-recog

The repository contains the complete Python source code, dependency requirements, saved trained models, test dataset, evaluation results, and project documentation.

## 4. Models Used and Performance Evaluation

### 4.1 Data Preparation

The complete dataset contains 20,000 records.

The dataset is divided into:

- **Training data:** 16,000 records (80%)
- **Held-out test data:** 4,000 records (20%)

A stratified train-test split is used to maintain proportional representation of all 26 letter classes in both training and test sets.

The random state is fixed at `42` to make the experiment reproducible.

### 4.2 Model Preprocessing

The following models use `StandardScaler` inside a scikit-learn pipeline:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

Feature scaling is fitted only on the training dataset. The fitted preprocessing pipeline is subsequently applied to test data, preventing data leakage.

Decision Tree and Gaussian Naive Bayes are trained directly on the numerical input features without feature scaling.

The trained classifiers and their required preprocessing steps are saved as Joblib artifacts for subsequent evaluation and deployment.

### 4.3 Evaluation Metrics

All five models are evaluated using the following six metrics:

| Evaluation Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified instances |
| AUC Score | Area Under the ROC Curve using One-vs-Rest macro averaging |
| Precision | Macro-averaged precision across all 26 classes |
| Recall | Macro-averaged recall across all 26 classes |
| F1 Score | Macro-averaged harmonic mean of precision and recall |
| MCC Score | Matthews Correlation Coefficient |

Macro averaging gives equal importance to each letter class.

### 4.4 Model Comparison Table

The following table presents the evaluation results obtained from the same 4,000-record held-out test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7742 | 0.9805 | 0.7748 | 0.7730 | 0.7728 | 0.7653 |
| Decision Tree | 0.8562 | 0.9544 | 0.8592 | 0.8561 | 0.8563 | 0.8506 |
| K-Nearest Neighbors | 0.9500 | 0.9965 | 0.9506 | 0.9498 | 0.9499 | 0.9480 |
| Gaussian Naive Bayes | 0.6522 | 0.9573 | 0.6641 | 0.6512 | 0.6479 | 0.6391 |
| **Support Vector Machine** | **0.9748** | **0.9995** | **0.9748** | **0.9746** | **0.9746** | **0.9737** |

### 4.5 Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved 77.42% accuracy. It provides a useful linear classification baseline but has limitations in capturing complex nonlinear relationships between visually similar letter shapes. |
| **Decision Tree** | Achieved 85.62% accuracy. It improves upon Logistic Regression by learning nonlinear decision rules, although its performance is lower than the distance-based and kernel-based classifiers. |
| **K-Nearest Neighbors** | Achieved 95.00% accuracy. Its strong performance indicates that similar letter shapes tend to occupy nearby regions in the scaled feature space. |
| **Gaussian Naive Bayes** | Achieved 65.22% accuracy, the lowest among the five classifiers. Its conditional independence assumption is less suitable for correlated letter-shape features. |
| **Support Vector Machine** | Achieved the highest accuracy of 97.48% and AUC of 0.9995. The RBF kernel effectively captures complex nonlinear relationships, resulting in the best overall classification performance. |

### 4.6 Overall Winner

**Support Vector Machine (SVM) is the overall best-performing classification model for the selected dataset.**

It achieved the highest score across all six evaluation metrics:

| Metric | SVM Result |
|---|---:|
| Accuracy | 0.9748 |
| AUC Score | 0.9995 |
| Precision | 0.9748 |
| Recall | 0.9746 |
| F1 Score | 0.9746 |
| MCC Score | 0.9737 |
| Average Rank | 1.00 |

The Streamlit application ranks the five models using all six evaluation metrics, with lower average rank indicating better overall performance.

SVM achieved an average rank of 1.00, ranking first across all six metrics.

K-Nearest Neighbors achieved the second-best overall performance with 95.00% accuracy.

Therefore, **Support Vector Machine is selected as the overall winner on the held-out test dataset**.

## 5. Saved Training Outputs

Running `train_models.py` generates the following model artifacts and evaluation files:

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

The `.joblib` files contain the trained classification models.

For Logistic Regression, KNN, and SVM, the saved artifacts also contain their fitted feature-scaling pipelines.

The `metadata.json` file contains information required to load and evaluate the models.

The `metrics.csv` file stores the calculated model evaluation results.

The `test_data.csv` file contains the held-out test dataset used for model evaluation.

## 6. Project Structure

```text
ml-a2-letter-recog/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   ├── decision_tree.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── logistic_regression.joblib
│   ├── support_vector_machine.joblib
│   └── metadata.json
│
├── .gitignore
├── README.md
├── requirements.txt
├── letter_recognition_raw.csv
├── app.py
├── train_models.py
├── verify_project.py
├── metrics.csv
└── test_data.csv
```

## 7. Local Setup and Execution

### 7.1 Prerequisites

- Python
- pip
- Git

### 7.2 Clone the GitHub Repository

```bash
git clone https://github.com/MayankManish2026/ml-a2-letter-recog.git
cd ml-a2-letter-recog
```

### 7.3 Create a Virtual Environment

```powershell
python -m venv .venv
```

### 7.4 Install Required Dependencies

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All application dependencies are specified in `requirements.txt`.

### 7.5 Train the Classification Models

To regenerate the five trained classifiers and their evaluation outputs:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

Training is optional when using the already saved model artifacts.

### 7.6 Run the Streamlit Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application opens in a web browser, typically at:

http://localhost:8501

## 8. Reproducibility Verification

Run the verification script:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```

The verification script:

- Loads all five saved trained classifiers.
- Evaluates them using `test_data.csv`.
- Recalculates the evaluation metrics.
- Compares the results with `metrics.csv`.
- Confirms reproducibility without retraining the models.

This ensures that the saved model artifacts reproduce the recorded evaluation results.

## 9. Python Environment

The project has been tested locally using Python 3.14.

Execution and validation in the BITS Virtual Lab environment using Python 3.12 are planned as part of the final assignment submission.

The application uses dependencies specified in `requirements.txt` to support reproducible installation.

## 10. Streamlit Web Application

The project includes an interactive Streamlit web application implemented in `app.py`.

The application loads saved classification models and evaluates their predictions without retraining them.

### 10.1 Model Selection

Users can select any of the five saved classifiers from the sidebar dropdown.

### 10.2 Dataset Selection and CSV Upload

The application supports:

- Evaluation using the included 4,000-record held-out test dataset.
- Uploading a labelled test CSV containing the `letter` target column and all 16 required numerical input features.

Uploaded test data is used only for evaluation, not model training.

### 10.3 Individual Model Evaluation

For the selected model, the application displays:

- Accuracy
- AUC Score (OvR macro)
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Matthews Correlation Coefficient

### 10.4 Confusion Matrix

A confusion matrix visualizes actual versus predicted letter classes across all 26 categories (A–Z).

### 10.5 Classification Report

The application displays class-wise:

- Precision
- Recall
- F1 Score
- Support

It also displays overall accuracy, macro-average and weighted-average results.

### 10.6 Test Data Prediction Preview

The application displays the first 30 test records with:

- Numerical input features
- Actual letter
- Predicted letter
- Prediction confidence

This allows users to inspect individual classification predictions.

### 10.7 Prediction CSV Download

Users can download the complete evaluation dataset with predicted letters and prediction confidence.

For the included held-out test dataset, the CSV export contains all 4,000 records, while the on-screen preview displays only the first 30.

### 10.8 All-Model Performance Comparison

The application compares all five trained classification models using the same held-out test dataset.

The comparison table displays:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- MCC Score
- Average Rank

The highest metric values and the lowest average rank are highlighted to make model comparison easier.

### 10.9 Overall Best-Performing Model

The application automatically identifies the best-performing model based on its average ranking across all six evaluation metrics.

For the included held-out dataset, **Support Vector Machine (SVM)** is identified as the winner, with:

- Accuracy: 0.9748
- Average Rank: 1.00

## 11. Streamlit Community Cloud Deployment

The application is intended to be deployed using Streamlit Community Cloud.

**GitHub Repository:**

https://github.com/MayankManish2026/ml-a2-letter-recog

**Application Entry Point:** `app.py`

**Deployment Branch:** `main`

**Live Streamlit Application URL:** To be updated after successful deployment.

The deployed application will use the same saved models, source code, and dependencies as the GitHub repository.

## 12. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, including:

- Dataset preparation and preprocessing
- Training five supervised classification models
- Evaluating six performance metrics
- Comparing model performance on held-out test data
- Saving and reusing trained model artifacts
- Verifying reproducibility
- Developing an interactive Streamlit application
- Preparing the application for cloud deployment

Among the five evaluated classifiers, **Support Vector Machine achieved the best overall performance**, with an accuracy of 97.48% and the highest score across all six evaluation metrics.

The Streamlit application provides an interactive interface for evaluating individual models, inspecting predictions, and comparing all five classifiers using consistent evaluation criteria.