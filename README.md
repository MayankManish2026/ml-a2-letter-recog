# ML Assignment 2 – Letter Recognition Classifier

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to recognize English capital letters (A–Z) using numerical features extracted from black-and-white character images.

The project uses the UCI Letter Recognition dataset and implements five classification algorithms on the same dataset. Their performance is evaluated using six standard classification metrics.

An interactive Streamlit web application is developed to demonstrate the trained models, compare their performance, evaluate test data, and display prediction results without retraining the saved models.

## 2. Dataset Description

**Dataset:** UCI Letter Recognition Dataset

**Source:** https://archive.ics.uci.edu/dataset/59/letter+recognition

The dataset contains numerical attributes describing the shape and characteristics of capital letters.

| Dataset Attribute | Description |
|---|---|
| Total instances | 20,000 |
| Input features | 16 numerical features |
| Target classes | 26 capital letters (A–Z) |
| Target column | `letter` |
| Missing values | None |
| Problem type | Multiclass classification |

Each record represents a letter image described through 16 numerical features, including bounding-box position, width, height, pixel characteristics, statistical moments, and edge-count information.

The original dataset is included in the repository as:

`letter_recognition_raw.csv`

### Data Preparation

The dataset was divided into training and held-out test datasets using an 80:20 stratified split.

- Training records: 16,000
- Test records: 4,000
- Random state: 42

Stratification ensures that all 26 letter classes are proportionally represented in both datasets.

The same held-out test dataset is used to evaluate all five classifiers, ensuring a consistent comparison.

### Model Preprocessing

Feature scaling is applied to Logistic Regression, K-Nearest Neighbors, and Support Vector Machine using `StandardScaler` within scikit-learn pipelines.

The scaler is fitted on the training data and retained with the trained model so that the same preprocessing is applied during prediction.

Decision Tree and Gaussian Naive Bayes are trained directly on the numerical features without feature scaling.

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/MayankManish2026/ml-a2-letter-recog

The public repository contains:

- Complete Python source code.
- Streamlit application (`app.py`).
- Model training script (`train_models.py`).
- Reproducibility verification script (`verify_project.py`).
- Requirements file (`requirements.txt`).
- Original dataset (`letter_recognition_raw.csv`).
- Held-out test dataset (`test_data.csv`).
- Evaluation results (`metrics.csv`).
- Five saved trained model files in the `models/` directory.
- Project documentation (`README.md`).

## 4. Models Used and Performance Evaluation

The following five classification algorithms were implemented using the UCI Letter Recognition dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

### Evaluation Metrics

All five models were evaluated using the same held-out test dataset and the following six metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the model's ability to distinguish between classes. The multiclass AUC is calculated using the One-vs-Rest strategy with macro averaging.
- **Precision:** Measures the proportion of correct positive predictions among all positive predictions. Macro averaging is used.
- **Recall:** Measures the proportion of actual positive instances correctly identified. Macro averaging is used.
- **F1 Score:** Harmonic mean of precision and recall, calculated using macro averaging.
- **Matthews Correlation Coefficient (MCC):** Measures the correlation between actual and predicted classifications.

Macro averaging gives equal importance to all 26 letter classes.

### Model Comparison Results

The following table presents the evaluation results obtained from the five trained classifiers.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7742 | 0.9805 | 0.7748 | 0.7730 | 0.7728 | 0.7653 |
| Decision Tree | 0.8562 | 0.9544 | 0.8592 | 0.8561 | 0.8563 | 0.8506 |
| K-Nearest Neighbors | 0.9500 | 0.9965 | 0.9506 | 0.9498 | 0.9499 | 0.9480 |
| Gaussian Naive Bayes | 0.6522 | 0.9573 | 0.6641 | 0.6512 | 0.6479 | 0.6391 |
| Support Vector Machine | **0.9748** | **0.9995** | **0.9748** | **0.9746** | **0.9746** | **0.9737** |

All values are reported to four decimal places.

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Provides a useful linear baseline with an accuracy of 77.42%. Its performance is comparatively lower because the relationships between numerical letter-shape features and letter classes can be nonlinear. |
| Decision Tree | Achieves an accuracy of 85.62%, improving upon Logistic Regression. It captures nonlinear decision boundaries through feature-based splits but does not perform as well as KNN and SVM on this dataset. |
| K-Nearest Neighbors | Achieves 95.00% accuracy and performs strongly because similar letter shapes tend to have nearby representations in the scaled feature space. Its performance is substantially better than the linear and tree-based baseline models. |
| Gaussian Naive Bayes | Achieves 65.22% accuracy, the lowest among the five models. Its conditional independence assumption is restrictive for correlated numerical features describing letter shapes. |
| Support Vector Machine | Achieves the highest accuracy of 97.48% and the highest AUC of 0.9995. The RBF kernel effectively captures nonlinear class boundaries and provides the best overall classification performance among the evaluated models. |

### Overall Winner

**Support Vector Machine (SVM)** is the overall best-performing model for the UCI Letter Recognition dataset.

It achieves the highest value across all six evaluation metrics:

- Accuracy: 0.9748
- AUC: 0.9995
- Precision: 0.9748
- Recall: 0.9746
- F1 Score: 0.9746
- MCC: 0.9737

The SVM model demonstrates strong classification performance across all 26 letter classes and consistently outperforms the other four classifiers.

The Streamlit application also calculates the average rank of each model across the six evaluation metrics. Support Vector Machine achieves an average rank of **1.00**, confirming its first-place ranking.

## 5. Saved Training Outputs

The `train_models.py` script trains the five classifiers and saves their trained model artifacts.

The saved outputs include:

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

The `.joblib` files contain the trained classifiers. For Logistic Regression, K-Nearest Neighbors, and Support Vector Machine, the saved artifacts also contain the fitted feature scaler.

The saved models are loaded directly by the Streamlit application. The application does not retrain the classifiers when evaluating uploaded test data.

## 6. Project Structure

The GitHub repository is organized as follows:

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
│
├── .streamlit/
│   └── config.toml
│
└── models/
    ├── decision_tree.joblib
    ├── gaussian_naive_bayes.joblib
    ├── k_nearest_neighbors.joblib
    ├── logistic_regression.joblib
    ├── support_vector_machine.joblib
    └── metadata.json
```

## 7. Setup and Execution

### Python Environment

The project dependencies are listed in `requirements.txt`.

The project was executed and verified in the BITS Virtual Lab using Python 3.12.7. Streamlit Community Cloud deployed the application using Python 3.12.16.

### Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on Linux:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Models

To reproduce the training process:

```bash
python train_models.py
```

This generates the saved model artifacts and evaluation results.

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### Reproducibility Verification

Run the verification script:

```bash
python verify_project.py
```

The verification script loads each saved classifier and checks whether its predictions on the held-out test data reproduce the recorded evaluation metrics.

The BITS Virtual Lab verification completed successfully:

```text
PASS - Logistic Regression
PASS - Decision Tree
PASS - K-Nearest Neighbors
PASS - Gaussian Naive Bayes
PASS - Support Vector Machine

All five saved models reproduce the recorded held-out metrics.
No model was retrained.
```

This confirms that all five saved model artifacts reproduce the recorded results without retraining.

## 8. Streamlit Web Application

The interactive web application is developed using Streamlit.

**Live Application:**

https://letter-recognition-mayankmanish.streamlit.app/

### Application Features

The application provides the following functionality:

1. **Dataset Evaluation:** Evaluates the included 4,000-row held-out test dataset.
2. **CSV Upload:** Allows users to upload a labelled test CSV containing all 16 input features and the target column `letter`.
3. **Model Selection:** Provides a dropdown to select any of the five saved classification models.
4. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
5. **Confusion Matrix:** Displays the confusion matrix for the selected model.
6. **Classification Report:** Displays class-wise precision, recall, F1 Score, and support.
7. **Model Comparison:** Compares all five saved models using the same held-out test dataset and all six evaluation metrics.
8. **Prediction Preview:** Displays the first 30 prediction records along with predicted letters and prediction confidence.
9. **CSV Download:** Allows users to download all 4,000 prediction records as a CSV file.

The application loads the previously saved trained models from the `models/` directory. It does not retrain or refit any model when a test CSV is uploaded.

## 9. Streamlit Community Cloud Deployment

The application is deployed on Streamlit Community Cloud using the public GitHub repository.

**Deployment Configuration:**

| Setting | Value |
|---|---|
| Repository | `MayankManish2026/ml-a2-letter-recog` |
| Branch | `main` |
| Main file | `app.py` |
| Python version | 3.12 |

**Live Deployment URL:**

https://letter-recognition-mayankmanish.streamlit.app/

The deployed application successfully installed its dependencies and started the Streamlit server.

## 10. Conclusion

This project implements and evaluates five machine learning classification algorithms on the UCI Letter Recognition dataset.

The comparison demonstrates the differences in classification performance between linear, tree-based, probabilistic, distance-based, and kernel-based approaches.

Among the five evaluated models, Support Vector Machine achieves the best overall performance, with an accuracy of 97.48% and an AUC score of 0.9995.

The project also demonstrates an end-to-end machine learning workflow, including dataset preparation, model training, performance evaluation, model persistence, reproducibility verification, interactive application development, GitHub version control, and deployment on Streamlit Community Cloud.