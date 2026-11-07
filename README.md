# ML Assignment 2 - Letter Recognition Classifier

## Objective

This project implements a multiclass classification workflow using the UCI Letter Recognition dataset.

The objective is to predict the correct capital letter, from `A` to `Z`, using 16 numeric features extracted from black-and-white letter images.

The project will compare the following five machine-learning classifiers on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

## Dataset

This project uses the UCI Letter Recognition dataset.

- Number of instances: 20,000
- Input features: 16 numeric features
- Target classes: 26 capital letters (`A` to `Z`)
- Target column: `letter`
- Missing values: None

Each row represents one character image. The original image is converted into numerical shape features, including bounding-box position, width, height, active-pixel characteristics, statistical moments, and edge-count information.

The raw dataset used by the training script is:

```text
letter_recognition_raw.csv
```

## First Milestone Scope

This initial version contains the model-training foundation:

- Dataset source file
- Python dependency list
- Training pipeline for the five required classifiers
- Reproducibility verification script
- Git ignore rules

Saved models, held-out test data, metrics, and the Streamlit web interface will be added in later commits.

## Project Structure

```text
ML_Assignment_2/
│
├── .gitignore
├── README.md
├── requirements.txt
├── letter_recognition_raw.csv
├── train_models.py
└── verify_project.py
```

## Training Workflow

The `train_models.py` script performs the following steps:

1. Loads the Letter Recognition dataset.
2. Separates input features from target labels.
3. Creates a stratified train-test split.
4. Trains all five classifiers using the same training data.
5. Calculates Accuracy, AUC, Precision, Recall, F1 Score, and MCC Score.
6. Saves trained-model artifacts and evaluation outputs.

## Local Setup

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run model training:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

## Reproducibility Verification

After the model-training script creates saved models and evaluation files, run:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```

This script loads the saved trained models and confirms that their predictions on the held-out test data reproduce the values stored in the evaluation results. It does not retrain any model.

## Python Version

This project has been tested locally with Python 3.14.

The final training and validation run will also be performed in the BITS Virtual Lab environment using Python 3.12.