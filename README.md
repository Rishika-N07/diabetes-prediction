# 🩺 Diabetes Prediction System using Machine Learning

## 📌 Project Overview

This project predicts whether a person is likely to have diabetes based on their health-related information.

The project uses Machine Learning classification techniques and a Streamlit web application to provide predictions.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can classify patients into:

- 0 → No Diabetes
- 1 → Diabetes

## 📊 Dataset

The project uses the Pima Indians Diabetes Dataset.

The dataset contains 768 patient records and 8 input features.

### Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target

- Outcome

## 🤖 Machine Learning Model

The following models were explored:

- Logistic Regression
- Decision Tree
- Random Forest

After model comparison and hyperparameter tuning, Logistic Regression was selected as the final model.

### Hyperparameter Tuning

Best C value:

**C = 1**

Cross-validation accuracy:

**79.31%**

Final test accuracy:

**70.78%**

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- GitHub

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Web Application
   ↓
Prediction
