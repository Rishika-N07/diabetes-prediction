import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==============================
# 2. CLEAN DATA
# ==============================

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns:
    df[column] = df[column].replace(0, df[column].median())


# ==============================
# 3. SEPARATE FEATURES & TARGET
# ==============================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==============================
# 4. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 5. CREATE PIPELINE
# ==============================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# ==============================
# 6. HYPERPARAMETER TUNING
# ==============================

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)


print("\n==============================")
print("HYPERPARAMETER TUNING")
print("==============================")

print("\nBest C:",
      grid_search.best_params_["model__C"])

print("Best Cross-Validation Accuracy:",
      round(grid_search.best_score_ * 100, 2), "%")


# ==============================
# 7. FINAL TEST EVALUATION
# ==============================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("FINAL TEST EVALUATION")
print("==============================")

print("\nTest Accuracy:",
      round(accuracy * 100, 2), "%")


# ==============================
# 8. CONFUSION MATRIX
# ==============================

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# ==============================
# 9. CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred
))
import joblib

joblib.dump(best_model, "diabetes_model.pkl")

print("Model saved successfully!")