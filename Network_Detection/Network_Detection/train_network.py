
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Dataset paths
DATA_DIR = Path(".")

train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

print("Loading datasets...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Separate input features and target labels
drop_columns = ["label", "attack_cat", "id"]

X_train = train.drop(
    columns=[c for c in drop_columns if c in train.columns]
)
y_train = train["label"].astype(int)

X_test = test.drop(
    columns=[c for c in drop_columns if c in test.columns]
)
y_test = test["label"].astype(int)

# Identify categorical and numeric features
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_cols = [
    c for c in X_train.columns
    if c not in categorical_cols
]

print("Numeric features:", len(numeric_cols))
print("Categorical features:", categorical_cols)

# Preprocess data inside the pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Build classifier
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("\nEvaluating on the supplied test set...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Attack"],
    digits=4
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4)
)

# Save complete preprocessing + model pipeline
output_path = DATA_DIR / "network_threat_model.joblib"
joblib.dump(model, output_path)

print("\nSaved model to:", output_path)
print("Training complete.")
