
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

DATA_DIR = Path(".")
MODEL_PATH = DATA_DIR / "network_threat_model.joblib"
TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"

print("Loading training data...")
data = pd.read_csv(TRAIN_PATH)

X = data.drop(
    columns=["label", "attack_cat", "id"],
    errors="ignore"
)
y = data["label"].astype(int)

# Split training data into fitting and validation portions.
# The supplied official test set is not used here.
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Loading saved baseline model...")
model = joblib.load(MODEL_PATH)

print("Fitting a validation-only model...")
model.fit(X_fit, y_fit)

print("Scoring validation records...")
attack_scores = model.predict_proba(X_val)[:, 1]

print("\nTHRESHOLD COMPARISON")
print(
    f"{'Threshold':>10} {'Accuracy':>10} "
    f"{'Precision':>11} {'Recall':>10} "
    f"{'F1':>10} {'False alarms':>14}"
)

results = []

for threshold in np.arange(0.30, 0.801, 0.05):
    predictions = (attack_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_val, predictions)

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms": int(fp),
        "false_alarm_rate": fp / (tn + fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })

    print(
        f"{threshold:10.2f} {accuracy:10.4f} "
        f"{precision:11.4f} {recall:10.4f} "
        f"{f1:10.4f} {fp:14d}"
    )

# Example selection rule: maximize F1 among thresholds
# that retain at least 95% attack recall.
eligible = [r for r in results if r["recall"] >= 0.95]

if eligible:
    best = max(eligible, key=lambda r: r["f1"])
    print("\nRecommended validation threshold:")
    print(best)
else:
    print(
        "\nNo tested threshold achieved 95% recall. "
        "Review the table before choosing a threshold."
    )

print("\nThis script does not overwrite the saved baseline model.")
