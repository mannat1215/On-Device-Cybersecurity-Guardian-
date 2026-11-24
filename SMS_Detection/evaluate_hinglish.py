
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Get the directory containing this Python file.
# This allows the script to locate its data and model files.
BASE_DIR = Path(__file__).resolve().parent

# Define paths for the Hinglish/OTP test dataset and saved model files.
DATA_FILE = BASE_DIR / "data" / "hinglish_otp_test_cases.tsv"
MODEL_FILE = BASE_DIR / "models" / "sms_model.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load the previously trained model and TF-IDF vectorizer.
# No model training takes place in this evaluation script.
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# Load the separate Hinglish and OTP test dataset.
# Each row contains a label (ham/spam) and its corresponding message.
data = pd.read_csv(
    DATA_FILE,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Standardize labels and handle missing message values.
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].fillna("").astype(str)

# Validate the test dataset before performing evaluation.
if data.empty:
    raise ValueError("The test dataset is empty.")

if not data["label"].isin(["ham", "spam"]).all():
    raise ValueError("Labels must be either 'ham' or 'spam'.")

if data["message"].str.strip().eq("").any():
    raise ValueError("One or more test messages are empty.")

# Display basic information about the test dataset.
print("SMS DETECTOR — NEW TEST DATA EVALUATION")
print("=" * 50)
print("Test messages:", len(data))
print("\nActual label counts:")
print(data["label"].value_counts())

# Convert the test messages into numerical TF-IDF features.
# The already-fitted vectorizer is reused without retraining.
X_test = vectorizer.transform(data["message"])
y_true = data["label"]

# Find the position of the spam class in the model's class list.
spam_index = list(model.classes_).index("spam")

# Calculate the estimated probability of each message being spam.
spam_scores = model.predict_proba(X_test)[:, spam_index]

# Create a results table containing the original messages and their scores.
results = data.copy()
results["spam_score"] = spam_scores

# Generate predictions using three different spam probability thresholds.
# A message is classified as spam if its score is greater than or equal
# to the selected threshold.
for threshold in [0.50, 0.60, 0.70]:
    results[f"prediction_{threshold:.2f}"] = [
        "spam" if score >= threshold else "ham"
        for score in spam_scores
    ]

# Display the actual label, estimated spam score and predictions
# for every individual test message.
print("\nINDIVIDUAL PREDICTIONS")
print("=" * 50)

for _, row in results.iterrows():
    print(f"\nActual: {row['label'].upper()}")
    print(f"Message: {row['message']}")
    print(f"Estimated spam score: {row['spam_score']:.2%}")
    print(f"Prediction at 0.50: {row['prediction_0.50'].upper()}")
    print(f"Prediction at 0.60: {row['prediction_0.60'].upper()}")
    print(f"Prediction at 0.70: {row['prediction_0.70'].upper()}")

# Compare model performance at the three selected thresholds.
print("\n\nTHRESHOLD COMPARISON")
print("=" * 70)

for threshold in [0.50, 0.60, 0.70]:
    predictions = results[f"prediction_{threshold:.2f}"]

    # Calculate the confusion matrix.
    # TN = correctly identified ham messages.
    # FP = legitimate ham messages incorrectly classified as spam.
    # FN = spam messages incorrectly classified as ham.
    # TP = correctly identified spam messages.
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=["ham", "spam"]
    ).ravel()

    # Calculate precision and recall from the confusion matrix.
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    # Display the evaluation metrics for the current threshold.
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])

    # Display precision, recall, F1-score and support for each class.
    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    ))

# Save all individual test predictions and spam scores to a CSV file.
# This file can be used for further analysis and project documentation.
output_file = BASE_DIR / "sms_test_results.csv"
results.to_csv(output_file, index=False)

# Display the location of the saved evaluation results.
print("\n" + "=" * 50)
print("Evaluation complete.")
print(f"Detailed results saved to: {output_file}")
print("The model was evaluated without retraining.")
