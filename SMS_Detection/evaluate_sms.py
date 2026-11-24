
import joblib
import pandas as pd

from pathlib import Path

# Import metrics used to evaluate the SMS classification model.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

# Get the directory containing this evaluation script.
BASE_DIR = Path(__file__).resolve().parent

# Define the test dataset and saved model file locations.
DATA_FILE = BASE_DIR / "data" / "sms_test_cases.tsv"
MODEL_FILE = BASE_DIR / "models" / "sms_model.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

# Load the previously trained machine learning model.
model = joblib.load(MODEL_FILE)

# Load the TF-IDF vectorizer used during training.
# The same vectorizer must be used for new SMS messages.
vectorizer = joblib.load(VECTORIZER_FILE)


# --------------------------------------------------
# 3. LOAD AND VALIDATE TEST DATA
# --------------------------------------------------

# Load the separate test dataset containing actual labels
# and SMS messages.
data = pd.read_csv(
    DATA_FILE,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Clean labels and message text.
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].fillna("").astype(str)

# Check that the test dataset contains messages.
if data.empty:
    raise ValueError("The test dataset is empty.")

# Ensure that every label is either ham or spam.
if not data["label"].isin(["ham", "spam"]).all():
    raise ValueError("Labels must be either 'ham' or 'spam'.")

# Ensure that no test message is empty.
if data["message"].str.strip().eq("").any():
    raise ValueError("One or more test messages are empty.")

# Display basic information about the test dataset.
print("SMS DETECTOR — NEW TEST DATA EVALUATION")
print("=" * 50)
print("Test messages:", len(data))
print("\nActual label counts:")
print(data["label"].value_counts())


# --------------------------------------------------
# 4. CONVERT SMS MESSAGES INTO TF-IDF FEATURES
# --------------------------------------------------

# Transform the test messages into numerical features
# using the already fitted TF-IDF vectorizer.
# No retraining or vocabulary fitting is performed here.
X_test = vectorizer.transform(data["message"])

# Store the actual labels for comparison.
y_true = data["label"]


# --------------------------------------------------
# 5. PREDICT SPAM PROBABILITIES
# --------------------------------------------------

# Find the position of the spam class in the model's class list.
spam_index = list(model.classes_).index("spam")

# Calculate the estimated probability of spam for each message.
spam_scores = model.predict_proba(X_test)[:, spam_index]


# --------------------------------------------------
# 6. GENERATE PREDICTIONS AT DIFFERENT THRESHOLDS
# --------------------------------------------------

# Create a copy of the test dataset to store prediction results.
results = data.copy()

# Add the estimated spam probability for each message.
results["spam_score"] = spam_scores

# Compare classification results at three probability thresholds.
# A message is classified as spam when its score is greater than
# or equal to the selected threshold.
for threshold in [0.50, 0.60, 0.70]:
    results[f"prediction_{threshold:.2f}"] = [
        "spam" if score >= threshold else "ham"
        for score in spam_scores
    ]


# --------------------------------------------------
# 7. DISPLAY INDIVIDUAL PREDICTIONS
# --------------------------------------------------

print("\nINDIVIDUAL PREDICTIONS")
print("=" * 50)

# Display the actual label, message, spam score and predictions
# produced using each threshold.
for _, row in results.iterrows():
    print(f"\nActual: {row['label'].upper()}")
    print(f"Message: {row['message']}")
    print(f"Estimated spam score: {row['spam_score']:.2%}")
    print(f"Prediction at 0.50: {row['prediction_0.50'].upper()}")
    print(f"Prediction at 0.60: {row['prediction_0.60'].upper()}")
    print(f"Prediction at 0.70: {row['prediction_0.70'].upper()}")


# --------------------------------------------------
# 8. COMPARE DIFFERENT THRESHOLDS
# --------------------------------------------------

print("\n\nTHRESHOLD COMPARISON")
print("=" * 70)

# Evaluate the predictions produced at each threshold.
for threshold in [0.50, 0.60, 0.70]:

    # Get predictions corresponding to the current threshold.
    predictions = results[f"prediction_{threshold:.2f}"]

    # Generate the confusion matrix.
    # Label order: ham, spam.
    #
    # tn = True Negatives
    # fp = False Positives
    # fn = False Negatives
    # tp = True Positives
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=["ham", "spam"]
    ).ravel()

    # Calculate spam precision and recall.
    # Return zero if the denominator is zero.
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    # Display the evaluation metrics.
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])

    # Display the detailed classification report.
    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    ))


# --------------------------------------------------
# 9. SAVE EVALUATION RESULTS
# --------------------------------------------------

# Define the output CSV file path.
output_file = BASE_DIR / "sms_test_results.csv"

# Save individual messages, actual labels, spam scores
# and predictions at different thresholds.
results.to_csv(output_file, index=False)

print("\n" + "=" * 50)
print("Evaluation complete.")
print(f"Detailed results saved to: {output_file}")
print("The model was evaluated without retraining.")
