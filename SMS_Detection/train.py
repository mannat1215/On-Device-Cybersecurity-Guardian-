
import pandas as pd
import joblib

from pathlib import Path

# Functions used to divide the dataset into training,
# validation and testing portions.
from sklearn.model_selection import train_test_split

# Converts text messages into numerical TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine learning classifiers.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Evaluation metrics used to measure model performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

# Get the directory containing this training script.
BASE_DIR = Path(__file__).resolve().parent

# Define the folders containing the SMS datasets and model files.
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Define the original SMS dataset and custom Indian SMS examples.
ORIGINAL_FILE = DATA_DIR / "SMSSpamCollection"
CUSTOM_FILE = DATA_DIR / "custom_sms_examples.tsv"

# Create the models folder if it does not already exist.
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND CLEAN ORIGINAL DATA
# --------------------------------------------------

# Load the original SMS dataset.
# Each row contains a label (ham/spam) and a message,
# separated by a tab character.
original_data = pd.read_csv(
    ORIGINAL_FILE,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

# Remove rows containing missing values.
original_data = original_data.dropna()

# Standardize labels by removing extra spaces and
# converting them to lowercase.
original_data["label"] = original_data["label"].str.strip().str.lower()

# Convert messages to strings and remove extra spaces.
original_data["message"] = original_data["message"].astype(str).str.strip()

# Keep only valid ham and spam labels.
original_data = original_data[
    original_data["label"].isin(["ham", "spam"])
]

# Remove empty messages.
original_data = original_data[original_data["message"] != ""]

# Remove duplicate messages with the same label.
original_data = original_data.drop_duplicates(
    subset=["label", "message"]
).reset_index(drop=True)

# Display the size and class distribution of the cleaned dataset.
print("Clean original dataset:", len(original_data))
print(original_data["label"].value_counts())


# --------------------------------------------------
# 3. SPLIT ORIGINAL DATA: 70% / 15% / 15%
# --------------------------------------------------

# First split:
# 70% of the original dataset is used for training.
# The remaining 30% is temporarily set aside.
#
# stratify maintains the ham/spam class proportions
# in the training and temporary datasets.
X_train_original, X_temp, y_train_original, y_temp = train_test_split(
    original_data["message"],
    original_data["label"],
    test_size=0.30,
    random_state=42,
    stratify=original_data["label"],
)

# Second split:
# Divide the remaining 30% equally into validation and test data.
# Each receives approximately 15% of the original dataset.
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

# Display the number of messages in each dataset.
print("\nOriginal data split:")
print("Training:", len(X_train_original))
print("Validation:", len(X_validation))
print("Testing:", len(X_test))


# --------------------------------------------------
# 4. LOAD CUSTOM INDIAN SMS EXAMPLES
# --------------------------------------------------

# Load additional custom SMS examples.
# These examples are stored in tab-separated format.
custom_data = pd.read_csv(
    CUSTOM_FILE,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

# Remove missing values.
custom_data = custom_data.dropna()

# Standardize labels and clean message text.
custom_data["label"] = custom_data["label"].str.strip().str.lower()
custom_data["message"] = custom_data["message"].astype(str).str.strip()

# Stop execution if any custom label is not ham or spam.
if not custom_data["label"].isin(["ham", "spam"]).all():
    raise ValueError("Invalid custom labels. Use only ham or spam.")

# Remove empty messages and duplicate examples.
custom_data = custom_data[custom_data["message"] != ""]
custom_data = custom_data.drop_duplicates(
    subset=["label", "message"]
).reset_index(drop=True)

# Display the number and distribution of custom examples.
print("\nCustom examples:", len(custom_data))
print(custom_data["label"].value_counts())


# --------------------------------------------------
# 5. ADD CUSTOM DATA TO TRAINING ONLY
# --------------------------------------------------

# Combine the original training messages with custom examples.
# Custom examples are added only to the training dataset.
X_train = pd.concat(
    [X_train_original, custom_data["message"]],
    ignore_index=True,
)

# Combine their corresponding labels.
y_train = pd.concat(
    [y_train_original, custom_data["label"]],
    ignore_index=True,
)

# Display the final training dataset size.
print("\nFinal training messages:", len(X_train))


# --------------------------------------------------
# 6. TF-IDF FEATURE EXTRACTION
# --------------------------------------------------

# Initialize the TF-IDF vectorizer.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
    sublinear_tf=True,
)

# Learn the vocabulary and TF-IDF weights using training messages.
# Transform validation and test messages using the same vocabulary.
X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)

# Display the dimensions of the resulting feature matrices.
print("\nTF-IDF shapes:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_validation_tfidf.shape)
print("Testing:", X_test_tfidf.shape)


# --------------------------------------------------
# 7. DEFINE CLASSIFIERS
# --------------------------------------------------

# Define three machine learning models for comparison.
models = {
    # Multinomial Naive Bayes classifier.
    "Multinomial Naive Bayes": MultinomialNB(),

    # Logistic Regression without class weighting.
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    # Logistic Regression with balanced class weights.
    # This helps account for differences in class distribution.
    "Logistic Regression (Balanced)": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
}

# Dictionaries to store evaluation results and trained models.
results = {}
trained_models = {}


# --------------------------------------------------
# 8. TRAIN AND COMPARE MODELS ON VALIDATION DATA
# --------------------------------------------------

# Train each classifier using the training dataset.
for name, model in models.items():

    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model

    # Predict labels for the validation dataset.
    predictions = model.predict(X_validation_tfidf)

    # Calculate overall accuracy.
    accuracy = accuracy_score(y_validation, predictions)

    # Calculate precision for the spam class.
    precision = precision_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )

    # Calculate recall for the spam class.
    recall = recall_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )

    # Calculate the F1-score for the spam class.
    f1 = f1_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )

    # Store the evaluation metrics for this classifier.
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Display the validation performance.
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall: {recall:.4f}")
    print(f"Spam F1-score: {f1:.4f}")


# --------------------------------------------------
# 9. SELECT BALANCED LOGISTIC REGRESSION
# --------------------------------------------------

# Select the balanced Logistic Regression model
# for the remaining evaluation steps.
selected_name = "Logistic Regression (Balanced)"
selected_model = trained_models[selected_name]

# Find the position of the spam class in the model's class list.
spam_index = list(selected_model.classes_).index("spam")

# Obtain spam probabilities for the validation dataset.
validation_probabilities = selected_model.predict_proba(
    X_validation_tfidf
)[:, spam_index]


# --------------------------------------------------
# 10. SELECT THRESHOLD USING VALIDATION DATA ONLY
# --------------------------------------------------

# Compare different probability thresholds.
# A message is classified as spam when its probability
# is greater than or equal to the selected threshold.
print("\nVALIDATION THRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1 | FP | FN")

threshold_results = []

# Test a range of possible spam thresholds.
for threshold in [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]:
    # Generate predictions using the current threshold.
    predictions = [
        "spam" if p >= threshold else "ham"
        for p in validation_probabilities
    ]

    # Generate the confusion matrix.
    # Label order: ham, spam.
    matrix = confusion_matrix(
        y_validation, predictions, labels=["ham", "spam"]
    )

    # Extract true negatives, false positives,
    # false negatives and true positives.
    tn, fp, fn, tp = matrix.ravel()

    # Calculate precision, recall and F1-score.
    precision = precision_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_validation, predictions, pos_label="spam", zero_division=0
    )

    # Store the metrics for this threshold.
    threshold_results.append((f1, threshold, precision, recall, fp, fn))

    # Display the results for the current threshold.
    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:.3f} | {fp:2d} | {fn:2d}"
    )

# Choose the threshold with the highest validation F1.
# If tied, prefer the higher threshold to reduce false positives.
best_f1 = max(item[0] for item in threshold_results)

best_threshold = max(
    item[1] for item in threshold_results if item[0] == best_f1
)

# Display the selected threshold and its validation F1-score.
print("\nSelected validation threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))


# --------------------------------------------------
# 11. FINAL EVALUATION ON UNTOUCHED TEST DATA
# --------------------------------------------------

# Obtain spam probabilities for the separate test dataset.
test_probabilities = selected_model.predict_proba(
    X_test_tfidf
)[:, spam_index]

# Apply the threshold selected using validation data.
test_predictions = [
    "spam" if p >= best_threshold else "ham"
    for p in test_probabilities
]

# Calculate test accuracy.
test_accuracy = accuracy_score(y_test, test_predictions)

# Generate the confusion matrix for the test dataset.
matrix = confusion_matrix(
    y_test, test_predictions, labels=["ham", "spam"]
)

# Extract the four confusion matrix values.
tn, fp, fn, tp = matrix.ravel()

# Display final test performance.
print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(f"Selected threshold: {best_threshold:.2f}")
print(f"Accuracy: {test_accuracy:.4f}")

# Display precision, recall, F1-score and other class metrics.
print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=["ham", "spam"],
        zero_division=0,
    )
)

# Display the confusion matrix.
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(matrix)

# Calculate false positive and false negative rates.
print(f"False positive rate: {fp / (fp + tn):.4f}")
print(f"False negative rate: {fn / (fn + tp):.4f}")


# --------------------------------------------------
# 12. INSPECT TEST MISCLASSIFICATIONS
# --------------------------------------------------

# Create a DataFrame containing actual and predicted labels,
# probabilities and original SMS messages.
error_data = pd.DataFrame({
    "message": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": test_predictions,
    "spam_probability": test_probabilities,
})

# Select messages where the predicted label differs from
# the actual label.
errors = error_data[
    error_data["actual"] != error_data["predicted"]
]

print("\nMISCLASSIFIED TEST MESSAGES")

# Display misclassified messages, if any.
if errors.empty:
    print("No misclassified messages.")
else:
    print(
        errors[
            ["actual", "predicted", "spam_probability", "message"]
        ].to_string(index=False)
    )


# --------------------------------------------------
# 13. MODEL COMPARISON
# --------------------------------------------------

# Display the validation metrics of all three classifiers.
print("\nVALIDATION MODEL COMPARISON")

for name, metrics in results.items():
    print(
        f"{name}: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}"
    )


# --------------------------------------------------
# 14. SAVE MODEL, VECTORIZER AND THRESHOLD
# --------------------------------------------------

# Save the selected balanced Logistic Regression model.
joblib.dump(
    selected_model,
    MODEL_DIR / "sms_model.pkl",
)

# Save the TF-IDF vectorizer used during training.
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl",
)

# Save the threshold selected using validation data.
joblib.dump(
    best_threshold,
    MODEL_DIR / "sms_threshold.pkl",
)

print("\nSaved model and vectorizer.")
print("Saved threshold:", best_threshold)
print("\nTraining and evaluation complete.")
print("Custom examples were used in training only.")
print("Threshold was selected using validation data only.")
print("Final performance was measured on untouched test data.")
