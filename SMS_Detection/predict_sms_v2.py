
import joblib
from pathlib import Path


# Get the location of the current Python file.
BASE_DIR = Path(__file__).resolve().parent

# Define the folder containing the trained SMS model files.
MODEL_DIR = BASE_DIR / "models"


# Load the trained V2 machine learning model.
model = joblib.load(MODEL_DIR / "sms_model_v2.pkl")

# Load the TF-IDF vectorizer used during model training.
# It converts SMS text into numerical features.
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer_v2.pkl")

# Load the spam classification threshold selected during validation.
SPAM_THRESHOLD = joblib.load(MODEL_DIR / "sms_threshold_v2.pkl")


def detect_sms(message):
    """
    Detect whether an SMS message is ham or spam
    using the trained V2 model.
    """

    # Handle empty messages safely.
    if not message or not message.strip():
        return {
            "module": "sms",
            "prediction": "ham",
            "spam_probability": 0.0,
            "risk_level": "low"
        }

    # Convert the input SMS into TF-IDF numerical features.
    features = vectorizer.transform([message])

    # Predict the probability that the message is spam.
    probabilities = model.predict_proba(features)[0]
    spam_probability = float(probabilities[1])

    # Compare the spam probability with the selected threshold.
    if spam_probability >= SPAM_THRESHOLD:
        prediction = "spam"
    else:
        prediction = "ham"

    # Assign a risk level based on the prediction and probability.
    if prediction == "spam":
        if spam_probability >= 0.80:
            risk_level = "high"
        else:
            risk_level = "medium"
    else:
        risk_level = "low"

    # Return the prediction and associated risk information.
    return {
        "module": "sms",
        "prediction": prediction,
        "spam_probability": round(spam_probability, 4),
        "risk_level": risk_level
    }


# Run this section only when this file is executed directly.
# It does not run when the function is imported by main.py.
if __name__ == "__main__":

    print("SMS V2 Detector")
    print("Spam threshold:", SPAM_THRESHOLD)

    # Allow the user to test multiple SMS messages.
    while True:
        message = input("\nEnter SMS (or type exit): ")

        if message.lower() == "exit":
            break

        result = detect_sms(message)
        print("SMS Result:", result)
