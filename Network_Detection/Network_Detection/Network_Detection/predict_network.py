
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("network_threat_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run train_network.py first."
    )

model = joblib.load(MODEL_PATH)

def predict_network(record):
    """
    record must contain the network feature columns
    used during training.
    """
    row = pd.DataFrame([record])

    probabilities = model.predict_proba(row)[0]
    attack_probability = float(probabilities[1])

    prediction = (
        "ATTACK" if attack_probability >= 0.5
        else "NORMAL"
    )

    return {
        "module": "network_threat_detection",
        "prediction": prediction,
        "attack_probability": round(attack_probability, 4),
        "risk_level": (
            "HIGH" if attack_probability >= 0.8
            else "MEDIUM" if attack_probability >= 0.4
            else "LOW"
        )
    }

# Load a real test record, excluding target/leakage columns.
test_path = Path("UNSW_NB15_testing-set.csv")
test_data = pd.read_csv(test_path)

feature_columns = [
    c for c in test_data.columns
    if c not in ["label", "attack_cat", "id"]
]

sample = test_data[feature_columns].iloc[0].to_dict()

print("Testing a sample network record:")
print(predict_network(sample))
