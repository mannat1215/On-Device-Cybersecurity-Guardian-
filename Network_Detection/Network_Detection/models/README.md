
import pandas as pd
from pathlib import Path

# Dataset folder
DATA_DIR = Path(".")

train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

print("Loading training dataset...")
train = pd.read_csv(train_path)

print("Loading testing dataset...")
test = pd.read_csv(test_path)

print("\n========== TRAINING DATA ==========")
print("Shape:", train.shape)
print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nMissing values:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nLabel distribution:")
if "label" in train.columns:
    print(train["label"].value_counts())
else:
    print("No 'label' column found.")

print("\nAttack category distribution:")
if "attack_cat" in train.columns:
    print(train["attack_cat"].value_counts(dropna=False))
else:
    print("No 'attack_cat' column found.")

print("\n========== TESTING DATA ==========")
print("Shape:", test.shape)

print("\nTesting label distribution:")
if "label" in test.columns:
    print(test["label"].value_counts())

print("\nDataset inspection complete.")
