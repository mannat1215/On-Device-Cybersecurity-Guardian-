
import pandas as pd
import re
import math
from urllib.parse import urlparse
from sklearn.metrics import fbeta_score

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. URL FEATURE EXTRACTION
# ==========================================

def extract_url_features(url):
    url = str(url).strip()

    # Add a temporary scheme only for parsing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        parsed_url = "https://" + url
    else:
        parsed_url = url

    parsed = urlparse(parsed_url)
    domain = (parsed.hostname or "").lower()

    # Remove the scheme so that the same URL has
    # the same lexical features with or without HTTPS.
    feature_url = re.sub(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        "",
        url
    ).lower()

    url_length = len(feature_url)
    domain_length = len(domain)

    is_domain_ip = int(
        re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain)
        is not None
    )

    domain_parts = domain.split(".") if domain else []
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    tld_length = len(tld)
    no_of_subdomain = max(len(domain_parts) - 2, 0)

    # Obfuscation
    no_of_obfuscated_char = feature_url.count("%")
    has_obfuscation = int(no_of_obfuscated_char > 0)
    obfuscation_ratio = (
        no_of_obfuscated_char / url_length if url_length else 0
    )

    # Letters
    no_of_letters = sum(c.isalpha() for c in feature_url)
    letter_ratio = (
        no_of_letters / url_length if url_length else 0
    )

    # Digits
    no_of_digits = sum(c.isdigit() for c in feature_url)
    digit_ratio = (
        no_of_digits / url_length if url_length else 0
    )

    # Query parameters
    no_of_equals = feature_url.count("=")
    no_of_question = feature_url.count("?")
    no_of_ampersand = feature_url.count("&")

    # Special characters
    special_chars = sum(not c.isalnum() for c in feature_url)
    special_char_ratio = (
        special_chars / url_length if url_length else 0
    )

    # Suspicious keywords
    suspicious_words = [
        "login", "signin", "verify", "verification",
        "account", "password", "secure", "update",
        "bank", "banking", "payment", "wallet",
        "confirm", "bonus", "prize", "free"
    ]

    has_suspicious_keyword = int(
        any(word in feature_url for word in suspicious_words)
    )

    # Shannon entropy
    frequency = {}
    for char in feature_url:
        frequency[char] = frequency.get(char, 0) + 1

    entropy = 0.0
    if url_length:
        for count in frequency.values():
            probability = count / url_length
            entropy -= probability * math.log2(probability)

    # IsHTTPS is deliberately excluded from the model.
    # The remaining features are independent of the URL scheme.
    return [
        url_length,
        domain_length,
        is_domain_ip,
        tld_length,
        no_of_subdomain,
        no_of_obfuscated_char,
        has_obfuscation,
        obfuscation_ratio,
        no_of_letters,
        letter_ratio,
        no_of_digits,
        digit_ratio,
        no_of_equals,
        no_of_question,
        no_of_ampersand,
        special_chars,
        special_char_ratio,
        int("@" in feature_url),
        has_suspicious_keyword,
        entropy
    ]


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    "phiusiiL+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
)

df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()

# Remove empty URLs
df = df[df["URL"] != ""].copy()

# Dataset labels: 0 = phishing, 1 = legitimate
y = df["label"].astype(int)

print("Dataset shape:", df.shape)
print("Total URLs:", len(df))
print("Unique URLs:", df["URL"].nunique())
print("Duplicate URLs:", df["URL"].duplicated().sum())
print("\nLabel counts:")
print(y.value_counts().sort_index())


# ==========================================
# 3. EXTRACT FEATURES
# ==========================================

print("\nExtracting URL features...")

feature_names = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfObfuscatedChar",
    "HasObfuscation",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDigitsInURL",
    "DigitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfSpecialChars",
    "SpecialCharRatio",
    "HasAtSymbol",
    "HasSuspiciousKeyword",
    "Entropy"
]

feature_values = df["URL"].apply(extract_url_features)

X = pd.DataFrame(
    feature_values.tolist(),
    columns=feature_names,
    index=df.index
)

print("Feature count:", len(feature_names))


# ==========================================
# 4. DOMAIN-BASED TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset by domain...")

def get_domain(url):
    url = str(url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    return (urlparse(url).hostname or "").lower()


if "Domain" in df.columns:
    domains = df["Domain"].fillna("").astype(str).str.lower().str.strip()
else:
    domains = pd.Series("", index=df.index, dtype=str)

for idx in df.index[domains == ""]:
    domains.at[idx] = get_domain(df.at[idx, "URL"])

# Keep rows with the same domain in the same split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=domains)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_domains = set(domains.iloc[train_idx])
test_domains = set(domains.iloc[test_idx])
overlap = train_domains.intersection(test_domains)

print("Training URLs:", len(X_train))
print("Testing URLs:", len(X_test))
print("Training domains:", len(train_domains))
print("Testing domains:", len(test_domains))
print("Domains appearing in both sets:", len(overlap))

if overlap:
    raise RuntimeError("Domain overlap detected!")

if y_train.nunique() < 2:
    raise RuntimeError("Training data contains only one class.")


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)
import joblib

joblib.dump(
    {
        "model": model,
        "feature_names": feature_names
    },
    "url_phishing_model.joblib"
)

print("Model saved successfully!")
joblib.dump(
    {
        "model": model,
        "feature_names": feature_names,
        "phishing_threshold": best_threshold
    },
    "url_phishing_model.joblib"
)



# ==========================================
# 6. TRAIN AND TUNE PHISHING THRESHOLD
# ==========================================

print("\nTraining model and tuning threshold...")

# Separate some training domains for validation.
# The final test domains remain untouched.
train_groups = domains.iloc[train_idx]

inner_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

fit_idx, val_idx = next(
    inner_splitter.split(
        X_train,
        y_train,
        groups=train_groups
    )
)

X_fit = X_train.iloc[fit_idx]
y_fit = y_train.iloc[fit_idx]

X_val = X_train.iloc[val_idx]
y_val = y_train.iloc[val_idx]

validation_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

validation_model.fit(X_fit, y_fit)

# Class 0 = phishing
phishing_index = list(validation_model.classes_).index(0)
val_probabilities = validation_model.predict_proba(X_val)[
    :, phishing_index
]

# Select a threshold using validation F2 score.
# F2 gives more weight to recall than precision.
best_threshold = 0.50
best_f2 = -1

for threshold in [i / 100 for i in range(10, 61, 5)]:
    val_predictions = [
        0 if probability >= threshold else 1
        for probability in val_probabilities
    ]

    score = fbeta_score(
        (y_val == 0).astype(int),
        [int(prediction == 0) for prediction in val_predictions],
        beta=2,
        zero_division=0
    )

    if score > best_f2:
        best_f2 = score
        best_threshold = threshold

print("Selected phishing threshold:", best_threshold)
print("Validation F2 score:", round(best_f2, 4))

# Train final model on all training domains.
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ==========================================
# 7. EVALUATE ON UNTOUCHED TEST DOMAINS
# ==========================================

print("\n================================")
print("DOMAIN-BASED MODEL RESULTS")
print("================================")

test_phishing_index = list(model.classes_).index(0)

test_probabilities = model.predict_proba(X_test)[
    :, test_phishing_index
]

# Apply the threshold chosen on validation data.
y_pred = [
    0 if probability >= best_threshold else 1
    for probability in test_probabilities
]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 5))
print("Phishing threshold:", best_threshold)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("Confusion Matrix (rows=true, columns=predicted)")
print("Order: Phishing, Legitimate")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# ==========================================
# 8. PREDICT A NEW URL
# ==========================================

def predict_url(url):
    if not url or not str(url).strip():
        return "INVALID URL", None

    features = extract_url_features(url)

    feature_df = pd.DataFrame(
        [features],
        columns=feature_names
    )

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    result = "PHISHING" if prediction == 0 else "LEGITIMATE"

    return result, confidence


# ==========================================
# 9. URL NORMALIZATION CHECK
# ==========================================

print("\n--- URL NORMALIZATION TEST ---")

url_a = "www.google.com"
url_b = "https://www.google.com"

features_a = extract_url_features(url_a)
features_b = extract_url_features(url_b)

print("Inputs:", url_a, "|", url_b)
print("Features identical:", features_a == features_b)


# ==========================================
# 10. INTERACTIVE URL TESTER
# ==========================================

print("\n================================")
print("URL PHISHING DETECTOR")
print("================================")
print("Enter a full URL or a domain.")
print("Type 'exit' to stop.")

while True:
    try:
        url = input("\nEnter a URL: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting URL detector.")
        break

    if url.lower() == "exit":
        break

    result, confidence = predict_url(url)

    if result == "INVALID URL":
        print("Please enter a non-empty URL.")
        continue

    print("Prediction:", result)
    print("Model confidence:", round(confidence, 2), "%")
