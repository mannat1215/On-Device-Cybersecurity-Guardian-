 URL Detection Module

This module is responsible for identifying phishing and legitimate URLs using machine learning.

It uses URL-based feature extraction and a Random Forest classifier to analyse suspicious URL patterns and classify them as phishing or legitimate.

Key Features
- Extracts 20 lexical and structural features from URLs.
- Uses domain-based data splitting to reduce data leakage.
- Trains a Random Forest classification model.
- Performs threshold tuning using the F2-score.
- Evaluates model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Predicts whether a user-provided URL is phishing or legitimate.

 Dataset
The module uses the PhiUSIIL Phishing URL Dataset.

Main File
- `url_detection.py` – Contains URL feature extraction, model training, evaluation, threshold selection, and URL prediction.
