import pandas as pd
import numpy as np
import re
import string
import pickle
import os # New: To create directories if they don't exist

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

## ---------------------------------
## 📄 TEXT CLEANING FUNCTION
## ---------------------------------

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing URLs.
    3. Removing punctuation.
    4. Removing numbers.
    5. Removing extra spaces.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

## ---------------------------------
## 🧠 PREDICTION FUNCTION (The Smart Part)
## ---------------------------------

def predict_fake_news(text, model, vectorizer):
    """
    Predicts whether a new piece of text is Fake (1) or True (0).
    """
    # 1. Clean the input text
    cleaned_text = clean_text(text)
    
    # 2. Vectorize the cleaned text (Must use the fitted vectorizer)
    text_vec = vectorizer.transform([cleaned_text])
    
    # 3. Get the prediction
    prediction = model.predict(text_vec)[0]
    
    # 4. Get the prediction probability (optional but helpful)
    probability = model.predict_proba(text_vec)[0]
    
    return prediction, probability

## ---------------------------------
## ⚙️ MAIN TRAINING AND SAVING LOGIC
## ---------------------------------

# Define dataset paths
TRUE_DATA_PATH = "Dataset/true.csv"
FAKE_DATA_PATH = "Dataset/fake.csv"
SAVED_DIR = "saved"

# Create the save directory if it doesn't exist
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

# Load data
try:
    true_df = pd.read_csv(TRUE_DATA_PATH)
    fake_df = pd.read_csv(FAKE_DATA_PATH)
except FileNotFoundError as e:
    print(f"Error: Dataset file not found. Ensure your files are at: {e.filename}")
    exit()

# Assign labels
true_df["label"] = 0 # 0 for True News
fake_df["label"] = 1 # 1 for Fake News

# Combine dataframes
df = pd.concat([true_df, fake_df], ignore_index=True)

# Clean text
df["text_clean"] = df["text"].apply(clean_text)

X = df["text_clean"]
y = df["label"]

# Split data before vectorization to ensure proper evaluation later
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
# Fit only on the training data!
X_train_vec = vectorizer.fit_transform(X_train_text)
# Transform test data using the fitted vectorizer
X_test_vec = vectorizer.transform(X_test_text)

print(f"Dataset Size: {len(df)} samples")
print(f"Training Data Size: {len(X_train_vec)} samples")
print("Starting Model Training...")

# TRAIN MODEL
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# EVALUATE
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL
MODEL_PATH = os.path.join(SAVED_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(SAVED_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\nModel Saved to: {MODEL_PATH}")
print(f"Vectorizer Saved to: {VECTORIZER_PATH}")
print("--- Training and Saving Complete! ---\n")

## ---------------------------------
## 🚀 EXAMPLE USAGE (The Automatic Detection System in Action)
## ---------------------------------

# This section simulates the deployment phase where you load the model
# and use it to detect fake news automatically.

print("\n--- Automatic Fake News Detection Test ---")

# Example 1: (Assume this is a TRUE headline)
true_example = "The World Health Organization (WHO) confirmed that the new COVID-19 variant is highly contagious but less severe than the previous strain."

# Example 2: (Assume this is a FAKE headline)
fake_example = "URGENT: Drinking lemon water every morning completely cures all forms of cancer, according to secret government reports released today."

test_cases = {
    "TRUE News Example": true_example,
    "FAKE News Example": fake_example
}

# Use the trained model and vectorizer for prediction
for name, text in test_cases.items():
    
    # Get the prediction
    pred, prob = predict_fake_news(text, model, vectorizer)
    
    # Interpret the results
    result = "FAKE NEWS (خطر ⚠️)" if pred == 1 else "TRUE NEWS (صحيح ✅)"
    
    # Get the probability for the predicted class
    confidence = prob[pred] * 100
    
    print(f"\nTest Case: {name}")
    print(f"   Text: {text[:80]}...")
    print(f"   Prediction: {result}")
    print(f"   Confidence: {confidence:.2f}%")
