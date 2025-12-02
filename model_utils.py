import pickle
import re
import string
from deep_translator import GoogleTranslator


# --------------------------
# Load Model and Vectorizer
# --------------------------

with open("saved/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("saved/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# --------------------------
# Clean English Text
# --------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------
# PREDICT (Arabic or English)
# --------------------------

def predict_news(text):
    # If input contains Arabic → translate to English
    if re.search(r"[\u0600-\u06FF]", text):
        text = GoogleTranslator(source="auto", target="en").translate(text)

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    return "Fake" if pred == 1 else "Real"
