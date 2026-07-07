import re

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "models"
VECTORIZER_PATH = f"{MODEL_DIR}/vectorizer.joblib"
POSITIVE_MODEL_PATH = f"{MODEL_DIR}/positive_model.joblib"
NEGATIVE_MODEL_PATH = f"{MODEL_DIR}/negative_model.joblib"

FRENCH_STOPWORDS = [
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "et", "en", "au",
    "aux", "avec", "ce", "ces", "pour", "par", "sur", "pas", "plus", "où", "mais",
    "ou", "donc", "ni", "car", "ne", "que", "qui", "quoi", "quand", "à", "son",
    "sa", "ses", "ils", "elles", "nous", "vous", "est", "sont", "cette", "cet",
    "aussi", "être", "avoir", "faire", "comme", "tout", "bien", "mal", "on", "lui",
    "j", "l", "d", "s", "n", "qu", "c", "m",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def build_vectorizer() -> CountVectorizer:
    return CountVectorizer(stop_words=FRENCH_STOPWORDS, max_features=300)


def train(texts, positive_labels, negative_labels):
    """Fit one shared vectorizer and two independent LogisticRegression models."""
    cleaned = [clean_text(t) for t in texts]

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(cleaned)

    positive_model = LogisticRegression(max_iter=1000)
    positive_model.fit(X, positive_labels)

    negative_model = LogisticRegression(max_iter=1000)
    negative_model.fit(X, negative_labels)

    return vectorizer, positive_model, negative_model


def save_artifacts(vectorizer, positive_model, negative_model):
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(positive_model, POSITIVE_MODEL_PATH)
    joblib.dump(negative_model, NEGATIVE_MODEL_PATH)


def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    positive_model = joblib.load(POSITIVE_MODEL_PATH)
    negative_model = joblib.load(NEGATIVE_MODEL_PATH)
    return vectorizer, positive_model, negative_model


def score_texts(vectorizer, positive_model, negative_model, texts):
    """Return a sentiment score in [-1, 1] for each text: P(positive) - P(negative)."""
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    p_positive = positive_model.predict_proba(X)[:, 1]
    p_negative = negative_model.predict_proba(X)[:, 1]
    scores = p_positive - p_negative
    return scores.tolist()
