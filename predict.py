import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("fake_news_model.h5")

stop_words = set(stopwords.words("english"))
max_len = 1000

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_fake_news(title, text):
    content = title + " " + text
    cleaned = clean_text(content)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    prob = model.predict(padded)[0][0]
    label = "True" if prob > 0.5 else "Fake"
    return label, float(prob)
