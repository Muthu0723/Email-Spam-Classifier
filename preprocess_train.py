# preprocess_train.py

import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

os.makedirs("model", exist_ok=True)
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete. Model and vectorizer saved.")
