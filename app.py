import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import email
from email import policy
from email.parser import BytesParser

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Load the trained model and vectorizer
with open("model/spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

def extract_eml_text(uploaded_file):
    try:
        msg = BytesParser(policy=policy.default).parse(uploaded_file)
        subject = msg['subject'] or ''
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body += part.get_content()
        else:
            body = msg.get_content()
        return f"{subject}\n{body}"
    except Exception as e:
        return f"Error parsing .eml file: {e}"

st.title("📧 Email Spam Classifier with .eml Support")

st.markdown("### 📄 Input Options")

# Select input method
input_method = st.radio("Choose input method:", ("Manual Text Input", "Upload Text File (.txt)", "Upload Email File (.eml)"))

input_msg = ""

if input_method == "Manual Text Input":
    input_msg = st.text_area("Enter the email text here:")

elif input_method == "Upload Text File (.txt)":
    uploaded_txt = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_txt is not None:
        input_msg = uploaded_txt.read().decode("utf-8")

elif input_method == "Upload Email File (.eml)":
    uploaded_eml = st.file_uploader("Upload a .eml file", type=["eml"])
    if uploaded_eml is not None:
        input_msg = extract_eml_text(uploaded_eml)

if st.button("Predict"):
    if input_msg.strip() == "":
        st.warning("Please enter or upload some text.")
    else:
        processed = preprocess_text(input_msg)
        vect_msg = vectorizer.transform([processed])
        prediction = model.predict(vect_msg)[0]

        if prediction == 1:
            st.error("🚨 Spam Email Detected!")
        else:
            st.success("✅ Not Spam (Ham Email)")
