import streamlit as st
import pandas as pd
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load saved model and vectorizer
@st.cache_resource
def load_resources():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return vectorizer, model, encoder

# Preprocessing function
def preprocess(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    tokens = [ps.stem(w) for w in tokens]
    return ' '.join(tokens)

# Load resources
vectorizer, model, encoder = load_resources()

# Streamlit UI
st.title("📧 Email Spam Classifier")

st.write("Enter the content of an email below to determine whether it's spam or not.")

user_input = st.text_area("Email Text", height=200)

if st.button("Predict Spam / Ham"):
    if user_input.strip() == "":
        st.warning("Please enter some email text to analyze.")
    else:
        # Preprocess and vectorize input
        cleaned_input = preprocess(user_input)
        vect_input = vectorizer.transform([cleaned_input]).toarray()
        pred = model.predict(vect_input)[0]
        label = encoder.inverse_transform([pred])[0].upper()
        if label == 'SPAM':
            st.error("⚠️ This email is predicted as **SPAM**.")
        else:
            st.success("✅ This email is predicted as **HAM (Not Spam)**.")

# Instructions to run:
# 1. Save trained model, vectorizer and encoder using pickle
# 2. Run this script with: streamlit run app.py
