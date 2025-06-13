import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load saved model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    lbl_encoder = pickle.load(f)

# Preprocessing function — no NLTK at all
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()  # simple whitespace split
    stops = ENGLISH_STOP_WORDS
    filtered = [w for w in tokens if w not in stops]
    return " ".join(filtered)

# Streamlit UI
st.title("📧 Spam Mail Prediction App")
user_input = st.text_area("Enter the email/message text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        vect = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vect)[0]
        label = lbl_encoder.inverse_transform([pred])[0].upper()
        if label == 'SPAM':
            st.error(f"⚠️ This is **{label}**.")
        else:
            st.success(f"✅ This is **{label}**.")
