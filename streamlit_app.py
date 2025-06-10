import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Tell NLTK where to find the downloaded data
nltk.data.path.append("./nltk_data")

# Load saved model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Streamlit UI
st.title("Spam Mail Prediction App")

user_input = st.text_area("Enter the email/message text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = preprocess(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input.toarray())

        if prediction[0] == 1:
            st.error("This is a SPAM message.")
        else:
            st.success("This is a HAM (not spam) message.")
