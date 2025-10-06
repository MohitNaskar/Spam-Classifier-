import streamlit as st
import pickle 
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenization

    y = []
    for i in text:
        if i.isalnum():  # keep only alphanumeric
            y.append(i)
    text = y[:] 
    y.clear() 

    # remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# load vectorizer and model
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

sms = st.text_input('Enter the message')

if st.button('Predict'):  # only run when user clicks
    # 1. preprocess
    transformed_sms = transform_text(sms)

    # 2. vectorize
    vector_input = tfidf_vectorizer.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header('Spam 🚨')
    else:
        st.header('Not Spam ✅')
