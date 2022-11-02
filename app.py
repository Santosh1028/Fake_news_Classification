import streamlit as st
import pickle

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


port_stem=PorterStemmer()
def stemming(content):
    review=re.sub('[^a-zA-Z]', ' ', content)
    review=review.lower()
    review=review.split()
    review=[port_stem.stem(word) for word in review if not word in stopwords.words('English')]
    review=' '.join(review)
    return review


vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

st.title('Fake News Classification')

# 1- Preprocessing
input_news=st.text_input("Enter the News:")

if st.button('Predict'):
    transformed_text = stemming(input_news)

    vector_input = vectorizer.transform([transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Fake News')

    else:
        st.header('Real News')