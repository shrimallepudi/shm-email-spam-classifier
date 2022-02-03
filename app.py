import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
import sklearn

tags_dict = {
                'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV
            } 

def message_transformer(text):
    
    # converting to lowercase
    text = text.lower()
    
    # converting words to tokens
    text = nltk.word_tokenize(text)
    
    y = []
    
    # checking for alpha numeric and eliminating special chars
    for i in text:
        if i.isalnum():
            y.append(i)
    

    text = y[:]
    y.clear()
    
    # removing english stopwords
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    
    # lemmatising the words
    text = y[:]
    y.clear()
    lem = WordNetLemmatizer()
    for i in text:
        tag = pos_tag([i])[0][1][0]
        if tag in tags_dict.keys():
            y.append(lem.lemmatize(i, tags_dict[tag]))
        else:
            y.append(lem.lemmatize(i, wordnet.NOUN))
            
    return " ".join(sorted(set(y), key=y.index))

tfidf = pickle.load(open('vectorize.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classifier')

input = st.text_area('Enter your message here')

if st.button('Predict'):
    transformed_text = message_transformer(input)
    input_text_vector = tfidf.transform([transformed_text])
    result = model.predict(input_text_vector)[0]

    if result == 1:
        st.header('Spam')
    elif result == 0:
        st.header('Not spam')