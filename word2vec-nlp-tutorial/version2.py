#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Competition : Bag of Words Meets Bag of Popcorn
Methods used: NLP and Deep Learning

"""
import pandas as pd

## Labeled Train Data
df1 = pd.read_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/labeledTrainData.tsv', delimiter="\t")
df1 = df1.drop(['id'], axis=1)

## Imdb Data
df2 = pd.read_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/imdb_master.csv',encoding="latin-1")
## The data has some unnecessary columns, they must be dropped
df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)
df2.columns = ["review","sentiment"]
## Sentiment column has some unlabeled rows. They must be excluded
## To make all data consistent, we map 'pos' and 'neg' to 1 and 0
df2.sentiment.value_counts()
df2 = df2[df2.sentiment != 'unsup']
df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})

## concatinate the imdb dataset to our labeled train data
df = pd.concat([df1, df2]).reset_index(drop=True)
df.head()



#------------------------------------------------------
# Text Processing
#------------------------------------------------------

import re
import nltk
## download 'wordnet' to use for lemmatization. 
## nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

## step 1: go through the text and replace 1. what is not-a-word-character (letter,
## digit or underscore) ^\w and 2. what is non-a-white-space ^\s with a space ' '.
## This step removes all white space and punctuations.
## step 2: make all words lowercase
## step 3: split the text and find the lemma of each word.
## step 4: since lemmatizer automatically assumes each word is a noun. We need to 
## feed words one more time with the assumption that they are verbs to get a more
## accurate lemma
## step 5: remove stop words
## step 6: Since the text is split in step 3. We must join all the list to create
## a new document. 
def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

## Apply the defined function to each row of column 'review'
df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))

## Applying deep learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
## pad_sequences is used to ensure that all sequences in a list have the same
## length. By default this is done by padding 0 in the beginning of each sequence
## until each sequence has the same length as the longest sequence.
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

## Takes 6000 most prevalent words and tokenize them. So, each word would have
## a numerical id
max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

## TBD
maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


## Ultimate model performance
df_test=pd.read_csv("~/Kaggle-Competitions/word2vec-nlp-tutorial/testData.tsv",header=0, delimiter="\t", quoting=3)
df_test.head()
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)