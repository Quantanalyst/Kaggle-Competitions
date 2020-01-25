#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Competition : Bag of Words Meets Bag of Popcorn
Methods used: 

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import textblob

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re
import pickle
from scipy.sparse import hstack

from sklearn import preprocessing, model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
## TfidfVectorizer and CountVectorizer are equivalent. The difference is the
## way they show frequency. CountVectorizer shows the count of words and 
## TfidfVectorizer show the probability
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer,roc_curve, roc_auc_score


# Load Data
print("Loading data...")

train = pd.read_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/labeledTrainData.tsv', sep="\t")
print("Train shape:", train.shape)
test = pd.read_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/testData.tsv', sep="\t")
print("Test shape:", test.shape)


additional_data = pd.read_csv(
    filepath_or_buffer='~/Kaggle-Competitions/word2vec-nlp-tutorial/imdb_master.csv',
    encoding="latin-1")
additional_data = additional_data.drop(['Unnamed: 0','type','file'],axis=1)
additional_data.columns = ["review","sentiment"]
sup_data = additional_data[additional_data['sentiment'] != 'unsup']
unsup_data = additional_data[additional_data['sentiment'] == 'unsup']


sup_data.to_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/imdb_master_supervised.csv')
unsup_data.to_csv('~/Kaggle-Competitions/word2vec-nlp-tutorial/imdb_master_unsupervised.csv')

train['sentiment'].value_counts() # balanced dataset

# Check the first review
print('The first review is:\n\n',train["review"][0])

# clean description
## BeautifulSoup is a Python library for pulling data out of HTML and XML files.
## Beautiful Soup is a library that makes it easy to scrape information from web
## pages. It sits atop an HTML or XML parser, providing Pythonic idioms for
## iterating, searching, and modifying the parse tree.
## In the original reviews there are some <br> </br> that can be cleaned by
## BeautifulSoup.
print("Cleaning train data...\n")
train['review'] = train['review'].map(lambda x: BeautifulSoup(x).get_text())
print("Cleaning test data...")
test['review'] = test['review'].map(lambda x: BeautifulSoup(x).get_text())


# function to clean data
stops = set(stopwords.words("english"))
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    return txt
    
y = train['sentiment']

# Bag of Words (word based)
ctv_word = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                           min_df = 200, max_features=5000,
            ngram_range=(1,2), stop_words = 'english')

print("Fitting Bag of Words Model on words...\n")
# Fitting CountVectorizer to both training and test sets
ctv_word.fit(list(train['review']) + list(test['review']))
train_ctv_word =  ctv_word.transform(train['review']) 
test_ctv_word = ctv_word.transform(test['review'])

print("Fitting Bag of Words Model on characters...\n")

# Bag of words (charater based)
# Character level analysis of TfidfVectorizer is a process that creates
# ngram from the corpus. In below example, the ngram_range=(2,6), which means
# the size of each gram must be between 2 and 6 characters and it slides through 
# each item with 1 character jump. For example, fitting below TfidfVectorizer on
# "I am superhero" would create following vocabulary 
# 'I ',' a','am',..., then 'I a',' am','am ',... then 'I am',' am ','am s',
# then 'I am ',' am s','am su', ... then 'I am su',' am sup', ...
# then
ctv_char = CountVectorizer(sublinear_tf=True, strip_accents='unicode',analyzer='char',
    stop_words='english', ngram_range=(2, 6), max_features=10000)

# Fitting CountVectorizer to both training and test sets
ctv_char.fit(list(train['review']) + list(test['review']))
train_ctv_char =  ctv_char.transform(train['review']) 
test_ctv_char = ctv_char.transform(test['review'])




# TF - IDF (words)
print("Fitting TF-IDF Model on words...\n")
tfv_word = TfidfVectorizer(min_df=150,  max_features= 5000, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1,2),
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv_word.fit(list(train['review']) + list(test['review']))
train_tfv_word =  tfv_word.transform(train['review'])
test_tfv_word = tfv_word.transform(test['review'])

# TF-IDF(char)
print("Fitting TF - IDF Model on characters...\n")
tfv_char = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='char',
    stop_words='english',ngram_range=(2, 6),max_features=10000)
tfv_char.fit(list(train['review']) + list(test['review']))
train_tfv_char = tfv_char.transform(train['review'])
test_tfv_char = tfv_char.transform(test['review'])

print("Combining Bag of words for words and characters...\n")
# bag of words for training set (words + char)
train_bow = hstack([train_ctv_word, train_ctv_char])
test_bow = hstack([test_ctv_word, test_ctv_char])

print("Combining TF-IDF for words and characters...\n")

# TF-IDF for test set (words + char)
train_tfidf = hstack([train_tfv_word, train_tfv_char])
test_tfidf = hstack([test_tfv_word, test_tfv_char])

clf_lr = LogisticRegression() # Logistic Regression Model

## 5-fold cross validation
print(cross_val_score(clf_lr, train_tfidf, y, cv=5, scoring=make_scorer(f1_score)))

"""
We can see that we are achieving an validation accuracy of 89%. Which is really interesting
"""
# Fit the logistic regression model
clf_lr.fit(train_tfidf,y)

# Make predictions on test data
preds = clf_lr.predict(test_tfidf)

# Make submission

sample['sentiment'] = preds
sample = sample[['id','sentiment']]
sample.to_csv('submissions.csv',index=False)