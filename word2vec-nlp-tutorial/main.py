#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:58:53 2019

@author: saeed
"""

import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix

import os
for file in os.listdir():
    print(file)
    
## Set some matplotlib configs for visualization    
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
LARGE_SIZE = 20

params = {
    'figure.figsize': (16, 8),
    'font.size': SMALL_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'legend.fontsize': BIG_SIZE,
    'figure.titlesize': LARGE_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': BIG_SIZE
}
plt.rcParams.update(params)


## Importing the dataset
usecols = ['sentiment','review']
train_data = pd.read_csv(
    filepath_or_buffer='~/Kaggle-Competitions/word2vec-nlp-tutorial/labeledTrainData.tsv',
    usecols=usecols, sep='\t')
additional_data = pd.read_csv(
    filepath_or_buffer='~/Kaggle-Competitions/word2vec-nlp-tutorial/imdb_master.csv',
    encoding="latin-1")[usecols]
unlabeled_data = pd.read_csv(
    filepath_or_buffer="~/Kaggle-Competitions/word2vec-nlp-tutorial/unlabeledTrainData.tsv", 
    error_bad_lines=False,
    sep='\t')
submission_data = pd.read_csv(
    filepath_or_buffer="~/Kaggle-Competitions/word2vec-nlp-tutorial/testData.tsv",
    sep='\t')

