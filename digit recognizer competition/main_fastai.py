"""
Title: Digit Recognizer (Kaggle Competition)
Created on Sun Jan 26 19:19:10 2020

@author: Saeed Mohajeryami, PhD

"""

import numpy as np
import pandas as pd
from PIL import Image  # display images and to create images from arrays

# the fast.ai library, used to easily build neural networks and train them
from fastai import *
from fastai.vision import *


import os # to get all files from a directory
from pathlib import Path # to easier work with paths


INPUT = Path("/home/saeed/On Github/Kaggle-Competitions/digit recognizer competition")
os.listdir(str(INPUT))

## import the dataset
train = pd.read_csv("train.csv")
test =  pd.read_csv("test.csv")



try:
    os.makedirs(TEST)
except:
    pass

def saveDigit(digit, filepath):
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)

    img = Image.fromarray(digit)
    img.save(filepath)