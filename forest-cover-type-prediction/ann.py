
"""
Forest Cover Type Prediction

@author: Saeed Mohajeryami, PhD

"""
## import basic packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



## import dataset
dataset = pd.read_csv("train.csv") 
#Drop 'Id'
dataset = dataset.iloc[:,1:]

print(dataset.describe())

## 1. data exploration

# Skewness of the distribution
print(dataset.skew())

# Values close to 0 show less skew
# Several attributes in Soil_Type show a large skew.
# Hence, some algos may benefit if skew is corrected

## 2. data interaction (correlation)
## The high correlation represents an opportunity to reduce the feature set 
## through transformations such as PCA
data=dataset.iloc[:,:10]  # only size features

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

## plot a correlation heatmap
sns.heatmap(data_corr)

# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,10): #for 'size' features
    for j in range(i+1,10): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()


## Data Visualization
# We will visualize all the attributes using Violin Plot - a combination of box and density plots

#names of all the attributes 
cols = dataset.columns
#number of attributes (exclude target)
size = len(cols)-1
#x-axis has target attribute to distinguish between classes
x = cols[size]
#y-axis shows values of an attribute
y = cols[0:size]
#Plot violin for all attributes
for i in range(0,size):
    sns.violinplot(data=dataset,x=x,y=y[i])  
    plt.show()

#Elevation is has a separate distribution for most classes. Highly correlated with the target and hence an important attribute
#Aspect contains a couple of normal distribution for several classes
#Horizontal distance to road and hydrology have similar distribution
#Hillshade 9am and 12pm display left skew
#Hillshade 3pm is normal
#Lots of 0s in vertical distance to hydrology
#Wilderness_Area3 gives no class distinction. As values are not present, others gives some scope to distinguish
#Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes
    
## Data Cleansing
#Removal list initialize
rem = []
#Add constant columns as they don't help in prediction process
for c in dataset.columns:
    if dataset[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
dataset.drop(rem,axis=1,inplace=True)
print(rem)
#Following columns are dropped: ['Soil_Type7', 'Soil_Type15']


## Artificial Neural Network
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

## train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Import libraries for deep learning
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

#Import libraries for encoding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

#no. of output classes
y = 7
r, c = dataset.shape
v = 1

#random state
np.random.seed(0)

# one hot encode class values
encoder = LabelEncoder()
y_train_en = encoder.fit_transform(y_train)
y_train_hot = np_utils.to_categorical(y_train_en,y) 
y_test_en = encoder.transform(y_test)
y_test_hot = np_utils.to_categorical(y_test_en,y) 


# define baseline model
def baseline(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(y, init='normal', activation='sigmoid'))
     # Compile model
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     return model

# define smaller model
def smaller(v):
 # create model
 model = Sequential()
 model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
 model.add(Dense(y, init='normal', activation='sigmoid'))
 # Compile model
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# define deeper model
def deeper(v):
 # create model
 model = Sequential()
 model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
 model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
 model.add(Dense(y, init='normal', activation='sigmoid'))
 # Compile model
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# Optimize using dropout and decay
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm

def dropout(v):
    #create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu',W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(v*(c-1)/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(y, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# define decay model
def decay(v):
    # create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
    model.add(Dense(y, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]


model = model = KerasClassifier(build_fn=baseline, v=v, nb_epoch=10, verbose=0)
model.fit(X_train,y_train_hot)
acc = model.score(X_train,y_train_hot)