#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:07:43 2019

@author: Saeed Mohajeryami, PhD

Title: House Prices (Kaggle Competition)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

training_set = pd.read_csv('train.csv')


#----------------------------------------------------------------
# Learning about the data
#----------------------------------------------------------------

## view columns
training_set.columns

## brief data examination
training_set['SalePrice'].describe()
sns.distplot(training_set['SalePrice'])


## Skewness
## Skewness refers to distortion or asymmetry in a symmetrical bell curve. If 
## the curve is shifted to the left/right, it is said to be skewed.
print("Skewness: %f" %training_set['SalePrice'].skew())

## Kurtosis
## Kurtosis is the sharpness of the peak of a distribution curve. So, the higher
## the number, the sharper is the peak.
print("Kurtosis: %f" %training_set['SalePrice'].kurt())

## scatterplot of GrLivArea/SalePrice 
training_set['GrLivArea'].describe()
var = 'GrLivArea' # Above grade (Ground) living area square feet (all floors)
data = pd.concat([training_set['SalePrice'], training_set[var]],axis = 1)
data.plot.scatter(x=var,y='SalePrice', ylim = (0,800000))
## scatterplot of TotalBsmtSF/SalePrice
training_set['TotalBsmtSF'].describe()
var = 'TotalBsmtSF' # Total sqare feet of basement area
data = pd.concat([training_set['SalePrice'], training_set[var]],axis = 1)
data.plot.scatter(x=var,y='SalePrice', ylim = (0,800000))

## relationship w/ categorical features
## box plot OverallQual/SalePrice (Overall material and finish quality)
sns.countplot(training_set['OverallQual'])
var = 'OverallQual'
data = pd.concat([training_set['SalePrice'], training_set[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

## box plot YearBuilt/SalePrice (Overall material and finish quality)
var = 'YearBuilt'
data = pd.concat([training_set['SalePrice'], training_set[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

## Correlation matrix
corrmat = training_set.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = training_set[cols].corr()
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(training_set[cols], size = 2.5)
plt.show()

#----------------------------------------------------------------
# How to deal with missing data problem?
#----------------------------------------------------------------
#Important questions when thinking about missing data:
#   How prevalent is the missing data?
#   Is missing data random or does it have a pattern?
#The answer to these questions is important for practical reasons
# because missing data can imply a reduction of the sample size.
# This can prevent us from proceeding with the analysis. Moreover,
# from a substantive perspective, we need to ensure that the missing data
# process is not biased and hidding an inconvenient truth.

total = training_set.isnull().sum().sort_values(ascending=False)
percent = (training_set.isnull().sum()/training_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# We'll consider that when more than 15% of the data is missing, we should
# delete the corresponding variable and pretend it never existed. 


# solution: 
# 1. features like PoolQC, MiscFeature, and Alley have more than 85% missing data
# we will delete the column
# 2. For Garage features, since we have another feature GarageCars, we are somehow
# assured that we have kept enough info about the garage. so, we can delete them
# all
# 3. Features like MasVnrArea and MasVnrType are not essential. 
# 4. For "Electrical", we keep it, but we delete the row that has this missing value

training_set = training_set.drop((missing_data[missing_data['Total'] > 1]).index,1)
training_set = training_set.drop(training_set.loc[training_set['Electrical'].isnull()].index)
training_set.isnull().sum().max() #just checking that there's no missing data mi
