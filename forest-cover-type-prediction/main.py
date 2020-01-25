
"""
Forest Cover Type Prediction

@author: Saeed Mohajeryami, PhD

"""
## import basic packages
import numpy as np
import pandas as pd

## import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

## import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

X = training_set.iloc[:,1:-1]
y = training_set.iloc[:,-1].values

## Data Cleansing
#Removal list initialize
rem = []
#Add constant columns as they don't help in prediction process
for c in X.columns:
    if X[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
X.drop(rem,axis=1,inplace=True)
test_set.drop(rem,axis=1,inplace=True)
print(rem)
#Following columns are dropped: ['Soil_Type7', 'Soil_Type15']
X = X.values


## train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


## feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##-------------------------------------------------------------------------
## GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)

y_pred = nb_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
nb_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train,y_train)

y_pred = dt_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
dt_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train,y_train)

y_pred = rf_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
rf_training_accuracy = accuracy_score(y_test,y_pred)

##-------------------------------------------------------------------------
## KNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)

y_pred = knn_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
knn_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## SVM
svc_classifier = SVC()
svc_classifier.fit(X_train,y_train)

y_pred = svc_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
svc_training_accuracy = accuracy_score(y_test,y_pred)

##-------------------------------------------------------------------------
## Extra Trees
ext_classifier = ExtraTreesClassifier(max_depth=6, n_estimators=100)
ext_classifier.fit(X_train,y_train)

y_pred = ext_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
ext_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## Gradient Boosting Classifier
gbc_classifier = GradientBoostingClassifier(max_depth=6, n_estimators=100)
gbc_classifier.fit(X_train,y_train)

y_pred = gbc_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
gbc_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## Bagging 
bag_classifier = BaggingClassifier(n_estimators=100, base_estimator=ext_classifier)
bag_classifier.fit(X_train,y_train)

y_pred = bag_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
bag_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## AdaBoost 
ada_classifier = AdaBoostClassifier(n_estimators=100)
ada_classifier.fit(X_train,y_train)

y_pred = ada_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
ada_training_accuracy = accuracy_score(y_test,y_pred)


##-------------------------------------------------------------------------
## XGBoost
xgb_classifier = XGBClassifier(max_depth=6, n_estimators=200)
xgb_classifier.fit(X_train,y_train)

y_pred = xgb_classifier.predict(X_test)

## metrics
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
xgb_training_accuracy = accuracy_score(y_test,y_pred)


## Model evaluation
models = pd.DataFrame({
    'Model': ['Random Forest', 'Naive Bayes', 'Decision Tree', 'KNN',
              'ExtraTrees','GradientBoosting','XGB', 'Support Vector Machine',
              'Bagging Classifier', 'AdaBoost'],
    'Score': [rf_training_accuracy, nb_training_accuracy, dt_training_accuracy,
              knn_training_accuracy,ext_training_accuracy,gbc_training_accuracy,
              xgb_training_accuracy, svc_training_accuracy,bag_training_accuracy,
              ada_training_accuracy]})
models.sort_values(by='Score', ascending=False)


## Prediction
Test = test_set.iloc[:,1:].values
Test = sc.transform(Test)
y_pred = bag_classifier.predict(Test)
## Writing the results to submission dataframe
submission = pd.DataFrame({
        "Id": test_set["Id"],
        "Cover_Type": y_pred
    })

## Writing the submission results into a csv file
submission.to_csv('submission.csv', index=False)



