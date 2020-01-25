"""
Titanic

@author: Saeed Mohajeryami, PhD

"""

## import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## import classification tools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

## Know your data

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]  ## combine both dataframes to perform cleaning on both

## Data Description
print(train_df.columns.values)
## Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
## Numerical --> Continous: Age, Fare. Discrete: SibSp, Parch.

## How many null variables??
print(train_df.isnull().sum())   ## Cabin(687) > Age(177) > Embarked(2)
print(test_df.isnull().sum())    ## Cabin(327) > Age(86)

## Describe numerical features
train_df.describe()
## Describe categorical features
train_df.describe(include=['O'])

## The results of data exploration phase
## 1. Ticket feature may be dropped from our analysis as it contains high ratio of
## duplicates (22%) and there may not be a correlation between Ticket and survival.
## 2. Cabin feature may be dropped as it is highly incomplete or contains many null 
## values both in training and test dataset.
## 3. PassengerId may be dropped from training dataset as it does not contribute to
## survival.
## 4. Name feature is relatively non-standard, may not contribute directly to survival,
## so maybe dropped.

## Correlation Analysis
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

## Visualization
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

## Droping Cabin and Ticket
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

## Feature Engineering
## Extracting titles from the Name column
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])

## We can replace many titles with a more common name or classify them as Rare.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

## We can convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
## Droping Name and PassengerId
## I kept PassengerId in test_df to use it for submission.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

## Now we convert features which contain strings to numerical values.
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
## Completing a numerical continuous feature:
## Now we should start estimating and completing features with missing or null values.
## We will first do this for the Age feature.
## We can consider three methods to complete a numerical continuous feature:
## 1. A simple way is to generate random numbers between mean and standard deviation.
## 2. More accurate way of guessing missing values is to use other correlated
## features. In our case we note correlation among Age, Gender, and Pclass. 
## Guess Age values using median values for Age across sets of Pclass and Gender
## feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and
## Gender=1, and so on...
## 3. Combine methods 1 and 2. So instead of guessing age values based on median, 
## use random numbers between mean and standard deviation, based on sets of Pclass
## and Gender combinations.
#DECISION: Method 1 and 3 will introduce random noise into our models. The results
## from multiple executions might vary. We will prefer method 2.
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
    
 ## Let us create Age bands and determine correlations with Survived.   
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

## Dropping AgeBand
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

## We can create a new feature for FamilySize which combines Parch and SibSp. 
## This will enable us to drop Parch and SibSp from our datasets.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

## We can create another feature called IsAlone.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

## Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

## We can also create an artificial feature combining Pclass and Age.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

## Embarked feature takes S, Q, C values based on port of embarkation.
## Our training dataset has two missing values. We simply fill these with the
## most common occurance.
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

## Converting categorical feature to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

## Replace Fare missing values with the most common occurance
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

## Let's create Fare bands
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
combine = [train_df, test_df]

## Convert the Fare feature to ordinal values based on the FareBand.
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]


### Modeling and Predict
X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

## Logistic Regression
log_classifier = LogisticRegression()
log_classifier.fit(X_train, y_train)
y_pred = log_classifier.predict(X_test)

log_training_accuracy = round(log_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', log_training_accuracy)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(log_classifier.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

## Support Vector Classifier
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred = svc_classifier.predict(X_test)

svc_training_accuracy = round(svc_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', svc_training_accuracy)

## K-Nearest Neighbors (KNN)
## In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short)
## is a non-parametric method used for classification and regression.
## A sample is classified by a majority vote of its neighbors, with the sample
## being assigned to the class most common among its k nearest neighbors.
## Support Vector Classifier
knn_classifier = KNeighborsClassifier(n_neighbors = 3)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

knn_training_accuracy = round(knn_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', knn_training_accuracy)

## Naiva Bayes Classifier
## In machine learning, naive Bayes classifiers are a family of simple probabilistic
## classifiers based on applying Bayes' theorem with strong (naive) independence 
## assumptions between the features. Naive Bayes classifiers are highly scalable,
## requiring a number of parameters linear in the number of variables (features) \
## in a learning problem.
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

nb_training_accuracy = round(nb_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', nb_training_accuracy)

## Perceptron
##The perceptron is an algorithm for supervised learning of binary classifiers
## (functions that can decide whether an input, represented by a vector of numbers, 
## belongs to some specific class or not). It is a type of linear classifier, 
## i.e. a classification algorithm that makes its predictions based on a linear 
## predictor function combining a set of weights with the feature vector. 
## The algorithm allows for online learning, in that it processes elements 
## in the training set one at a time.
perc_classifier = Perceptron()
perc_classifier.fit(X_train, y_train)
y_pred = perc_classifier.predict(X_test)

perc_training_accuracy = round(perc_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', perc_training_accuracy)


## Linear SVC
lsvc_classifier = LinearSVC()
lsvc_classifier.fit(X_train, y_train)
y_pred = lsvc_classifier.predict(X_test)

lsvc_training_accuracy = round(lsvc_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', lsvc_training_accuracy)

## Stochastic Gradient Descent
sgd_classifier = SGDClassifier()
sgd_classifier.fit(X_train, y_train)
y_pred = sgd_classifier.predict(X_test)

sgd_training_accuracy = round(sgd_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', sgd_training_accuracy)

## Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

dt_training_accuracy = round(dt_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', dt_training_accuracy)

## Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

rf_training_accuracy = round(rf_classifier.score(X_train, y_train) * 100, 2)
print('training accuracy is =', rf_training_accuracy)

## Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [svc_training_accuracy, knn_training_accuracy, log_training_accuracy, 
              rf_training_accuracy, nb_training_accuracy, perc_training_accuracy, 
              sgd_training_accuracy, lsvc_training_accuracy, dt_training_accuracy]})
models.sort_values(by='Score', ascending=False)

## Writing the results to submission dataframe
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })

## Writing the submission results into a csv file
submission.to_csv('submission.csv', index=False)