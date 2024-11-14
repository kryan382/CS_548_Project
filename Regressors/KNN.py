# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:48:49 2024

@author: kevry
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
#%% Initilization 
# Define the file path to your encoded dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/CSV/MoviesDataset_Encoded.csv"

# Load the data
MovieData = pd.read_csv(file_path)

# Define X and y
X = MovieData[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3', 'studio_4']]
y = MovieData['rating']  # Assuming 'rating' is the column name for the target variable

# Feature selection
selector = SelectKBest(f_regression, k=11)
X = selector.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#%% Scorers
def Accuracy_score(orig,pred):
    MAPE = np.mean((np.abs(orig-pred))/orig)
    return(MAPE)


def Accuracy_score2(orig,pred):
    MAE = np.mean((np.abs(orig-pred)))
    return(MAE)


def Accuracy_score3(orig,pred):
    orig =  np.array(orig)
    pred =  np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count = count +1
    a_20 = count/len(orig)
    return(a_20)


# Custom Scoring MAPE
custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)
#custom scoring MAE calulation
custom_Scoring2 = make_scorer(Accuracy_score2,greater_is_better=True)
#custom scoring a_20 calulation
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

mape_values = []
trial = 0
a_20values = []
mae_values = []
#%% Test
for neighbor in range(1,101):
    knn = KNeighborsRegressor(n_neighbors=neighbor, weights='distance')
    cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
    Accuracy_Values = cross_val_score(knn, X_train, y_train, cv = cv_scores, scoring = custom_Scoring)
    trial = trial +1
        
    mape_values.append(Accuracy_Values)
    print('Trial #:',trial)
    print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
    print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
    
    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values2 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring2)
    mae_values.append(Accuracy_Values2)
    print('\n"MAE index" for 5-fold Cross Validation:\n', Accuracy_Values2)
    print('\nFinal Average Accuracy MAE index of the model:', round(Accuracy_Values2.mean(),4))


    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values3 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))