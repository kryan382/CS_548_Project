# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:33:58 2024

Linear Regression Build for Project

@author: kevry
"""
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
#%% Initilization 
# Define the file path to your encoded dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/CSV/MoviesDataset_Encoded.csv"

# Load the data
MovieData = pd.read_csv(file_path)

# Define X and y
X = MovieData[['actor_1', 'director', 'genre_1', 'studio_1','studio_2','studio_3','genre_2','genre_3','actor_2','actor_3']]
y = MovieData['rating']  

# Feature selection
selector = SelectKBest(f_regression, k=10)
X = selector.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#%% Scorers
def Accuracy_score(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    mape = np.mean(np.abs((orig - pred) / np.abs(orig))) 
    return mape

def Accuracy_score2(orig,pred):
    MAE = np.mean((np.abs(orig-pred)))
    return(MAE)

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)

    count = 0
    for i in range(len(orig)):
        if (abs(pred[i]) <= 1.2 * abs(orig[i])) and (abs(pred[i]) >= 0.8 * abs(orig[i])):
            count += 1
    a_20 = count / len(orig)
    return a_20


# Custom Scoring MAPE
custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)
#custom scoring MAE calulation
custom_Scoring2 = make_scorer(Accuracy_score2,greater_is_better=True)
#custom scoring a_20 calulation
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

mape_values = []
a_20values = []
trial = 0
mae_values = []
#%% Cross validation MAPE

#Initlize
lr = LinearRegression()
cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
Accuracy_Values = cross_val_score(lr, X_train, y_train, cv = cv_scores, scoring = custom_Scoring)

mape_values.append(Accuracy_Values)
trial = trial +1
print('Trial #:',trial)
print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
#%% Cross Validation A_20

#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values3 = cross_val_score(lr,X_train,y_train,\
                                   cv=CV,scoring=custom_Scoring3)

print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))

a_20values.append(Accuracy_Values3)

#%%
#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values2 = cross_val_score(lr,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring2)
mae_values.append(Accuracy_Values2)
print('\n"MAE index" for 5-fold Cross Validation:\n', Accuracy_Values2)
print('\nFinal Average Accuracy MAE index of the model:', round(Accuracy_Values2.mean(),4))

#%% Export
#MAPE
df_metrics = pd.DataFrame(mape_values, index=range(1, 2), columns=range(1, 16))   

#MAE
df_metricsMAE = pd.DataFrame(mae_values, index=range(1, 2), columns=range(1, 16))   

#A20
df_metricsA20 = pd.DataFrame(a_20values, index=range(1, 2), columns=range(1, 16))   
    
file_path = "C:/spydertest/csv/CumulRot.xlsx"

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='LNRtrainMAPE', index=False, startrow=0, startcol=0)
    df_metricsMAE.to_excel(writer, sheet_name='LNRtrainMAE', index=False, startrow=0, startcol=0)
    df_metricsA20.to_excel(writer, sheet_name='LNRtrainA20', index=False, startrow=0, startcol=0)
    
print("\nSuccesfully Exported to Excel!")
