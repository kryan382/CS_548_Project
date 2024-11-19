# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:42:57 2024

@author: kevry
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilization 
# Define the file path to your encoded dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/CSV/MoviesDataset_Encoded.csv"

# Load the data
MovieData = pd.read_csv(file_path)

# Define X and y
X = MovieData[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3','studio_4']]
y = MovieData['rating']  

# Feature selection
selector = SelectKBest(f_regression, k=10)
X = selector.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#%% Scorers
def Accuracy_score(orig, pred):
    # orig = 10.0 ** orig
    # pred = 10.0 ** pred
    
    orig = np.array(orig)
    pred = np.array(pred)
    
    mape = np.mean(np.abs((orig - pred) / np.abs(orig))) 
    return mape

def Accuracy_score2(orig,pred):
    # orig = 10.0 ** orig
    # pred = 10.0 ** pred
    MAE = np.mean((np.abs(orig-pred)))
    return(MAE)

def Accuracy_score3(orig,pred):
    # orig = 10.0 ** orig
    # pred = 10.0 ** pred
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
#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
rfr = RandomForestRegressor(n_estimators=25, random_state=17, max_depth=9)

#Which regressor is being used IMPORTANT *****
regressor = lnr
#%% MAPE

cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
Accuracy_Values = cross_val_score(regressor, X, y, cv = cv_scores, scoring = custom_Scoring)

print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))

#%% MAE

#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values2 = cross_val_score(regressor,X ,y,\
                                   cv=CV,scoring=custom_Scoring2)

print('\n"MAE " for 5-fold Cross Validation:\n', Accuracy_Values2)
print('\nFinal Average MAE index of the model:', round(Accuracy_Values2.mean(),4))
#%% A-20


#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values3 = cross_val_score(regressor,X ,y,\
                                   cv=CV,scoring=custom_Scoring3)

print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))

#%% Export
#mape
# MAPE
df_metrics = pd.DataFrame(Accuracy_Values, index=range(1, 16), columns=range(1, 2))   

#MAE
df_metricsMAE = pd.DataFrame(Accuracy_Values2, index=range(1, 16), columns=range(1, 2))   
#A20
df_metricsa20 = pd.DataFrame(Accuracy_Values3, index=range(1, 16), columns=range(1, 2))     
    
file_path = "C:/spydertest/csv/CumulRot.xlsx"
if regressor == lnr:
    #Export the DataFrame to an Excel file on a specific sheet
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='LinearFinalMape', index=False, startrow=0, startcol=0)
        df_metricsMAE.to_excel(writer, sheet_name='LinearFinalMAE', index=False, startrow=0, startcol=0)
        df_metricsa20.to_excel(writer, sheet_name='LinearFinalA20', index=False, startrow=0, startcol=0)
        
elif regressor == knn:
    #Export the DataFrame to an Excel file on a specific sheet
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='KNNFinalMape', index=False, startrow=0, startcol=0)
        df_metricsMAE.to_excel(writer, sheet_name='KNNFinalMAE', index=False, startrow=0, startcol=0)
        df_metricsa20.to_excel(writer, sheet_name='KNNFinalA20', index=False, startrow=0, startcol=0)    
    
else:
    #Export the DataFrame to an Excel file on a specific sheet
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='RFRFinalMape', index=False, startrow=0, startcol=0)
        df_metricsMAE.to_excel(writer, sheet_name='RFRFinalMAE', index=False, startrow=0, startcol=0)
        df_metricsa20.to_excel(writer, sheet_name='RFRFinalA20', index=False, startrow=0, startcol=0)