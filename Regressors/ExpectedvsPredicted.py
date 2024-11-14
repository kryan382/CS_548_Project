# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:25:59 2024

@author: kevry
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
X = MovieData[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3', 'studio_4']]
y = MovieData['rating']  

# Feature selection
selector = SelectKBest(f_regression, k=11)
X = selector.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
rfr = RandomForestRegressor(n_estimators=25, random_state=17, max_depth=9)
gbr = GradientBoostingRegressor(max_depth=8,random_state=17,n_estimators=50,learning_rate=0.1)

#Which regressor is being used IMPORTANT *****
regressor = gbr
#%% Test and analysis

# Train the model
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# y_test = 10 ** y_test 
# y_pred = 10 ** y_pred 

#RSME Calculations
def rsm_error(actual,predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

#Mape calculations
def Accuracy_score(orig,pred):
    MAPE = np.mean((np.abs(orig-pred))/orig)
    return(MAPE)


def Accuracy_score3(orig,pred):
    orig =  np.array(orig)
    pred =  np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count = count +1
    a_20 = count/len(orig)
    return(a_20)

#Statistical calculations
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
mape=Accuracy_score(y_test, y_pred)
rmse=rsm_error(y_test,y_pred)
a_20 = Accuracy_score3(y_test,y_pred)

#Print statements
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2:",r2)
print("MAPE:",mape)
print("RSME:", rmse)
print("A_20:", a_20)

print("Y_test\n",y_test)
# print("The number of values in y_test",len(y_test))
print("Y_pred",y_pred)