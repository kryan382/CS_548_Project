# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:50:17 2024

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

#%% Initialization
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/CSV/MoviesDataset_Encoded.csv"

MovieData = pd.read_csv(file_path)

X = MovieData[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3', 'studio_4']]
y = MovieData['rating']  

y = np.log10(y)

selector = SelectKBest(f_regression, k=11)
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

#%% Regressors
lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
rfr = RandomForestRegressor(n_estimators=200, random_state=17, max_depth=7)
gbr = GradientBoostingRegressor(max_depth=8, random_state=17, n_estimators=50, learning_rate=0.1)

regressor = rfr

#%% Train the model
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_test = 10 ** y_test
y_pred = 10 ** y_pred

#%% Evaluation metrics
def rsm_error(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

def Accuracy_score(orig, pred):
    return np.mean((np.abs(orig - pred)) / orig)

def Accuracy_score3(orig, pred):
    count = np.sum((pred >= 0.8 * orig) & (pred <= 1.2 * orig))
    return count / len(orig)

def evaluate_performance(y_true, y_pred, segment_name="Overall"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = Accuracy_score(y_true, y_pred)
    rmse = rsm_error(y_true, y_pred)
    a_20 = Accuracy_score3(y_true, y_pred)
    
    print(f"\nPerformance Metrics for {segment_name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2: {r2}")
    print(f"MAPE: {mape}")
    print(f"RSME: {rmse}")
    print(f"A_20: {a_20}")

# Evaluate overall performance
evaluate_performance(y_test, y_pred)

#%% Segment Analysis
low_ratings = MovieData[MovieData['rating'] < 2.5]
mid_ratings = MovieData[(MovieData['rating'] >= 2.5) & (MovieData['rating'] < 3.5)]
high_ratings = MovieData[MovieData['rating'] >= 3.5]

segments = [("Low Ratings", low_ratings), ("Mid Ratings", mid_ratings), ("High Ratings", high_ratings)]

for segment_name, segment_data in segments:
    X_seg = segment_data[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3', 'studio_4']]
    y_seg = segment_data['rating']
    y_seg_log = np.log10(y_seg)

    # Use the trained model to predict
    X_seg = selector.transform(X_seg)
    y_pred_seg = regressor.predict(X_seg)

    y_seg = 10 ** y_seg_log
    y_pred_seg = 10 ** y_pred_seg

    # Evaluate performance for the segment
    evaluate_performance(y_seg, y_pred_seg, segment_name)
