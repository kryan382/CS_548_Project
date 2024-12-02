# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:30:49 2024

@author: kevry
"""

import pickle
from sklearn.ensemble import RandomForestRegressor  # Replace with your chosen model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# Load dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/CSV/MoviesDataset_Encoded.csv"
MovieData = pd.read_csv(file_path)

# Define features and target
X = MovieData[['actor_1', 'actor_2', 'actor_3', 'director', 'genre_1', 'genre_2', 'genre_3', 'studio_1', 'studio_2', 'studio_3', 'studio_4']]
y = MovieData['rating']

# Feature selection
selector = SelectKBest(f_regression, k=11)
X = selector.fit_transform(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Train the final model
final_model = RandomForestRegressor(n_estimators=200, random_state=17, max_depth=7)  # Replace with your chosen regressor and hyperparameters
final_model.fit(X_train, y_train)

# Save the trained model to a pickle file
model_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/FinalModel.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

print(f"Final model saved to {model_path}")
