# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:00:51 2024

Simple file to determine which features are chosen first, data for report

@author: kevry
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer

# Load the dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Filtered.csv"
MovieData = pd.read_csv(file_path)

# Define the target and drop unnecessary columns
target = 'rating'
MovieData = MovieData.drop(columns=['id', 'date', 'minute'])

# Apply count encoding to categorical columns
for column in ['top_actors', 'director', 'studios', 'genres']:
    if column in MovieData.columns:
        MovieData[f"{column}_count"] = MovieData[column].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) else 0)

# Drop the original categorical columns after encoding
MovieData = MovieData.drop(columns=['top_actors', 'director', 'studios', 'genres'])

# Define input features X and target y
X = MovieData.drop(columns=[target])
y = MovieData[target]

# Ensure only numeric columns remain in X
X = X.select_dtypes(include=[np.number])

# Check if any non-numeric columns were accidentally retained
if X.shape[1] < MovieData.shape[1] - 1:
    print("Non-numeric columns were removed from X.")

# Normalize numerical features
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Initialize KNN Regressor and custom scorers (as defined earlier)
# Initialize KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')

# Define custom scorers
def Accuracy_score(orig, pred):
    orig = 10.0 ** orig
    pred = 10.0 ** pred
    MAPE = np.mean(np.abs((orig - pred)) / orig)
    return MAPE

def Accuracy_score3(orig, pred):
    orig = 10 ** np.array(orig)
    pred = 10 ** np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if (pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score, greater_is_better=True)
custom_Scoring3 = make_scorer(Accuracy_score3, greater_is_better=True)


# Initialize a dictionary to store selected features for each k
selected_features_dict = {}

# Iterate over k values from 1 to 9
for k in range(1, 5):
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X_normalized, y)

    # Get selected features
    selected_features_mask = selector.get_support()
    selected_features = X.columns[selected_features_mask]
    
    # Store selected features for the current k, reversed order
    selected_features_dict[f'k={k}'] = selected_features[::-1].tolist()

# Convert the dictionary to a DataFrame
df_selected_features = pd.DataFrame.from_dict(selected_features_dict, orient='index').T

# Flip the DataFrame so that the first chosen feature is on top
df_selected_features = df_selected_features.apply(lambda x: pd.Series(x.dropna().values))

# Export the selected features table to Excel
export_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Selected_Features.xlsx"
with pd.ExcelWriter(export_path, mode='a', engine='openpyxl') as writer:
    df_selected_features.to_excel(writer, sheet_name='Selected_Features', index=False)

print(f"Selected features for each k have been successfully exported to {export_path}")
