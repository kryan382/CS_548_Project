# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:41:41 2024

@author: kevry
"""

import pandas as pd

# Path to the master CSV file
master_csv_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Master.csv"

# Load the master CSV
master_df = pd.read_csv(master_csv_path)

# Specify columns that must not contain NaN or empty values
required_columns = ['top_actors', 'director', 'studios', 'rating', 'genres', 'minute']

# Remove rows where any of the required columns have missing values
filtered_df = master_df.dropna(subset=required_columns)

# Filter for movies with a time length between 60 and 240 minutes
filtered_df = filtered_df[(filtered_df['minute'] >= 60) & (filtered_df['minute'] <= 240)]

# Save the filtered DataFrame to a new CSV
filtered_csv_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Filtered.csv"
filtered_df.to_csv(filtered_csv_path, index=False)

print("Filtered CSV with complete data and valid time length created successfully at:", filtered_csv_path)


