# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:58:32 2024

@author: kevry
"""

import pandas as pd

# Path to the Studios.csv file
studios_file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Studios.csv"

# Load the data
studios_df = pd.read_csv(studios_file_path)

# Convert 'studio' column values to strings, replace NaN with empty string
studios_df['studio'] = studios_df['studio'].fillna('').astype(str)

# Group by 'id' and combine studios into a single string separated by commas
combined_studios = studios_df.groupby('id')['studio'].apply(lambda x: ', '.join(x)).reset_index()

# Rename the combined column to 'studios' for clarity
combined_studios = combined_studios.rename(columns={'studio': 'studios'})

# Save the condensed DataFrame to a new CSV
combined_studios_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Condensed_Studios.csv"
combined_studios.to_csv(combined_studios_path, index=False)

print("Condensed studios file created successfully at:", combined_studios_path)
