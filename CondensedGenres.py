# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:25:10 2024

@author: kevry
"""

import pandas as pd

# Path to the Genres.csv file
genres_file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Genres.csv"

# Load the data
genres_df = pd.read_csv(genres_file_path)

# Convert the 'genre' column to strings, replacing NaN or missing values with an empty string
genres_df['genre'] = genres_df['genre'].fillna('').astype(str)

# Group by 'id' and combine genres into a single string separated by commas
combined_genres = genres_df.groupby('id')['genre'].apply(lambda x: ', '.join(x)).reset_index()

# Rename the combined column to 'genres' for clarity
combined_genres = combined_genres.rename(columns={'genre': 'genres'})

# Save the condensed DataFrame to a new CSV
condensed_genres_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Condensed_Genres.csv"
combined_genres.to_csv(condensed_genres_path, index=False)

print("Condensed genres file created successfully at:", condensed_genres_path)
