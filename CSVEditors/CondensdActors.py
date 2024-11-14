# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:39:21 2024

@author: kevry
"""

import pandas as pd

# Path to the Actors.csv file
actors_file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Actors.csv"

# Load the data
actors_df = pd.read_csv(actors_file_path)

# Group by 'id' (movie_id) and retain the first three actors for each movie
top_three_actors = actors_df.groupby('id').apply(lambda x: x.head(3)).reset_index(drop=True)

# Ensure only the relevant columns are retained: id and name (actor's name)
top_three_actors = top_three_actors[['id', 'name']]

# Convert the 'name' column to strings, replacing NaN or missing values with an empty string
top_three_actors['name'] = top_three_actors['name'].fillna('').astype(str)

# Combine the top three actors' names into a single cell per movie, separated by commas
combined_actors = top_three_actors.groupby('id')['name'].apply(lambda x: ', '.join(x)).reset_index()

# Rename the column to 'top_actors' for clarity
combined_actors = combined_actors.rename(columns={'name': 'top_actors'})

# Save the condensed DataFrame to a new CSV
condensed_actors_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Condensed_Actors.csv"
combined_actors.to_csv(condensed_actors_path, index=False)

print("Condensed actors file with top three actors combined created successfully at:", condensed_actors_path)


