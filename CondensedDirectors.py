# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:48:01 2024

@author: kevry
"""

import pandas as pd

# Path to the Crew.csv file
crew_file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Crew.csv"

# Load the data
crew_df = pd.read_csv(crew_file_path)

# Filter to keep only rows where the role is 'Director'
directors_df = crew_df[crew_df['role'] == 'Director']

# Group by 'id' and keep only the first director for each movie
directors_df = directors_df.groupby('id').first().reset_index()

# Keep only relevant columns: id (movie_id) and name (director's name)
directors_df = directors_df[['id', 'name']]

# Rename the column to make it clear that this is the director
directors_df = directors_df.rename(columns={'name': 'director'})

# Save the condensed DataFrame to a new CSV
directors_csv_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/Condensed_Crew.csv"
directors_df.to_csv(directors_csv_path, index=False)

print("Condensed crew file with only the first director created successfully at:", directors_csv_path)

