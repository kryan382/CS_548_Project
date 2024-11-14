# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:00:21 2024

This code splits the actor, genre and studio columns into seperate columns 
to be easier used by the ml models

@author: kevry
"""

import pandas as pd

# Load the dataset
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Filtered.csv"
MovieData = pd.read_csv(file_path)

# Split top_actors into three separate columns
actor_columns = MovieData['top_actors'].str.split(', ', expand=True)
actor_columns = actor_columns.rename(columns={0: 'actor_1', 1: 'actor_2', 2: 'actor_3'})
MovieData = pd.concat([MovieData, actor_columns[['actor_1', 'actor_2', 'actor_3']]], axis=1)

# Split genres into multiple columns
genre_columns = MovieData['genres'].str.split(', ', expand=True)
max_genres = genre_columns.shape[1]
genre_columns = genre_columns.rename(columns={i: f'genre_{i+1}' for i in range(max_genres)})
MovieData = pd.concat([MovieData, genre_columns], axis=1)

# Split studios into multiple columns
studio_columns = MovieData['studios'].str.split(', ', expand=True)
max_studios = studio_columns.shape[1]
studio_columns = studio_columns.rename(columns={i: f'studio_{i+1}' for i in range(max_studios)})
MovieData = pd.concat([MovieData, studio_columns], axis=1)

# Drop the original combined columns
MovieData = MovieData.drop(columns=['top_actors', 'genres', 'studios'])

# Save the modified dataset
output_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Split.csv"
MovieData.to_csv(output_path, index=False)

print(f"Dataset with split columns saved successfully at {output_path}")
