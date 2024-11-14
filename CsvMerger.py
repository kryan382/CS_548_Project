# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:28:26 2024

@author: kevry
"""

import pandas as pd
import os

# Define the path to each CSV file
base_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/"
movies_file = os.path.join(base_path, "Movies.csv")
actors_file = os.path.join(base_path, "Condensed_Actors.csv")
crew_file = os.path.join(base_path, "Condensed_Crew.csv")
studios_file = os.path.join(base_path, "Condensed_Studios.csv")
genres_file = os.path.join(base_path, "Condensed_Genres.csv")

# Load the main Movies.csv
movies_df = pd.read_csv(movies_file)

# Load each condensed file
actors_df = pd.read_csv(actors_file)
crew_df = pd.read_csv(crew_file)
studios_df = pd.read_csv(studios_file)
genres_df = pd.read_csv(genres_file)

# Merge all files into a single master DataFrame on 'id'
master_df = movies_df.merge(actors_df, on="id", how="left")
master_df = master_df.merge(crew_df, on="id", how="left")
master_df = master_df.merge(studios_df, on="id", how="left")
master_df = master_df.merge(genres_df, on="id", how="left")

# Save the master DataFrame to a new CSV
master_csv_path = os.path.join(base_path, "MoviesDataset_Master.csv")
master_df.to_csv(master_csv_path, index=False)

print("Master CSV created successfully at:", master_csv_path)
