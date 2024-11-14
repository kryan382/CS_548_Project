# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:14:02 2024

Encoder to give numerical values to the input parameters. This allows us to use
them in the ML models

@author: kevry
"""

import pandas as pd
import json

# File paths
file_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Split.csv"
encoded_output_path = "C:/Homework Assignments/MachineLearning548/Final Project/Dataset/MoviesDataset_Encoded.csv"

# Load the data
MovieData = pd.read_csv(file_path)

# Initialize dictionaries to store unique IDs for each category
actor_name_to_id, studio_name_to_id, genre_name_to_id, director_name_to_id = {}, {}, {}, {}
current_actor_id, current_studio_id, current_genre_id, current_director_id = 1, 1, 1, 1

# Define encoding functions for each column
def encode_actor(actor):
    global current_actor_id
    if actor not in actor_name_to_id:
        actor_name_to_id[actor] = current_actor_id
        current_actor_id += 1
    return actor_name_to_id[actor]

def encode_studio(studio):
    global current_studio_id
    if studio not in studio_name_to_id:
        studio_name_to_id[studio] = current_studio_id
        current_studio_id += 1
    return studio_name_to_id[studio]

def encode_genre(genre):
    global current_genre_id
    if genre not in genre_name_to_id:
        genre_name_to_id[genre] = current_genre_id
        current_genre_id += 1
    return genre_name_to_id[genre]

def encode_director(director):
    global current_director_id
    if director not in director_name_to_id:
        director_name_to_id[director] = current_director_id
        current_director_id += 1
    return director_name_to_id[director]

# Encode actor columns
for col in ['actor_1', 'actor_2', 'actor_3']:
    MovieData[col] = MovieData[col].fillna('').apply(encode_actor)

# Encode studio columns
for col in ['studio_1', 'studio_2', 'studio_3', 'studio_4']:
    MovieData[col] = MovieData[col].fillna('').apply(encode_studio)

# Encode genre columns
for col in ['genre_1', 'genre_2', 'genre_3']:
    MovieData[col] = MovieData[col].fillna('').apply(encode_genre)

# Encode director column
MovieData['director'] = MovieData['director'].fillna('').apply(encode_director)

# Save the updated DataFrame to the specified output path
MovieData.to_csv(encoded_output_path, index=False)
print(f"Encoded dataset saved to {encoded_output_path}")

# Save dictionaries to JSON files
with open("actor_name_to_id.json", 'w') as f:
    json.dump(actor_name_to_id, f)
with open("studio_name_to_id.json", 'w') as f:
    json.dump(studio_name_to_id, f)
with open("genre_name_to_id.json", 'w') as f:
    json.dump(genre_name_to_id, f)
with open("director_name_to_id.json", 'w') as f:
    json.dump(director_name_to_id, f)

print("Encoding dictionaries saved successfully for actors, studios, genres, and directors.")
