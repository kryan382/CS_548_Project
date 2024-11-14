# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:39:16 2024

@author: kevry
"""

import json
import pandas as pd
from sklearn.linear_model import LinearRegression  # Replace with your chosen model
import numpy as np

#In Here the final model must be loaded in using pickle
# Load the saved model
# with open('final_trained_model.pkl', 'rb') as f:
    # model = pickle.load(f)

model = LinearRegression() 

# Load JSON mappings
with open('actor_name_to_id.json', 'r') as f:
    actor_name_to_id = json.load(f)
with open('director_name_to_id.json', 'r') as f:
    director_name_to_id = json.load(f)
with open('genre_name_to_id.json', 'r') as f:
    genre_name_to_id = json.load(f)
with open('studio_name_to_id.json', 'r') as f:
    studio_name_to_id = json.load(f)

# Function to get encoded IDs for actors, directors, genres, and studios
def get_encoded_id(name, name_to_id_map):
    return name_to_id_map.get(name, 0)  # 0 if not found

# Prompt for up to three actors
actors = []
for i in range(3):
    actor_name = input(f"Enter actor {i+1} (or press Enter to skip): ")
    if actor_name:
        actor_id = get_encoded_id(actor_name, actor_name_to_id)
        actors.append(actor_id)
    else:
        actors.append(0)

# Prompt for the director
director_name = input("Enter the director's name: ")
director_id = get_encoded_id(director_name, director_name_to_id)

# Prompt for up to three genres
genres = []
for i in range(3):
    genre_name = input(f"Enter genre {i+1} (or press Enter to skip): ")
    if genre_name:
        genre_id = get_encoded_id(genre_name, genre_name_to_id)
        genres.append(genre_id)
    else:
        genres.append(0)

# Prompt for up to three studios
studios = []
for i in range(3):
    studio_name = input(f"Enter studio {i+1} (or press Enter to skip): ")
    if studio_name:
        studio_id = get_encoded_id(studio_name, studio_name_to_id)
        studios.append(studio_id)
    else:
        studios.append(0)

# Prepare the input for the model (assuming order of features as defined)
input_data = pd.DataFrame([actors + [director_id] + genres + studios],
                          columns=['actor_1', 'actor_2', 'actor_3', 'director', 
                                   'genre_1', 'genre_2', 'genre_3', 
                                   'studio_1', 'studio_2', 'studio_3'])

# Predict the rating
predicted_rating = model.predict(input_data)
print(f"\nEstimated movie rating: {predicted_rating[0]:.2f}")
