# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:32:43 2024

@author: kevry
"""

import numpy as np
import pandas as pd
import json
from keras.models import load_model

# Load the trained neural network model
model_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/FinalModel/movie_cnn_model_with_a20.h5"
cnn_model = load_model(model_path, compile=True)

# Load the dictionaries for encoding
actor_dict_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/JsonFiles/actor_name_to_id.json"
genre_dict_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/JsonFiles/genre_name_to_id.json"
studio_dict_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/JsonFiles/studio_name_to_id.json"
director_dict_path = "C:/Homework Assignments/MachineLearning548/Final Project/PythonCode/JsonFiles/director_name_to_id.json"

with open(actor_dict_path, 'r') as file:
    actor_dict = json.load(file)
with open(genre_dict_path, 'r') as file:
    genre_dict = json.load(file)
with open(studio_dict_path, 'r') as file:
    studio_dict = json.load(file)
with open(director_dict_path, 'r') as file:
    director_dict = json.load(file)

# Function to encode input
def encode_input(name, dictionary):
    return dictionary.get(name, 0)  # Return 0 if the name is not found

# Prompt user for inputs
print("Please enter movie details to predict the rating:")
actors = []
for i in range(1, 4):  # Prompt for up to 3 actors
    actor = input(f"Enter Actor {i} (leave blank if none): ").strip()
    if actor:
        actors.append(encode_input(actor, actor_dict))
    else:
        actors.append(0)

director = input("Enter Director: ").strip()
director_encoded = encode_input(director, director_dict)

genres = []
for i in range(1, 4):  # Prompt for up to 3 genres
    genre = input(f"Enter Genre {i} (leave blank if none): ").strip()
    if genre:
        genres.append(encode_input(genre, genre_dict))
    else:
        genres.append(0)

studios = []
for i in range(1, 4):  # Prompt for up to 3 studios
    studio = input(f"Enter Studio {i} (leave blank if none): ").strip()
    if studio:
        studios.append(encode_input(studio, studio_dict))
    else:
        studios.append(0)

# Combine all inputs into a feature array
input_features = np.array(actors + [director_encoded] + genres + studios).reshape(1, -1)

# Reshape input to match the 3D shape expected by the CNN (samples, timesteps, features)
input_features_reshaped = input_features.reshape(1, input_features.shape[1], 1)

# Predict rating using the CNN model
predicted_rating = cnn_model.predict(input_features_reshaped)

# Display the predicted rating
print(f"\nPredicted Rating: {predicted_rating[0][0]:.2f}")
