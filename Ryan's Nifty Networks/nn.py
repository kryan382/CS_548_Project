# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.metrics import MeanAbsoluteError
import tensorflow as tf

#%% Load Dataset
file_path = "C:/Users/ryanj/Code/CS_548_Applied_ML/CS_548_Project/CSV/MoviesDataset_Encoded.csv"
MovieData = pd.read_csv(file_path)

# Define X and y
X = MovieData[['actor_1', 'director', 'genre_1', 'studio_1', 'studio_2', 
               'studio_3', 'genre_2', 'genre_3', 'actor_2', 'actor_3']].values
y = MovieData['rating'].values

# Reshape X to 3D for CNN input: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

#%% Define `a_20` Custom Metric
def a_20_accuracy(y_true, y_pred):
    """
    Custom accuracy metric: calculates the percentage of predictions
    within ±20% of the actual values.
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    
    # Absolute error calculation
    abs_error = tf.abs(y_true - y_pred)
    
    # Condition: within 20% of the actual value
    within_20 = tf.logical_and(
        y_pred >= 0.8 * y_true,
        y_pred <= 1.2 * y_true
    )
    
    # Calculate the accuracy
    return tf.reduce_mean(tf.cast(within_20, tf.float32))

"""
Best hyperparameters:
Filters: 32
Kernel size: 5
Dense layer units: 96
Dropout rate: 0.2
Learning rate: 0.001
Optimizer: rmsprop

"""

#%% Build the CNN Model
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)), 
    Dropout(0.2),  
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    Flatten(),
    Dense(96, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=RMSprop(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error', a_20_accuracy])

#%% Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

#%% Evaluate the Model
train_loss, train_mae, train_a20 = model.evaluate(X_train, y_train)
test_loss, test_mae, test_a20 = model.evaluate(X_test, y_test)

print(f"Training Loss: {train_loss:.4f}, Training MAE: {train_mae:.4f}, Training a_20: {train_a20:.4f}")
print(f"Testing Loss: {test_loss:.4f}, Testing MAE: {test_mae:.4f}, Testing a_20: {test_a20:.4f}")

#%% Save Model and Metrics
model.save("movie_cnn_model_with_a20.h5")

metrics_df = pd.DataFrame({
    "epoch": range(1, len(history.history['loss']) + 1),
    "train_loss": history.history['loss'],
    "val_loss": history.history['val_loss'],
    "train_mae": history.history['mean_absolute_error'],
    "val_mae": history.history['val_mean_absolute_error'],
    "train_a20": history.history['a_20_accuracy'],
    "val_a20": history.history['val_a_20_accuracy']
})

metrics_df.to_csv("cnn_training_metrics_with_a20.csv", index=False)
print("Model and metrics saved!")
