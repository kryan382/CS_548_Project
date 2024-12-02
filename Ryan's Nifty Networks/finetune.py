import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler

# Load Dataset
file_path = "C:/Users/ryanj/Code/CS_548_Applied_ML/CS_548_Project/CSV/MoviesDataset_Encoded.csv"
MovieData = pd.read_csv(file_path)

# Define features (X) and target (y)
X = MovieData[['actor_1', 'director', 'genre_1', 'studio_1', 'studio_2', 
               'studio_3', 'genre_2', 'genre_3', 'actor_2', 'actor_3']].values
y = MovieData['rating'].values

# Scale X for better optimization performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to 3D for CNN input
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=17)

# Define a_20 Accuracy Metric
def a_20_accuracy(y_true, y_pred):
    """
    Custom a_20 accuracy metric: predictions within ±20% of actual values.
    """
    within_20 = tf.logical_and(
        y_pred >= 0.8 * y_true,
        y_pred <= 1.2 * y_true
    )
    return tf.reduce_mean(tf.cast(within_20, tf.float32))

# Build Model for Hyperparameter Tuning
def build_model(hp):
    input_layer = Input(shape=(X_train.shape[1], 1), name="input_layer")
    
    # CNN Layers
    x = Conv1D(
        filters=hp.Choice("filters", [16, 32, 64]),
        kernel_size=hp.Choice("kernel_size", [3, 5]),
        activation='relu',
        padding='same'
    )(input_layer)
    x = Flatten()(x)
    x = Dropout(hp.Choice("dropout_rate", [0.2, 0.3]))(x)
    
    # Dense Layers
    x = Dense(
        units=hp.Int("dense_units", 32, 128, step=32),
        activation='relu'
    )(x)
    
    # Output Layer
    output_layer = Dense(1, activation='linear', name="output_layer")(x)
    
    # Define optimizer
    optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    if optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    
    # Build and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error', a_20_accuracy]
    )
    return model

# Use Keras Tuner's RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_a_20_accuracy", direction="max"),
    max_trials=10,
    executions_per_trial=1,
    directory="hyperparam_tuning",
    project_name="cnn_tuning"
)


# Run the Hyperparameter Tuning
tuner.search(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.3,
    verbose=1
)

# Get the Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Train Final Model with Best Hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.3
)

# Evaluate Final Model on Test Data
test_loss, test_mae, test_a20 = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.2f}, Test MAE: {test_mae:.2f}, Test a_20: {test_a20:.2f}")

# Display Best Hyperparameters
print("Best hyperparameters:")
print(f"Filters: {best_hps.get('filters')}")
print(f"Kernel size: {best_hps.get('kernel_size')}")
print(f"Dense layer units: {best_hps.get('dense_units')}")
print(f"Dropout rate: {best_hps.get('dropout_rate')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")
print(f"Optimizer: {best_hps.get('optimizer')}")