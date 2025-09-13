# ==============================================================================
# Project: Advanced Predictive Maintenance for Turbofan Engines
# Author: Abhishek Varsani
#
# Description:
# This script implements a comprehensive machine learning pipeline for predicting
# the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.
# The project includes:
#   - Data loading for all four FD scenarios (FD001-FD004).
#   - Advanced feature engineering using rolling statistics.
#   - Training and comparison of multiple models (Random Forest, SVR, etc.).
#   - Hyperparameter tuning to optimize model performance.
#   - Implementation and evaluation of a Deep Learning LSTM model.
#   - Final evaluation on the official test sets.
#
# Dependencies:
#   - pandas, numpy, scikit-learn, matplotlib, seaborn
#   - joblib
#   - tensorflow
#
# Usage:
# 1. Ensure the required data files (e.g., train_FD001.txt, RUL_FD001.txt) are
#    in the same directory as this script.
# 2. Install the dependencies using the requirements.txt file:
#    pip install -r requirements.txt
# 3. Run the script from the terminal:
#    python your_script_name.py
#
# Notes:
#   - This script generates and displays multiple plots.
#   - The best-performing model and scaler are saved as 'best_random_forest_model.pkl'
#     and 'scaler.pkl' for later use.
# ==============================================================================

# --- Step 0: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # For saving and loading the model
import tensorflow as tf # For the deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten

# --- Step 1: DATA LOADING AND CLEANING ---
print("--- Step 1: Loading and Cleaning Data ---")

col_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
            [f'sensor_measure_{i}' for i in range(1, 22)]

train_dfs = []
test_dfs = []
truth_dfs = []

# Loop through each of the four scenarios (FD001 to FD004)
for i in range(1, 5):
    train_file = f'train_FD00{i}.txt'
    test_file = f'test_FD00{i}.txt'
    truth_file = f'RUL_FD00{i}.txt'

    try:
        # Load training data
        train_df = pd.read_csv(train_file, sep=' ', header=None)
        train_df.drop(columns=[26, 27], inplace=True)
        train_df.columns = col_names
        train_df['scenario'] = f'FD00{i}'
        train_dfs.append(train_df)

        # Load test data
        test_df = pd.read_csv(test_file, sep=' ', header=None)
        test_df.drop(columns=[26, 27], inplace=True)
        test_df.columns = col_names
        test_df['scenario'] = f'FD00{i}'
        test_dfs.append(test_df)

        # Load truth data for the test set
        truth_df = pd.read_csv(truth_file, sep=' ', header=None)
        truth_df.drop(columns=[1], inplace=True)
        truth_df.columns = ['RUL']
        truth_dfs.append(truth_df)

    except FileNotFoundError:
        print(f"Warning: File not found: {train_file}, {test_file}, or {truth_file}. Skipping this scenario.")
        continue

all_train_df = pd.concat(train_dfs, ignore_index=True)
all_test_df = pd.concat(test_dfs, ignore_index=True)
all_truth_df = pd.concat(truth_dfs, ignore_index=True)

print("Combined training data shape:", all_train_df.shape)
print("Combined test data shape:", all_test_df.shape)
print("Combined truth data shape:", all_truth_df.shape)
print("-" * 50)


# --- Step 2: FEATURE ENGINEERING ---
print("--- Step 2: Feature Engineering (RUL Calculation) ---")

# Calculate RUL for the training set
max_cycles_per_unit = all_train_df.groupby(['unit_number', 'scenario'])['time_in_cycles'].max().reset_index()
max_cycles_per_unit.rename(columns={'time_in_cycles': 'max_cycles'}, inplace=True)
all_train_df = all_train_df.merge(max_cycles_per_unit, on=['unit_number', 'scenario'], how='left')
all_train_df['RUL'] = all_train_df['max_cycles'] - all_train_df['time_in_cycles']

print("RUL for training set created.")
print("-" * 50)


# --- Step 3: ADVANCED FEATURE ENGINEERING ---
print("--- Step 3: Advanced Feature Engineering (Rolling Stats) ---")

# We'll use a sliding window for feature engineering
window_size = 20

# List of sensor and operating setting columns to apply rolling features to
cols_to_roll = [f'sensor_measure_{i}' for i in range(1, 22)]
# You can add operational settings if they show a strong trend
# cols_to_roll.extend(['op_setting_1', 'op_setting_2', 'op_setting_3'])

# Create new rolling features for both training and test data
for df in [all_train_df, all_test_df]:
    for col in cols_to_roll:
        df[f'{col}_rolling_mean_{window_size}'] = df.groupby(['unit_number', 'scenario'])[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df[f'{col}_rolling_std_{window_size}'] = df.groupby(['unit_number', 'scenario'])[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        ).fillna(0) # Fill NaNs for the first cycle with 0

print("Rolling mean and standard deviation features created for both training and test data.")
print("-" * 50)

# --- Step 4: DATA PREPARATION FOR MODELING ---
print("--- Step 4: Data Preparation ---")

# Define the features and target
feature_cols = [col for col in all_train_df.columns if '_rolling_mean' in col or '_rolling_std' in col]
X = all_train_df[feature_cols]
y = all_train_df['RUL']

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for models sensitive to scale (SVR, K-NN, LSTMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Data split and scaled.")
print("-" * 50)


# --- Step 5: MODEL TRAINING AND HYPERPARAMETER TUNING ---
print("--- Step 5: Model Training and Hyperparameter Tuning ---")

# A. Train traditional models
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'K-NN': KNeighborsRegressor()
}

predictions = {}
for name, model in models.items():
    print(f"Training standard {name}...")
    if name in ['SVR', 'K-NN']:
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_val)
    print(f"Standard {name} predictions complete.")

# B. Hyperparameter Tuning for RandomForestRegressor
print("\n--- Hyperparameter Tuning for RandomForestRegressor ---")
# Reduced parameter grid to speed up execution
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

print("Starting Grid Search... this may take a while.")
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print("\nBest RandomForest Parameters:", grid_search.best_params_)
print("Best RandomForest Validation R-squared:", grid_search.best_score_)
predictions['Tuned RandomForest'] = best_rf_model.predict(X_val)


# C. Deep Learning Model (LSTM)
print("\n--- Training LSTM Model ---")

# Reshape data for LSTM [samples, timesteps, features]
# We'll use a single timestep as our current features are not sequential per row
# A more advanced approach would use a sliding window to create timesteps
X_train_lstm = np.array(X_train_scaled).reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_lstm = np.array(X_val_scaled).reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])

# Build the LSTM model
lstm_model = Sequential([
    LSTM(100, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1) # Output layer for regression
])
lstm_model.compile(optimizer='adam', loss='mse')

print("Starting LSTM training... this may take a few minutes.")
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_val_lstm, y_val), verbose=1)

predictions['LSTM'] = lstm_model.predict(X_val_lstm).flatten()
print("LSTM predictions complete.")
print("-" * 50)


# --- Step 6: FINAL EVALUATION ON THE OFFICIAL TEST SET ---
print("--- Step 6: Final Evaluation on the Official Test Set ---")

# To correctly evaluate, we only need predictions for the final cycle of each engine.
# We'll find the last cycle for each engine in the test set.
last_cycles = all_test_df.groupby(['unit_number', 'scenario'])['time_in_cycles'].max().reset_index()
last_cycles_df = all_test_df.merge(last_cycles, on=['unit_number', 'scenario', 'time_in_cycles'], how='inner')
X_test_final_cycles = last_cycles_df[feature_cols]

# A. Evaluate the best model (Tuned RandomForest)
test_predictions_rf = best_rf_model.predict(X_test_final_cycles)
test_rmse_rf = np.sqrt(mean_squared_error(all_truth_df['RUL'], test_predictions_rf))
test_r2_rf = r2_score(all_truth_df['RUL'], test_predictions_rf)
print(f"Tuned RandomForest Test Set RMSE: {test_rmse_rf:.4f}")
print(f"Tuned RandomForest Test Set R-squared: {test_r2_rf:.4f}")

# B. Evaluate the LSTM model
X_test_final_cycles_scaled = scaler.transform(X_test_final_cycles)
X_test_final_cycles_lstm = np.array(X_test_final_cycles_scaled).reshape(X_test_final_cycles_scaled.shape[0], 1, X_test_final_cycles_scaled.shape[1])
test_predictions_lstm = lstm_model.predict(X_test_final_cycles_lstm).flatten()
test_rmse_lstm = np.sqrt(mean_squared_error(all_truth_df['RUL'], test_predictions_lstm))
test_r2_lstm = r2_score(all_truth_df['RUL'], test_predictions_lstm)
print(f"LSTM Test Set RMSE: {test_rmse_lstm:.4f}")
print(f"LSTM Test Set R-squared: {test_r2_lstm:.4f}")
print("-" * 50)


# --- Step 7: MODEL SAVING AND FINAL SUMMARY ---
print("--- Step 7: Saving Best Model and Final Summary ---")

# Save the best model (Tuned RandomForest)
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
print("Best model (Tuned RandomForest) saved to 'best_random_forest_model.pkl'")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to 'scaler.pkl'")

all_scores = {}
for name, preds in predictions.items():
    all_scores[name] = {
        'MSE': mean_squared_error(y_val, preds),
        'RMSE': np.sqrt(mean_squared_error(y_val, preds)),
        'MAE': mean_absolute_error(y_val, preds),
        'R-squared': r2_score(y_val, preds)
    }
all_scores['Tuned RandomForest Test'] = {'RMSE': test_rmse_rf, 'R-squared': test_r2_rf, 'MAE': np.nan, 'MSE': np.nan}
all_scores['LSTM Test'] = {'RMSE': test_rmse_lstm, 'R-squared': test_r2_lstm, 'MAE': np.nan, 'MSE': np.nan}

final_summary_df = pd.DataFrame(all_scores).T
print("\nFinal Model Performance Summary:")
print(final_summary_df)
print("-" * 50)


# --- Step 8: VISUALIZATION ---
print("--- Step 8: Visualizing Results ---")
sns.set_style("whitegrid")

# Plot 1: RUL over time for a single engine
plt.figure(figsize=(10, 6))
engine_data = all_train_df[(all_train_df['unit_number'] == 1) & (all_train_df['scenario'] == 'FD001')]
sns.lineplot(data=engine_data, x='time_in_cycles', y='sensor_measure_11_rolling_mean_20')
plt.title('Sensor 11 Rolling Mean vs. Time for Engine FD001-001')
plt.xlabel('Time in Cycles')
plt.ylabel('Sensor 11 Rolling Mean Reading')
plt.show()

# Plot 2: All Model Performance Comparison (RMSE)
rmse_scores_df = final_summary_df.sort_values(by='RMSE', ascending=False)
plt.figure(figsize=(14, 8))
sns.barplot(x=rmse_scores_df.index, y='RMSE', data=rmse_scores_df, palette='viridis', hue=rmse_scores_df.index, legend=False)
plt.title('RMSE Comparison of All Models (Validation & Test Sets)')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.show()

# Plot 3: Predicted vs. Actual RUL on the Validation Set
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 12))
fig.suptitle('Predicted vs. Actual RUL on Validation Set', fontsize=20, y=1.02)
axes = axes.flatten()
model_list = ['LinearRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'SVR', 'K-NN', 'Tuned RandomForest', 'LSTM']
for i, name in enumerate(model_list):
    ax = axes[i]
    if name == 'Tuned RandomForest':
        preds = predictions['Tuned RandomForest']
    elif name == 'LSTM':
        preds = predictions['LSTM']
    else:
        preds = predictions[name]
        
    ax.scatter(y_val, preds, alpha=0.5, color=sns.color_palette("viridis", len(model_list))[i])
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax.set_title(f'{name}', fontsize=14)
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Predicted vs. Actual on the Official Test Set (Best Model)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
fig.suptitle('Predicted vs. Actual RUL on Official Test Set', fontsize=20, y=1.02)

# Tuned RandomForest plot
axes[0].scatter(all_truth_df['RUL'], test_predictions_rf, alpha=0.5, color='green')
axes[0].plot([all_truth_df['RUL'].min(), all_truth_df['RUL'].max()], [all_truth_df['RUL'].min(), all_truth_df['RUL'].max()], 'r--', lw=2)
axes[0].set_title(f'Tuned RandomForest (RMSE: {test_rmse_rf:.2f})')
axes[0].set_xlabel('Actual RUL')
axes[0].set_ylabel('Predicted RUL')
axes[0].grid(True)

# LSTM plot
axes[1].scatter(all_truth_df['RUL'], test_predictions_lstm, alpha=0.5, color='purple')
axes[1].plot([all_truth_df['RUL'].min(), all_truth_df['RUL'].max()], [all_truth_df['RUL'].min(), all_truth_df['RUL'].max()], 'r--', lw=2)
axes[1].set_title(f'LSTM (RMSE: {test_rmse_lstm:.2f})')
axes[1].set_xlabel('Actual RUL')
axes[1].set_ylabel('Predicted RUL')
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("--- Project Complete! ---")
