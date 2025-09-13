Predictive Maintenance for Turbofan Engines

Project Overview

This repository contains a comprehensive machine learning pipeline for predicting the Remaining Useful Life (RUL) of turbofan engines. The goal of this project is to build and evaluate various regression models to accurately forecast when an engine will fail based on its operational data.

The project uses the publicly available NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, a widely-used benchmark in predictive maintenance. The pipeline includes:

•	Data Preprocessing and Feature Engineering: Handling raw time-series data and creating advanced features.

•	Model Comparison: Training and evaluating several classical machine learning models, including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression (SVR), and K-Nearest Neighbors (K-NN).

•	Hyperparameter Tuning: Optimizing the performance of the best-performing classical model (Random Forest) using GridSearchCV.

•	Deep Learning Implementation: Building and training a Long Short-Term Memory (LSTM) network to leverage the sequential nature of the data.

•	Final Evaluation: Comparing the final models on a held-out test set to determine the best-performing solution.


Dataset

The C-MAPSS dataset simulates the operational data of turbofan engines under various flight conditions. Each engine starts with a different degree of initial wear and tear. The data includes:

•	unit_number: A unique identifier for each engine.

•	time_in_cycles: The operational time of the engine.

•	operational_settings: Three settings that vary during a flight.

•	sensor_readings: 21 sensor values that record engine performance.


Methodology

1. Data Cleaning and RUL Calculation

The raw data from all four FD scenarios (FD001-FD004) were combined and preprocessed. The Remaining Useful Life (RUL) for each engine was calculated based on the maximum number of cycles for each engine.

2. Feature Engineering

To capture the engine's degradation trend, several rolling mean and rolling standard deviation features were engineered from the raw sensor readings. This process helps to smooth out noise and highlight the long-

term patterns in the data.

3. Model Training and Comparison

Multiple models were trained to establish a baseline performance. The Random Forest model emerged as the top performer on the validation set, demonstrating its effectiveness in capturing complex, non-linear relationships in the data.

4. Hyperparameter Tuning

GridSearchCV was used to find the optimal parameters for the Random Forest Regressor. This process involved systematically testing different combinations of parameters to maximize the model's performance on the validation set. The final optimized model showed improved performance on the test set.

5. Deep Learning with LSTM

An LSTM model was implemented to handle the time-series nature of the data. The model was trained to predict the RUL based on sequences of sensor readings.


Results

The models were evaluated using Root Mean Squared Error (RMSE) and R-squared (R2) score.

Tuned RandomForest

Test Set RMSE :-38.20	

Test set R-squared :-0.441

LSTM	              

Test Set RMSE :-37.46	 

Test set R-squared :-0.462

The LSTM model achieved the best performance on the final test set, slightly outperforming the tuned Random Forest model.


How to Run the Code

1.	Clone the repository:

2.	git clone [https://github.com/ABHISHEKVARSANI001/Predictive-Maintenance]

3.	cd your-repository-name

4.	Install dependencies:

5.	pip install -r requirements.txt

6.	Run the script:

7.	python adv_m.py


File Structure

•	adv_m.py: The main Python script containing the full predictive maintenance pipeline.

•	requirements.txt: A list of all required Python libraries.

•	best_random_forest_model.pkl: A serialized file of the best-performing Random Forest model.

•	scaler.pkl: The trained data scaler object.

•	README.md: This file.
