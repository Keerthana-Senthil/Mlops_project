# -*- coding: utf-8 -*-
"""Avocado price prediction run with mlflow

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kTuMuzjJeC_hsQeByKr-rH763RDU4CAB
"""

# Install required packages
!pip install mlflow pyngrok

# Import necessary libraries
import mlflow
import mlflow.sklearn
from pyngrok import ngrok
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up MLflow tracking URI
os.makedirs("/content/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:///content/mlruns")

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pyngrok import ngrok

# # Authenticate ngrok
# NGROK_AUTH_TOKEN = "2ginQmZnMpu0eSTA6vJdaKoeN6i_6Bfg6hPWTYmtcnsWLYwA5"  # Replace with your actual ngrok auth token
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# # Start an ngrok tunnel to expose the MLflow UI
# ngrok_tunnel = ngrok.connect(5000)
# print("MLflow Tracking UI:", ngrok_tunnel.public_url)

# Start the MLflow UI
get_ipython().system_raw("mlflow ui --host 0.0.0.0 --port 5000 &")

# Load and preprocess the dataset
data = pd.read_csv('avocado.csv')
data['type'] = data['type'].map({'conventional': 0, 'organic': 1})
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data.drop('Date', axis=1, inplace=True)

# Creating dummy variables
dummies = pd.get_dummies(data[['Year', 'region', 'Month']], drop_first=True)
df_dummies = pd.concat([data[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type']], dummies], axis=1)
target = data['AveragePrice']

# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(df_dummies, target, test_size=0.30, random_state=42)

# Standardizing the data
cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
scaler = StandardScaler()
X_train[cols_to_std] = scaler.fit_transform(X_train[cols_to_std])
X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])

# Define hyperparameter grids for models
param_grids = {
    "Decision Tree": [
        {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 4},
        {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4}
    ],
    "Random Forest": [
        {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_estimators": 150, "max_depth": 25, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_estimators": 250, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4}
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.01, "subsample": 0.9},
        {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 250, "max_depth": 9, "learning_rate": 0.1, "subsample": 0.7},
        {"n_estimators": 300, "max_depth": 11, "learning_rate": 0.2, "subsample": 0.6}
    ]
}

# Define models to train
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(random_state=0),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=0)
}

# Function to train and log model
def train_and_log_model(model_name, model, params_grid):
    mlflow.set_experiment(model_name)
    with mlflow.start_run(run_name="Parent Run") as parent_run:
        for params in params_grid:
            with mlflow.start_run(run_name=f"{model_name} - {params}", nested=True):
                model.set_params(**params)

                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
                mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
                mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

                # Log the model
                mlflow.sklearn.log_model(model, "model")

                print(f"Logged {model_name} with params: {params} to MLflow")

# Train and log Decision Tree model
train_and_log_model("Decision Tree", models["Decision Tree"], param_grids["Decision Tree"])

# Train and log Random Forest model
train_and_log_model("Random Forest", models["Random Forest"], param_grids["Random Forest"])

# Train and log XGBoost model
train_and_log_model("XGBoost", models["XGBoost"], param_grids["XGBoost"])

# Print the MLflow tracking URL
print("MLflow Tracking UI:", ngrok_tunnel.public_url)

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pyngrok import ngrok

# # Authenticate ngrok
# NGROK_AUTH_TOKEN = "2ginQmZnMpu0eSTA6vJdaKoeN6i_6Bfg6hPWTYmtcnsWLYwA5"  # Replace with your actual ngrok auth token
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# # Start an ngrok tunnel to expose the MLflow UI
# ngrok_tunnel = ngrok.connect(5000)
# print("MLflow Tracking UI:", ngrok_tunnel.public_url)

# Start the MLflow UI
get_ipython().system_raw("mlflow ui --host 0.0.0.0 --port 5000 &")

# Load and preprocess the dataset
data = pd.read_csv('avocado.csv')
data['type'] = data['type'].map({'conventional': 0, 'organic': 1})
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data.drop('Date', axis=1, inplace=True)

# Creating dummy variables
dummies = pd.get_dummies(data[['Year', 'region', 'Month']], drop_first=True)
df_dummies = pd.concat([data[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type']], dummies], axis=1)
target = data['AveragePrice']

# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(df_dummies, target, test_size=0.30, random_state=42)

# Standardizing the data
cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
scaler = StandardScaler()
X_train[cols_to_std] = scaler.fit_transform(X_train[cols_to_std])
X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])

# Define the best performing models with their parameters
best_models = {
    "Linear Regression": (LinearRegression(), {}),
    "Decision Tree": (DecisionTreeRegressor(random_state=0), {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4}),
    "Random Forest": (RandomForestRegressor(random_state=0), {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}),
    "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=0), {"n_estimators": 250, "max_depth": 9, "learning_rate": 0.1, "subsample": 0.7})
}

# Create or set an MLflow experiment
experiment_name = "Avocado Price Prediction Main Experiment"
mlflow.set_experiment(experiment_name)

# Train and log each model with sub-runs for different hyperparameters
with mlflow.start_run(run_name="Main Experiment") as parent_run:
    for name, (model, params) in best_models.items():
        with mlflow.start_run(run_name=f"{name} - {params}", nested=True):
            model.set_params(**params)

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
            mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
            mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            print(f"Logged {name} with params: {params} to MLflow")

# Print the MLflow tracking URL
print("MLflow Tracking UI:", ngrok_tunnel.public_url)

import joblib

# Train the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=0)
xgb_model.fit(X_train, y_train)

# Save the XGBoost model
xgb_model_file = "xgboost_model.pkl"
joblib.dump(xgb_model, xgb_model_file)

print("XGBoost model saved as", xgb_model_file)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pyngrok import ngrok

# # Authenticate ngrok
# NGROK_AUTH_TOKEN = "2ginQmZnMpu0eSTA6vJdaKoeN6i_6Bfg6hPWTYmtcnsWLYwA5"  # Replace with your actual ngrok auth token
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# # Start an ngrok tunnel to expose the MLflow UI
# ngrok_tunnel = ngrok.connect(5000)
# print("MLflow Tracking UI:", ngrok_tunnel.public_url)

# Start the MLflow UI
get_ipython().system_raw("mlflow ui --host 0.0.0.0 --port 5000 &")

# Load and preprocess the dataset
data = pd.read_csv('avocado.csv')
data['type'] = data['type'].map({'conventional': 0, 'organic': 1})
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data.drop('Date', axis=1, inplace=True)

# Creating dummy variables
dummies = pd.get_dummies(data[['Year', 'region', 'Month']], drop_first=True)
df_dummies = pd.concat([data[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type']], dummies], axis=1)
target = data['AveragePrice']

# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(df_dummies, target, test_size=0.30, random_state=42)

# Standardizing the data
cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
scaler = StandardScaler()
X_train[cols_to_std] = scaler.fit_transform(X_train[cols_to_std])
X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])

# Define hyperparameter grid to search over
param_grid = [
    {'n_estimators': 50, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': 20},
    {'n_estimators': 200, 'max_depth': None},
]

# Create or set an MLflow experiment
experiment_name = "Avocado Price Prediction RF"
mlflow.set_experiment(experiment_name)

# Train and log each model with different hyperparameters
with mlflow.start_run(run_name="Random Forest Hyperparameter Tuning") as parent_run:
    for params in param_grid:
        with mlflow.start_run(run_name=f"RF - {params}", nested=True):
            model = RandomForestRegressor(**params, random_state=0)

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
            mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
            mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            print(f"Logged Random Forest with params: {params} to MLflow")

# Print the MLflow tracking URL
print("MLflow Tracking UI:", ngrok_tunnel.public_url)

