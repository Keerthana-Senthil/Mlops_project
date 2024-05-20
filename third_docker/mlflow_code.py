import mlflow.sklearn
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
if os.getenv("DOCKER_ENV"):
    mlflow.set_tracking_uri("http:127.0.0.1:5000")
else:
    mlflow.set_tracking_uri("http://localhost:5000")
print("Entered")
import logging
import sys

# Configure logging to print to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Example log statement
logging.info("This will appear in Docker logs")
# Load data
data = pd.read_csv('./dataset/Avocado_train.csv')
data['type'] = data['type'].map({'conventional': 0, 'organic': 1})
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data.drop('Date', axis=1, inplace=True)

# Create dummy variables
dummies = pd.get_dummies(data[['Year', 'region', 'Month']], drop_first=True)
df_dummies = pd.concat([data[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type']], dummies], axis=1)
target = data['AveragePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_dummies, target, test_size=0.30, random_state=42)

# Standardize data pipeline
cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
scaler = StandardScaler()
X_train[cols_to_std] = scaler.fit_transform(X_train[cols_to_std])
X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(random_state=0),
    "Support Vector Machine": SVR(gamma='scale'),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=0)
}


# Set experiment
experiment_name = "Avocado Price Prediction2"
mlflow.set_experiment(experiment_name)

# Train and log models
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged {name} to MLflow")
