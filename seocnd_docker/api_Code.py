import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from typing import List
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from io import StringIO
import joblib
from prometheus_client import Histogram,generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from starlette.responses import Response


def load_model_function(path: str) -> Sequential:
    """Function to load the saved model."""
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_values(model: Sequential, data: pd.DataFrame) -> List[float]:
    """Predict values using the loaded model."""
    try:
        predictions = model.predict(data)
        return predictions.flatten().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting values: {str(e)}")

print("Start")
# Run the FastAPI app with Uvicorn
app = FastAPI()

# Load the model
model_path = "./model/xgboost_model.pkl"  # Update with your model path
print(model_path)
# Check if the file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)



# Create gauges to monitor API metrics
api_usage_created = Gauge('api_usage_created', 'Time when API usage occurred', ['client_ip'])
apt_upgrades_pending = Gauge('apt_upgrades_pending', 'Number of pending APT package upgrades')
input_length_gauge = Gauge('input_length', 'Length of input text')
api_time_gauge = Gauge('api_time', 'Total time taken by the API')
tl_time_gauge = Gauge('tl_time', 'Effective processing time (microseconds per character)')
http_request_size_bytes_created = Histogram('http_request_size_bytes_created',
                                            'HTTP request size in bytes',
                                            buckets=(0, 100, 1000, 5000, 10000, float('inf')))
net_conntrack_dialer_conn_attempted_total = Counter('net_conntrack_dialer_conn_attempted_total', 'Total connection attempts made by dialer', ['dialer_name'])

# Endpoint for predicting values and calculating RMSE
@app.post("/predict")
async def predict_and_evaluate(file: UploadFile = File(...), request: Request = Request):
    """Endpoint to predict values from an uploaded CSV file and calculate RMSE."""
    try:
        start_time = time.time()  # Start time

        # Get client IP address
        client_ip = request.client.host

        # Read and parse CSV file
        contents = await file.read()
        contents = contents.decode('utf-8')
        csv_data = StringIO(contents)
        df = pd.read_csv(csv_data)
        df['type'] = df['type'].map({'conventional': 0, 'organic': 1})
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df.drop('Date', axis=1, inplace=True)
        dummies = pd.get_dummies(df[['Year', 'region', 'Month']], drop_first=True)
        X_test = pd.concat([df[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type']], dummies], axis=1)
        y_test = df['AveragePrice']

        # Use the loaded model to make predictions on the test data
        loaded_predictions = model.predict(X_test)

        # Calculate evaluation metrics
        loaded_mae = mean_absolute_error(y_test, loaded_predictions)
        loaded_mse = mean_squared_error(y_test, loaded_predictions)
        loaded_r2 = r2_score(y_test, loaded_predictions)

        # Log metrics with MLflow or print them
        print("Loaded XGBoost Model Evaluation Metrics:")
        print("Mean Absolute Error:", loaded_mae)
        print("Mean Squared Error:", loaded_mse)
        print("R-squared Score:",loaded_r2)

        end_time = time.time()  # Record end time
        api_time = end_time - start_time  # Calculate total API time
        input_length = len(df)  # Get length of input text

        # Calculate T/L time (microseconds per character)
        tl_time = (api_time * 1e6) / input_length

        # Set gauge values
        input_length_gauge.set(input_length)
        api_time_gauge.set(api_time)
        tl_time_gauge.set(tl_time)
        # Logging client IP details
        print(f"Client IP: {client_ip}")

        return {"mae": loaded_mae, "mse": loaded_mse}  # Returning predictions and RMSE
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Handler for root path
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Model Prediction API"}  # Update message accordingly

# Endpoint for serving Prometheus metrics
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
