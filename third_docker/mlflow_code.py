# Import necessary libraries
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

data = pd.read_csv('./dataset/Avocado_train.csv')
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

# Define models to train
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(random_state=0),
    "Support Vector Machine": SVR(gamma='scale'),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=0)
}

# Create or set an MLflow experiment
experiment_name = "Avocado Price Prediction"
mlflow.set_experiment(experiment_name)

# Train and log each model
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged {name} to MLflow")


