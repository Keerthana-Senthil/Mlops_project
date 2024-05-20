from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("LoadModelAndTest") \
    .getOrCreate()

# Define paths
model_path = "./model/avocado_lin_reg/model.h5"

# Load the saved pipeline model
loaded_pipeline_model = PipelineModel.load(model_path)

# Load test dataset
test_data_path = "./dataset/Avocado_test.csv"  # Update this path to your test dataset location
df_test = spark.read.csv(test_data_path, header=True, inferSchema=True)

# Apply the loaded model to the test dataset
df_test_transformed = loaded_pipeline_model.transform(df_test)

# Show predictions
df_test_transformed.select(['features', 'prediction']).show(10, False)

# Optionally, evaluate the model using an evaluator if you have the ground truth labels
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="AveragePrice", 
    predictionCol="prediction", 
    metricName="rmse"
)

rmse = evaluator.evaluate(df_test_transformed)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")