from pyspark.sql import SparkSession
from pyspark.ml.feature import SQLTransformer, StringIndexer, StandardScaler
import os
from pyspark.sql.functions import to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import keras



os.environ['PYSPARK_PYTHON'] = 'C:\\Program Files\\Python311\\python.exe'
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-22'  # Replace with your actual Java path
os.environ['SPARK_HOME'] = 'C:\\Program Files\\spark-3.5.1-bin-hadoop3'  # Replace with your actual Spark path

# Print the environment variables to debug
print("JAVA_HOME:", os.environ.get('JAVA_HOME'))
print("SPARK_HOME:", os.environ.get('SPARK_HOME'))
print("PYSPARK_PYTHON:", os.environ.get('PYSPARK_PYTHON'))
# Create SparkSession
spark = SparkSession.builder \
    .master("local[*]")\
        .appName("SimpleSQLTransformerExample") \
            .getOrCreate()

# Sample DataFrame
df_avocado_train = spark.read.csv(".\dataset\Avocado_train.csv", header=True, inferSchema=True)
df_avocado_test = spark.read.csv(".\dataset\Avocado_test.csv", header=True, inferSchema=True)
COLUMNS = ['AveragePrice', 'type']
COLUMNS = [f"`{col}`" for col in COLUMNS]

LOG_COLUMNS =  ['4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags']
LOG_COLUMNS = [f"LOG(`{col}`+1) AS `LOG {col}`" for col in LOG_COLUMNS]

# Define SQL transformation
sql_trans = SQLTransformer(
    statement=f"""
    SELECT
        {', '.join(COLUMNS)},
        {', '.join(LOG_COLUMNS)},
        YEAR(to_date(__THIS__.Date, 'dd-MM-yyyy')) AS year,
        MONTH(to_date(__THIS__.Date, 'dd-MM-yyyy')) AS month
    FROM __THIS__
    """
)

# Apply transformation

transformed_df = sql_trans.transform(df_avocado_train)

# Show the transformed DataFrame
transformed_df.show()

"""**MinMaxScaler**

The MinMaxScaler is a trasformer of type estimator. This kind of object needs to be fitted to the data before being used to transform it. So, the conlumns need to be converted to a vector first.

The MinMaxScaler is a transformer that scales each feature individually to a given range (often [0, 1]).
"""

month_vec_ass = VectorAssembler(inputCols=['month'], outputCol='month_vec')
month_vec_ass.transform(sql_trans.transform(df_avocado_train))

df_avocado_month_ass = month_vec_ass.transform(sql_trans.transform(df_avocado_train))

month_scaler = MinMaxScaler(inputCol='month_vec', outputCol='month_scaled')
month_scaler = month_scaler.fit(df_avocado_month_ass)

month_scaler.transform(df_avocado_month_ass).select( ['month', 'month_vec', 'month_scaled'] ).show(10)

"""**StringIndexer**

Assigns a numerical value to each category in a column. As the column "type" only has two categories, it will be transformed into a column with only two values: 0 and 1, which is equivalent to applying a OneHotEncoder.
"""

str_indexer = StringIndexer(inputCol="type", outputCol="type_index")

str_indexer = str_indexer.fit(df_avocado_train)

str_indexer.transform(df_avocado_train).select( ["type", "type_index"] ).show(4)

"""**VectorAssembler**

This transformer combines a given list of columns into a single vector column. The vector column is named "features" by default, and it will be used later to train the model.

"""

# Apply transformations
## SQL transformer
df_avocado_train_transformed = sql_trans.transform(df_avocado_train)

## String indexer
df_avocado_train_transformed = str_indexer.transform(df_avocado_train_transformed)

## Month scaler (vector assembler + minmax scaler)
df_avocado_train_transformed = month_vec_ass.transform(df_avocado_train_transformed)
df_avocado_train_transformed = month_scaler.transform(df_avocado_train_transformed)

# Join all features into a single vector
numerical_vec_ass = VectorAssembler(
    inputCols=['year', 'month_scaled', 'LOG 4225', 'LOG 4770', 'LOG Small Bags', 'LOG Large Bags', 'LOG XLarge Bags'],
    outputCol='features_num'
)
df_avocado_train_transformed = numerical_vec_ass.transform(df_avocado_train_transformed)

# Join all categorical features into a single vector
categorical_vec_ass = VectorAssembler(
    inputCols=['type_index'],
    outputCol='features_cat'
)
df_avocado_train_transformed = categorical_vec_ass.transform(df_avocado_train_transformed)


# See the result
df_avocado_train_transformed.select(['features_cat', 'features_num', 'AveragePrice']).show(4, False)

"""**StandardScaler**

It is a transformer that standardizes features by removing the mean and scaling to unit variance. It is very similar to the MinMaxScaler, but it does not bound the values to a specific range. It is also a type of estimator, so it needs to be fitted to the data before being used to transform it.
"""

std_scaler = StandardScaler(
    inputCol="features_num",
    outputCol="features_scaled",
    withStd=True,
    withMean=True
)

std_scaler = std_scaler.fit(df_avocado_train_transformed)
std_scaler.transform(df_avocado_train_transformed).select(['features_scaled']).show(5, False)
"""**Join everything together**"""

# Create a pipeline
prepro_pipe = Pipeline(stages=[
    sql_trans,
    str_indexer,
    month_vec_ass,
    month_scaler,
    numerical_vec_ass,
    categorical_vec_ass,
    std_scaler,

    # Join all features into a single vector
    VectorAssembler(
        inputCols=['features_scaled', 'features_cat'],
        outputCol='features'
    ),
])


# Fit the pipeline
pipeline_model = prepro_pipe.fit(df_avocado_train)

# Transform the data
df_avocado_train_transformed = pipeline_model.transform(df_avocado_train)

# See the result
df_avocado_train_transformed.select(['features', 'AveragePrice']).show(4, False)

"""## Define the model"""

from pyspark.ml.regression import LinearRegression

from pyspark.ml.evaluation import RegressionEvaluator

"""### Training a Linear Regression Model

**Training a linear regression model**
"""

# Create a linear regression model
lin_reg = LinearRegression(
    featuresCol='features',
    labelCol='AveragePrice',
    predictionCol='prediction',
    maxIter=500,
    regParam=0.3,       # Regularization
    elasticNetParam=0.8 # Regularization mixing parameter. 1 for L1, 0 for L2.
)

# Explain parameter
print(lin_reg.explainParams())

# Fit the model
lin_reg_model = lin_reg.fit(df_avocado_train_transformed)

# See the output
df_avocado_train_pred = lin_reg_model.transform(df_avocado_train_transformed)
df_avocado_train_pred.select(['features', 'AveragePrice', 'prediction']).show(4, False)

"""**Evaluating the model**

The evaluator is a transformer that takes a dataset and returns a single value. In this case, it will return the RMSE value.
"""

reg_eval = RegressionEvaluator(
    labelCol='AveragePrice',
    predictionCol='prediction',
    metricName='rmse' # Root mean squared error
)
"""### Creating the Full Pipeline - Hyperparameter Tuning with Cross Validation"""
"""Define the pipeline"""

ml_pipeline = Pipeline(stages=[
    prepro_pipe, # Preprocessing pipeline
    lin_reg     # Linear regression model
])

"""Define the parameter grid. To build a parameter grid, we use the ParamGridBuilder class. We use the original regression object to reference the parameters we want to tune."""

param_grid = ParamGridBuilder() \
    .addGrid(lin_reg.regParam, [0.05, 0.1, 0.2, 0.5]) \
    .addGrid(lin_reg.elasticNetParam, [0.01, 0.5, 1.0]) \
    .build()

reg_eval = RegressionEvaluator(
    labelCol='AveragePrice',
    predictionCol='prediction',
    metricName='rmse' # Root mean squared error
)

"""Join everything together using a CrossValidator object."""

crossval_ml = CrossValidator(
    estimator=ml_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=reg_eval,
    numFolds=10
)

"""Train the model"""


crossval_ml_model = crossval_ml.fit(df_avocado_train)

best_model = crossval_ml_model.bestModel
best_score = crossval_ml_model.avgMetrics[0]

print("Best model: ", best_model)
print("Best score: ", best_score)

best_lin_reg_params = best_model.stages[-1].extractParamMap()
print("Best score (RMSE):", best_score, end="\n\n")
for parameter, value in best_lin_reg_params.items():
    print(f"{str(parameter):50s}, {value}")
# Before starting a new run, end any active run
mlflow.set_experiment("Avocado mlflow")
if mlflow.active_run():
    mlflow.end_run()
# Log parameters


# Train the model
with mlflow.start_run(run_name="log_metrics"):
    # Fit cross-validated model
    mlflow.log_param("maxIter", lin_reg.getMaxIter())
    mlflow.log_param("regParam", lin_reg.getRegParam())
    mlflow.log_param("elasticNetParam", lin_reg.getElasticNetParam())
    mlflow.log_param("numFolds", crossval_ml.getNumFolds())
    crossval_ml_model = crossval_ml.fit(df_avocado_train)
    
    # Evaluate best model
    df_avocado_test_pred = crossval_ml_model.transform(df_avocado_test)
    evaluator = RegressionEvaluator(labelCol='AveragePrice', predictionCol='prediction', metricName='rmse')
    rmse = evaluator.evaluate(df_avocado_test_pred)
    
    # Log metrics
    mlflow.log_metric("RMSE", rmse)

# End MLFlow run
mlflow.end_run()
"""### Evaluate the best model on the test set"""

df_avocado_test_pred = best_model.transform(df_avocado_test)

# show scores
print(reg_eval.evaluate(df_avocado_test_pred))

spark = SparkSession.builder.appName("FindSparkUsername").getOrCreate()

# Get the Spark username
spark_username = spark.sparkContext.sparkUser()
print(f"Spark is running under the username: {spark_username}")

# Assuming your directory is "model" and the model name is "avocado_lin_reg"
model_dir = "./model/avocado_lin_reg"
pipeline_model_dir = "./model/avocado_pipeline"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(pipeline_model_dir, exist_ok=True)

# Ensure the paths are correctly handled
model_path = os.path.join(model_dir, "model.h5")
pipeline_model_path = os.path.join(pipeline_model_dir, "pipeline")

# Save the best linear regression model
best_model.save(model_path)

# Save the pipeline model
#pipeline_model.write().overwrite().save(pipeline_model_path)