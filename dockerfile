# Start with the Airflow base image
FROM apache/airflow:2.5.1

# Install additional dependencies for Airflow
RUN pip install pyspark mlflow 

# Copy Airflow dags and plugins
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
