# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/yellow_tripdata_2025_01-1.parquet"
file_type = "parquet"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "yellow_tripdata_2025_01_1_parquet"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `yellow_tripdata_2025_01_1_parquet`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "yellow_tripdata_2025_01_1_parquet"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# COMMAND ----------

# Initialize Spark Session
spark = SparkSession.builder.appName("NYC Taxi Analysis").getOrCreate()

# COMMAND ----------

# Define schema to ensure proper data typing
schema = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("PULocationID", IntegerType(), True),
    StructField("DOLocationID", IntegerType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("congestion_surcharge", DoubleType(), True),
    StructField("Airport_fee", DoubleType(), True)
])

# COMMAND ----------

# Load data from CSV (replace with the actual path to your data)
# In Databricks, this could be in DBFS or mounted storage
#df = spark.read.csv("/FileStore/tables/yellow_tripdata_2025_01-1.parquet", header=True, schema=schema)


# COMMAND ----------

# Display basic info about the dataset
print("Dataset size:", df.count(), "rows")
print("Dataset columns:", len(df.columns), "columns")
df.printSchema()


# COMMAND ----------

# Data Preprocessing
# 1. Calculate trip duration in minutes
df = df.withColumn("trip_duration_minutes", 
                  (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60)


# COMMAND ----------

# 2. Extract time features
df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime")) \
       .withColumn("pickup_day", dayofweek("tpep_pickup_datetime")) \
       .withColumn("pickup_month", month("tpep_pickup_datetime"))


# COMMAND ----------

print("Original row count:", df.count())

# COMMAND ----------

# Count null values in each column - handling different data types
from pyspark.sql.functions import col, count, when

# For non-numeric columns, we only check isNull()
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
null_counts.show()

# COMMAND ----------

# 3. Clean data - remove outliers and invalid values
df_cleaned = df.filter(
    (col("trip_distance") > 0) & 
    (col("fare_amount") > 0) & 
    (col("trip_duration_minutes") > 0) & 
    (col("trip_duration_minutes") < 24*60)  # Remove trips longer than 24 hours
)

# Save cleaned dataset
df_cleaned.write.mode("overwrite").parquet("/path/to/cleaned_taxi_data")


# COMMAND ----------

df.count()

# COMMAND ----------

df.select("trip_distance", "fare_amount", "trip_duration_minutes").show(5)

# COMMAND ----------

# Count null values in each column - handling different data types
from pyspark.sql.functions import col, count, when

# For non-numeric columns, we only check isNull()
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
null_counts.show()

# COMMAND ----------

# Reload the Parquet file with more diagnostics
df_reload = spark.read.option("mergeSchema", "true").parquet("/FileStore/tables/yellow_tripdata_2025_01-1.parquet")

# Check if this has the right structure
df_reload.printSchema()
df_reload.limit(5).show()


# COMMAND ----------

print("After trip_distance filter:", df.filter(col("trip_distance") > 0).count())
print("After fare_amount filter:", df.filter(col("fare_amount") > 0).count())
print("After trip_duration_minutes > 0 filter:", df.filter(col("trip_duration_minutes") > 0).count())
print("After trip_duration_minutes < 24*60 filter:", df.filter(col("trip_duration_minutes") < 24*60).count())

# COMMAND ----------

df.select("trip_distance", "fare_amount", "trip_duration_minutes").show(5)

# COMMAND ----------

print("Row count:", df_cleaned.count())

# COMMAND ----------

# Data Analysis
# 1. Summary statistics for key attributes
summary_stats = df_cleaned.select("trip_distance", "fare_amount", "tip_amount", 
                                 "trip_duration_minutes").summary("min", "25%", "mean", "75%", "max")
summary_stats.show()


# COMMAND ----------

# 2. Average fare by distance buckets
df_cleaned.groupBy(ceil("trip_distance").alias("distance_bucket")) \
    .agg(avg("fare_amount").alias("avg_fare"), 
         count("*").alias("trip_count")) \
    .orderBy("distance_bucket") \
    .show(20)

# COMMAND ----------

# 3. Tip percentage by trip distance
df_cleaned = df_cleaned.withColumn("tip_percentage", 
                                  when(col("fare_amount") > 0, 
                                       (col("tip_amount") / col("fare_amount")) * 100).otherwise(0))

avg_tip_by_distance = df_cleaned.groupBy(ceil("trip_distance").alias("distance_bucket")) \
    .agg(avg("tip_percentage").alias("avg_tip_percentage"), 
         count("*").alias("trip_count")) \
    .orderBy("distance_bucket")

avg_tip_by_distance.show(20)

# COMMAND ----------

# 4. Busiest pickup locations
top_pickup_locations = df_cleaned.groupBy("PULocationID") \
    .count() \
    .orderBy(desc("count")) \
    .limit(10)

top_pickup_locations.show()

# COMMAND ----------

# 5. Average trip distance by hour of day
df_cleaned.groupBy("pickup_hour") \
    .agg(avg("trip_distance").alias("avg_distance"),
         count("*").alias("trip_count")) \
    .orderBy("pickup_hour") \
    .show(24)

# COMMAND ----------

# Option 1: Modify VectorAssembler to handle nulls
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"  # or "keep" depending on your needs
)

# Option 2: Remove null values from your dataset first
df_cleaned = df_cleaned.na.drop(subset=feature_cols)


# COMMAND ----------

# Model Building: Predict fare amount
# 1. Prepare features
feature_cols = ["trip_distance", "trip_duration_minutes", "passenger_count", "pickup_hour", "pickup_day"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
model_data = assembler.transform(df_cleaned)

# COMMAND ----------

# 1. Apply the modified VectorAssembler to your data
model_data = assembler.transform(df_cleaned)

# 2. Cache the data to improve performance
model_data = model_data.cache()
model_data.count()  # Force caching

# 3. Split the data into training and testing sets
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# 4. Cache the split datasets
train_data = train_data.cache()
test_data = test_data.cache()
train_data.count()  # Force caching
test_data.count()   # Force caching

# COMMAND ----------

# 5. Train the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
lr_model = lr.fit(train_data)

# COMMAND ----------

# 6. Make predictions on the test data
predictions = lr_model.transform(test_data)

# COMMAND ----------

# 7. Evaluate the model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="fare_amount", 
    predictionCol="prediction", 
    metricName="rmse")

rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

# 8. Examine model coefficients
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)

# COMMAND ----------

# 9. Show sample predictions
predictions.select("fare_amount", "prediction", "features").show(5)

# COMMAND ----------

# Create visualization for busiest times of day (display in Databricks notebook)
result = df_cleaned.groupBy("pickup_hour") \
    .count() \
    .orderBy("pickup_hour") \
    .collect()

hours = [row["pickup_hour"] for row in result]
counts = [row["count"] for row in result]

plt.figure(figsize=(12, 6))
plt.bar(hours, counts)
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.title("Taxi Trip Distribution by Hour of Day")
plt.xticks(range(0, 24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
display(plt.gcf())

# Spatial analysis - top trip routes
top_routes = df_cleaned.groupBy("PULocationID", "DOLocationID") \
    .count() \
    .orderBy(desc("count")) \
    .limit(15)

top_routes.show()

# COMMAND ----------

# 6. Save the model
lr_model.write().overwrite().save("/path/to/fare_prediction_model")

# COMMAND ----------


