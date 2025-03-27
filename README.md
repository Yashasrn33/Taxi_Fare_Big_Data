# Taxi_Fare_Big_Data
## NYC Taxi Fare Prediction
This project analyzes New York City taxi trip data from July 2024 - January 2025 and builds a machine learning model to predict taxi fares.

### Overview
This Databricks notebook demonstrates the following:
Data loading and preprocessing of NYC taxi trip data
Exploratory data analysis
Feature engineering
Building and evaluating a linear regression model for fare prediction
Data Source
The dataset used is yellow_taxi_tripdata (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in parquet format, containing various features of taxi trips including pickup and dropoff times, locations, fare amounts, and more.
Key Features
Data cleaning and preprocessing
Feature engineering (e.g., trip duration calculation, time-based features)
Exploratory data analysis
Linear regression model for fare prediction
Model evaluation and interpretation
Main Findings
The model achieves an R-squared value of 0.67.
Key factors influencing fare amounts include trip distance, duration, and time of day.

Usage
Ensure you have access to a Databricks environment.
Upload the yellow_tripdata_2025_01-1.parquet file to your Databricks File System (DBFS).
Import the notebook into your Databricks workspace.
Update the file_location variable with the correct path to your data file.
Run the notebook cells sequentially to reproduce the analysis and model.
Dependencies
PySpark
Matplotlib
Future Work
Incorporate more features or external data sources
Experiment with advanced machine learning models
Develop a real-time prediction system

Author
Yashas R
License
MIT License
