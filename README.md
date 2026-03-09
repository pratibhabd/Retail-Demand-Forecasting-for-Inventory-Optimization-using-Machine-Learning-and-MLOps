# End-to-End Retail Demand Forecasting for Inventory Optimization

This project implements an **end-to-end machine learning pipeline for retail demand forecasting** using AWS cloud services. The system predicts **daily store-level product family sales** and demonstrates how machine learning models can be deployed and monitored in a production environment.

The pipeline covers the **complete lifecycle of a machine learning system**, including data ingestion, feature engineering, model training, deployment, and monitoring.

---

# Project Overview

Retail businesses rely on accurate demand forecasting to optimize inventory levels, reduce stockouts, and minimize overstocking costs. However, forecasting retail demand is challenging due to multiple influencing factors such as seasonality, promotions, holidays, store characteristics, and external economic signals.

This project builds a **cloud-native forecasting pipeline** that leverages AWS services to create a scalable and production-ready solution.

The system performs the following tasks:

- Data ingestion into a **cloud data lake**
- Exploratory data analysis to understand retail patterns
- Feature engineering including **lag and rolling statistics**
- Feature storage using **Amazon SageMaker Feature Store**
- Model training using **XGBoost**
- Model deployment using **SageMaker endpoints**
- Production monitoring using **SageMaker Model Monitor**

---

# Dataset

This project uses the **Corporación Favorita Grocery Sales Forecasting Dataset** available on Kaggle.

Dataset Source:  
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

The dataset contains historical retail sales data from a large grocery chain in Ecuador.

### Key datasets used

| Dataset | Description |
|------|------|
| train.csv | Historical daily sales data |
| stores.csv | Store metadata |
| transactions.csv | Daily store transactions |
| oil.csv | Global oil prices |
| holidays_events.csv | National and regional holidays |

The main dataset contains **over 3 million sales records**.

---

# System Architecture

The project implements a **cloud-native ML architecture** using AWS services.

Pipeline stages:

1. **Data Lake Creation**
   - Raw datasets uploaded to **Amazon S3**

2. **Serverless Analytics**
   - SQL queries executed using **Amazon Athena**

3. **Exploratory Data Analysis**
   - Trend analysis
   - seasonality analysis
   - promotion analysis
   - correlation analysis

4. **Feature Engineering**
   - time features
   - lag features
   - rolling statistics
   - store metadata integration
   - holiday indicators

5. **Feature Store**
   - features stored in **Amazon SageMaker Feature Store**
   - ensures consistency between training and inference

6. **Model Training**
   - gradient boosting model using **XGBoost**

7. **Model Deployment**
   - deployed as **SageMaker endpoint**

8. **Monitoring**
   - **SageMaker Model Monitor**
   - data drift detection
   - model quality monitoring

---

# Feature Engineering

Several predictive features were engineered to improve forecasting accuracy.

### Time Features
- day of week
- month
- year
- weekend indicator

### Lag Features

Lag features capture historical dependencies in sales.

Example:

lag_7 = sales from previous week
lag_14 = sales from two weeks prior


Mathematically:

\[
Lag_k = y_{t-k}
\]

### Rolling Statistics

Rolling averages smooth short-term fluctuations.

Example:

rolling_mean_7
rolling_mean_14
rolling_mean_30


Formula:

\[
RollingMean_w = \frac{1}{w}\sum_{i=1}^{w} y_{t-i}
\]

### External Variables

- promotions
- transactions
- oil prices
- holidays

These signals help the model capture external influences on retail demand.

---

# Model

The forecasting model is built using **XGBoost**, a gradient boosting algorithm optimized for structured datasets.

Prediction model:

\[
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
\]

where each \(f_k\) is a decision tree.

Objective function:

\[
L = \sum l(y_i, \hat{y}_i) + \sum \Omega(f_k)
\]

where

- \(l\) is the loss function
- \(\Omega\) is the regularization term

Key hyperparameters used:

| Parameter | Value |
|---|---|
| learning rate | 0.03 |
| max depth | 4 |
| num rounds | 500 |
| subsample | 0.7 |
| colsample | 0.7 |

---

# Model Evaluation

Model performance was evaluated using **Root Mean Squared Error (RMSE)**.

\[
RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}
\]

The XGBoost model significantly outperformed the naive baseline forecast.

Evaluation steps:

- baseline forecasting
- model prediction
- RMSE comparison
- actual vs predicted visualization

---

# Deployment

The trained model was deployed using **Amazon SageMaker endpoints**.

Deployment steps:

1. model artifact stored in S3
2. SageMaker model object created
3. endpoint configuration defined
4. model deployed to endpoint

Predictions are generated using the SageMaker runtime API.

Example:

```python
runtime.invoke_endpoint()

# Monitoring

Production monitoring is implemented using Amazon SageMaker Model Monitor to ensure the deployed model remains reliable over time.

Monitoring includes:

Data Capture

The endpoint captures inference requests and stores them in Amazon S3, including:

input features

model predictions

timestamps

endpoint metadata

Baseline Creation

A baseline dataset was generated using predictions from training data. Baseline statistics and constraints were computed to define expected feature distributions.

Data Drift Detection

Production feature distributions are compared with the baseline dataset to detect drift.

Population Stability Index (PSI) is used to measure distribution shifts:

### Population Stability Index (PSI)

The Population Stability Index is used to detect data drift between baseline and production data.

\[
PSI = \sum_{i=1}^{n} (P_i - Q_i) \ln\left(\frac{P_i}{Q_i}\right)
\]

where:

- \(P_i\) = proportion of observations in bin *i* for the **baseline dataset**
- \(Q_i\) = proportion of observations in bin *i* for the **production dataset**


