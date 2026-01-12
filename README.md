# BTL_ML
# Deep Learning vs Classic Machine Learning for IoT Time-Series Prediction

This repository contains the implementation of a comparative study between a Deep Learning model (CNN1D)
and a classic Machine Learning model (Linear Regression) for multivariate time-series prediction using IoT
sensor data.

The system predicts future **temperature** and **humidity** values based on historical sensor readings and
deploys the trained models in a real-time IoT pipeline using InfluxDB.

---

## Problem Description

IoT systems continuously generate sensor data such as temperature and humidity.  
Accurate prediction of these values is essential for:

- Smart home automation
- Environmental monitoring
- Predictive control and early warning systems

This project formulates the task as a **multivariate time-series regression problem**, where the goal is to
predict the next time step given a short historical window of sensor readings.

---

## Models Used

### Deep Learning Model (CNN1D)

- One-dimensional Convolutional Neural Network
- Input shape: `(n_steps, n_features)`
  - `n_steps = 3`
  - `n_features = 2` (temperature, humidity)
- Architecture:
  - Conv1D (64 filters, kernel size = 2, ReLU)
  - MaxPooling1D
  - Flatten
  - Dense (50 units, ReLU)
  - Dense output (2 values)

The CNN1D model is chosen for its ability to capture **temporal dependencies** efficiently with low
computational cost, making it suitable for real-time IoT applications.

---

### Classic Machine Learning Model (Linear Regression)

- Linear Regression implemented using `scikit-learn`
- Input data is flattened:
  - `(n_steps, n_features) → (n_steps × n_features)`
- Serves as a **baseline model** for comparison with Deep Learning

---

## Dataset

- Source: IoT sensor data (historical readings)
- Features:
  - Temperature
  - Humidity
- Preprocessing steps:
  - Remove missing values
  - Normalize data using Min-Max scaling
  - Convert time-series to supervised learning format using sliding windows

---

## Implementation Details

- Programming language: **Python**
- Libraries:
  - TensorFlow / Keras (Deep Learning)
  - scikit-learn (Linear Regression, preprocessing, evaluation)
  - InfluxDB Client (real-time data integration)
- Evaluation metric: **Mean Squared Error (MSE)**

Both models are trained offline and then used for real-time inference.

---

## Experimental Results

| Model | MSE |
|------|-----|
| Linear Regression | ~0.00171 |
| CNN1D (Deep Learning) | ~0.00165 |

Although the numerical difference in MSE is relatively small, qualitative analysis shows that the CNN1D
model produces smoother predictions and follows the ground truth more closely, especially during small
sensor fluctuations.

---

## Real-Time IoT Pipeline

The trained models are integrated into a real-time pipeline:

1. Sensor data is queried from **InfluxDB**
2. Latest values are extracted and preprocessed
3. Predictions are generated using both DL and ML models
4. Predicted values are written back to InfluxDB for monitoring and visualization

Predicted fields:
- `DL_temperature_predict`
- `DL_humidity_predict`
- `ML_temperature_predict`
- `ML_humidity_predict`
