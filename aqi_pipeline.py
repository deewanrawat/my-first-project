import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === AUTOMATION SETUP ===
BASE_DIR = "AQI_Prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

AQI_CSV = os.path.join(DATA_DIR, "aqi_data.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_bilstm_model.h5")
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "predictions.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")

# === STEP 1: DATA ACQUISITION ===
def fetch_aqi_data():
    url = "https://api.openaq.org/v1/measurements"
    params = {"country": "IN", "limit": 1000, "parameter": ["pm25", "pm10", "o3", "no2", "so2", "co"]}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["results"]
        df = pd.DataFrame(data)
        df.to_csv(AQI_CSV, index=False)
        print("[✔] AQI data downloaded successfully!")
    else:
        print("[✖] Failed to fetch data.")

# === STEP 2: DATA PROCESSING ===
def preprocess_data():
    df = pd.read_csv(AQI_CSV)
    
    df = df[["location", "city", "parameter", "value", "unit", "date.utc"]]
    df["date.utc"] = pd.to_datetime(df["date.utc"])
    df.dropna(inplace=True)
    
    # Pivot pollutants to columns
    df_pivot = df.pivot_table(values="value", index=["date.utc"], columns="parameter").reset_index()
    df_pivot.fillna(method="ffill", inplace=True)

    # AQI Calculation
    def calculate_aqi(pm25, pm10, o3, no2, so2, co):
        return 0.5 * pm25 + 0.3 * pm10 + 0.1 * o3 + 0.05 * no2 + 0.03 * so2 + 0.02 * co

    df_pivot["AQI"] = calculate_aqi(df_pivot["pm25"], df_pivot["pm10"], df_pivot["o3"], df_pivot["no2"], df_pivot["so2"], df_pivot["co"])
    
    # Scaling
    scaler = MinMaxScaler()
    df_pivot[df_pivot.columns[1:]] = scaler.fit_transform(df_pivot[df_pivot.columns[1:]])
    
    df_pivot.to_csv(PROCESSED_CSV, index=False)
    print("[✔] Data preprocessing complete.")

# === STEP 3: MODEL TRAINING ===
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    df = pd.read_csv(PROCESSED_CSV)
    data_values = df["AQI"].values

    X, y = create_sequences(data_values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Bi-LSTM Model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(10, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)
    print("[✔] Model training complete and saved.")

# === STEP 4: MODEL EVALUATION ===
def evaluate_model():
    df = pd.read_csv(PROCESSED_CSV)
    data_values = df["AQI"].values

    X, y = create_sequences(data_values)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    with open(METRICS_FILE, "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR² Score: {r2}\n")
    
    pd.DataFrame({"Actual_AQI": y_test, "Predicted_AQI": y_pred.flatten()}).to_csv(PREDICTIONS_CSV, index=False)
    
    print(f"[✔] Model evaluation complete:\nMAE: {mae}, RMSE: {rmse}, R²: {r2}")

# === STEP 5: AUTOMATED EXECUTION ===
def run_pipeline():
    print("\n=== Starting AQI Prediction Pipeline ===")
    fetch_aqi_data()
    preprocess_data()
    train_model()
    evaluate_model()
    print("\n=== Pipeline Execution Completed Successfully! ===")

if __name__ == "__main__":
    run_pipeline()
