# Remaining Useful Life Prediction (LSTM)

## Overview
This project predicts the Remaining Useful Life (RUL) of turbofan engines using time-series sensor data.

## Features
- LSTM-based deep learning model
- Feature scaling and sequence generation
- Model evaluation (MAE, RMSE)
- FastAPI deployment for real-time predictions

## How to Run

### Train Model
python train_models.py

### Run API
uvicorn app:app --reload

### Test API
Go to:
http://127.0.0.1:8000/docs

## Input Format
- 30 timesteps
- Each timestep = N features

## Output
- Predicted RUL value

## Tech Stack
- Python
- TensorFlow (LSTM)
- FastAPI
- PostgreSQL
