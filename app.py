import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
import joblib

# -------------------------
# Config
# -------------------------

SEQ_LEN = 30

# -------------------------
# Load model + scaler
# -------------------------

model = load_model("lstm_rul_model.keras")
scaler = joblib.load("scaler.pkl")

# -------------------------
# FastAPI app
# -------------------------

app = FastAPI()

# -------------------------
# Input schema
# -------------------------

class SensorInput(BaseModel):
    data: list  # expected: 30 x num_features


# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict_rul")
def predict_rul(input_data: SensorInput):

    # Convert input to numpy
    arr = np.array(input_data.data)

    # -------------------------
    # Validation
    # -------------------------

    if arr.ndim != 2:
        return {"error": "Input must be a 2D list (timesteps x features)"}

    if arr.shape[0] != SEQ_LEN:
        return {"error": f"Input must have exactly {SEQ_LEN} timesteps"}

    num_features = scaler.n_features_in_

    if arr.shape[1] != num_features:
        return {
            "error": f"Each timestep must have {num_features} features"
        }

    # -------------------------
    # Scaling
    # -------------------------

    # Flatten → scale → reshape back
    arr = arr.reshape(-1, num_features)
    arr = scaler.transform(arr)
    arr = arr.reshape(1, SEQ_LEN, num_features)

    # -------------------------
    # Prediction
    # -------------------------

    pred = model.predict(arr, verbose=0)

    return {"predicted_rul": float(pred[0][0])}