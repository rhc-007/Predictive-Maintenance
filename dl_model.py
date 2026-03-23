import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# -------------------------
# Config
# -------------------------

SEQ_LEN = 30

DB_URL = "postgresql://postgres:12345678@localhost:5432/pred_maintainance"


# -------------------------
# Load data
# -------------------------

engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM turbofan_features", engine)

print("Columns in dataset:")
print(df.columns)

df = df.dropna()

# -------------------------
# Features and target
# -------------------------

feature_cols = df.drop(columns=["rul", "engine_id", "data_version"]).columns

X = df[feature_cols]
y = df["rul"]

# -------------------------
# Scale features
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

df_scaled = pd.concat(
    [
        df[["engine_id"]].reset_index(drop=True),
        X_scaled.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

# -------------------------
# Engine-level train/test split
# -------------------------

engine_ids = df_scaled["engine_id"].unique()

train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_df = df_scaled[df_scaled["engine_id"].isin(train_engines)]
test_df = df_scaled[df_scaled["engine_id"].isin(test_engines)]

# -------------------------
# Sequence creation
# -------------------------

def create_sequences(data, seq_len):

    X_seq = []
    y_seq = []

    engines = data["engine_id"].unique()

    for engine in engines:

        engine_data = data[data["engine_id"] == engine]

        features = engine_data[feature_cols].values
        rul = engine_data["rul"].values

        for i in range(len(engine_data) - seq_len):

            X_seq.append(features[i:i+seq_len])
            y_seq.append(rul[i+seq_len])

    return np.array(X_seq), np.array(y_seq)


X_train, y_train = create_sequences(train_df, SEQ_LEN)
X_test, y_test = create_sequences(test_df, SEQ_LEN)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------
# Build LSTM model
# -------------------------

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(SEQ_LEN, len(feature_cols)),
        return_sequences=False
    )
)

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -------------------------
# Train model
# -------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -------------------------
# Evaluate model
# -------------------------

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nLSTM Evaluation")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0,125],[0,125],'r--')
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("LSTM Predictions vs True RUL")
plt.show()

model.save("lstm_rul_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("LSTM model saved.")