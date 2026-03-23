import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, avg, max as spark_max

from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# Paths
# -------------------------
TEST_FILE = "D:\\Capstone Project\\Dataset\\test_FD001.txt"
RUL_FILE = "D:\\Capstone Project\\Dataset\\RUL_FD001.txt"

MODEL_PATH = "final_model.joblib"
METADATA_PATH = "model_metadata.json"

# -------------------------
# Load trained model
# -------------------------
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

feature_columns = metadata["features"]

print("Loaded model:", metadata["model_type"])

# -------------------------
# Spark session
# -------------------------
spark = (
    SparkSession.builder
    .appName("evaluate-test-fd001")
    .master("local[*]")
    .getOrCreate()
)

# -------------------------
# Column names (NASA format)
# -------------------------
cols = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# -------------------------
# Read test data (Pandas first)
# -------------------------
pdf_test_raw = pd.read_csv(
    TEST_FILE,
    sep=r"\s+",
    header=None
)

pdf_test_raw = pdf_test_raw.iloc[:, :len(cols)]
pdf_test_raw.columns = cols

# Convert Pandas → Spark
df_test = spark.createDataFrame(pdf_test_raw)

# -------------------------
# Feature engineering (same as training)
# -------------------------
window_spec = Window.partitionBy("engine_id").orderBy("cycle")
rolling_window = window_spec.rowsBetween(-4, 0)

df_test = df_test.withColumn(
    "max_cycle",
    spark_max("cycle").over(Window.partitionBy("engine_id"))
)

df_feat = df_test.withColumn(
    "cycle_ratio",
    col("cycle") / col("max_cycle")
)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

for s in sensor_cols:
    df_feat = (
        df_feat
        .withColumn(f"{s}_delta", col(s) - lag(s, 1).over(window_spec))
        .withColumn(f"{s}_roll_mean", avg(s).over(rolling_window))
    )

# -------------------------
# Keep only LAST cycle per engine
# -------------------------
df_last = df_feat.filter(col("cycle") == col("max_cycle"))

# Convert to Pandas
pdf_test = df_last.toPandas()

spark.stop()

# -------------------------
# Drop rows with nulls (from rolling/lag)
# -------------------------
pdf_test = pdf_test.dropna()

# -------------------------
# Align feature columns
# -------------------------
X_test = pdf_test[feature_columns]

# -------------------------
# Predict
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Load true RUL values
# -------------------------
true_rul = pd.read_csv(
    RUL_FILE,
    header=None,
    names=["rul"]
)

# If dropna removed rows (rare edge case), align lengths
if len(true_rul) != len(y_pred):
    true_rul = true_rul.iloc[:len(y_pred)]

# -------------------------
# Evaluation
# -------------------------
mae = mean_absolute_error(true_rul["rul"], y_pred)
rmse = np.sqrt(mean_squared_error(true_rul["rul"], y_pred))

print("\nFinal Evaluation on Test_FD001")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# -------------------------
# Feature Importance
# -------------------------

feature_names = metadata["features"]

importances = model.feature_importances_

feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 15 Most Important Features:")
print(feat_imp_df.head(15))

# Plot top 15
plt.figure(figsize=(10, 6))
plt.barh(
    feat_imp_df.head(15)["feature"][::-1],
    feat_imp_df.head(15)["importance"][::-1]
)
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -------------------------
# Engine-level error
# -------------------------

results_df = pd.DataFrame({
    "engine_id": pdf_test["engine_id"],
    "true_rul": true_rul["rul"],
    "pred_rul": y_pred
})

results_df["abs_error"] = np.abs(
    results_df["true_rul"] - results_df["pred_rul"]
)

engine_error = results_df.sort_values("abs_error", ascending=False)

print("\nWorst 10 Engines:")
print(engine_error.head(10))

# -------------------------
# Error vs True RUL plot
# -------------------------

plt.figure(figsize=(8,6))
plt.scatter(true_rul["rul"], y_pred, alpha=0.6)
plt.plot([0, 125], [0, 125], 'r--')
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs Predicted RUL")
plt.tight_layout()
plt.show()

# -------------------------
# Residual Analysis
# -------------------------

residuals = true_rul["rul"] - y_pred

print("\nResidual Statistics:")
print("Mean Residual:", np.mean(residuals))
print("Std Residual:", np.std(residuals))
print("Max Overprediction:", np.min(residuals))
print("Max Underprediction:", np.max(residuals))

plt.figure(figsize=(8,6))
plt.hist(residuals, bins=20)
plt.axvline(0, linestyle='--')
plt.xlabel("Residual (True - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()