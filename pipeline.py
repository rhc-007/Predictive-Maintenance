from sqlalchemy import create_engine, text
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lit, lag, avg
)

# -------------------
# PostgreSQL setup
# -------------------

DB_URL = "postgresql://postgres:12345678@localhost:5432/pred_maintainance"
engine = create_engine(DB_URL)

with engine.begin() as conn:
    conn.execute(text("""
        DROP TABLE IF EXISTS turbofan_features
    """))

# -------------------
# Spark session
# -------------------

spark = (
    SparkSession.builder
    .appName("turbofan-feature-engineering-clean")
    .master("local[*]")
    .getOrCreate()
)

jdbc_url = "jdbc:postgresql://localhost:5432/pred_maintainance"
jdbc_props = {
    "user": "postgres",
    "password": "12345678",
    "driver": "org.postgresql.Driver"
}

# -------------------
# Read raw data
# -------------------

df = spark.read.jdbc(
    url=jdbc_url,
    table="raw.turbofan_train",
    properties=jdbc_props
)

# -------------------
# Feature engineering
# -------------------

window_spec = Window.partitionBy("engine_id").orderBy("cycle")
rolling_window = window_spec.rowsBetween(-4, 0)

# Select useful sensors (literature-based)
useful_sensor_ids = [2, 3, 4, 7, 11, 12, 15, 17, 20, 21]
sensor_cols = [f"sensor_{i}" for i in useful_sensor_ids]

df_features = df

# Generate delta + rolling features
for sensor in sensor_cols:
    df_features = (
        df_features
        .withColumn(
            f"{sensor}_delta",
            col(sensor) - lag(sensor, 1).over(window_spec)
        )
        .withColumn(
            f"{sensor}_roll_mean",
            avg(sensor).over(rolling_window)
        )
    )

df_features = df_features.withColumn("data_version", lit("sensor_only_v2"))

# -------------------
# Final feature set (NO cycle, NO cycle_ratio)
# -------------------

select_cols = (
    ["engine_id"]
    + sensor_cols
    + [f"{s}_delta" for s in sensor_cols]
    + [f"{s}_roll_mean" for s in sensor_cols]
    + ["rul", "data_version"]
)

df_curated = df_features.select(*select_cols)

# -------------------
# Write to PostgreSQL
# -------------------

df_curated.write.jdbc(
    url=jdbc_url,
    table="turbofan_features",
    mode="overwrite",
    properties=jdbc_props
)

row_count = df_curated.count()

spark.stop()
print(f"Spark pipeline run complete — {row_count} rows written.")