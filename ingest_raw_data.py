import pandas as pd
from sqlalchemy import create_engine, text

CSV_PATH = "D:/Capstone Project/Dataset/train_FD001.txt"
DB_URL = "postgresql://<user>:<password>@localhost:5432/pred_maintainance"

cols = (
    ["engine_id", "cycle",
     "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# -------------------------
# Load raw data
# -------------------------

df = pd.read_csv(
    CSV_PATH,
    sep=r"\s+",
    header=None,
    names=cols
)

# -------------------------
# Compute RUL
# -------------------------

max_cycles = (
    df.groupby("engine_id")["cycle"]
    .max()
    .reset_index()
    .rename(columns={"cycle": "max_cycle"})
)

df = df.merge(max_cycles, on="engine_id", how="left")

df["rul"] = df["max_cycle"] - df["cycle"]

# 🔥 IMPORTANT: Cap RUL at 125 (literature standard)
df["rul"] = df["rul"].clip(upper=125)

# Drop helper column
df = df.drop(columns=["max_cycle"])

# -------------------------
# Sanity check
# -------------------------

print("RUL min:", df["rul"].min())
print("RUL max:", df["rul"].max())
print("Total rows:", len(df))

# -------------------------
# Write to PostgreSQL (clean reset)
# -------------------------

engine = create_engine(DB_URL)

with engine.begin() as conn:
    conn.execute(text("DROP SCHEMA IF EXISTS raw CASCADE"))
    conn.execute(text("CREATE SCHEMA raw"))

df.to_sql(
    "turbofan_train",
    engine,
    schema="raw",
    index=False,
    if_exists="replace"
)

print("Raw ingestion complete.")