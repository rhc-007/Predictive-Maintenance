import pandas as pd
import numpy as np
import joblib
import json
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -----------------------
# Load data
# -----------------------

engine = create_engine(
    "postgresql://postgres:12345678@localhost:5432/pred_maintainance"
)

df = pd.read_sql("SELECT * FROM turbofan_features", engine)
df = df.dropna()

# Separate features and target
X = df.drop(columns=["rul", "engine_id", "data_version"])
y = df["rul"]

feature_columns = X.columns.tolist()

# -----------------------
# Engine-level split
# -----------------------

engine_ids = df["engine_id"].unique()

train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_mask = df["engine_id"].isin(train_engines)
test_mask = df["engine_id"].isin(test_engines)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

groups_train = df.loc[train_mask, "engine_id"]

# -----------------------
# Hyperparameter grids
# -----------------------

rf_param_grid = {
    "n_estimators": [300],
    "max_depth": [12, 15, None],
    "min_samples_split": [2, 5]
}

gb_param_grid = {
    "n_estimators": [300],
    "learning_rate": [0.03, 0.05],
    "max_depth": [3, 4],
    "subsample": [0.8, 1.0]
}
# -----------------------
# GroupKFold setup
# -----------------------

gkf = GroupKFold(n_splits=3)

# -----------------------
# Initialize models
# -----------------------

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
gb = GradientBoostingRegressor(random_state=42)

# -----------------------
# Grid Search with GroupKFold
# -----------------------

rf_grid = GridSearchCV(
    rf,
    rf_param_grid,
    cv=gkf,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

gb_grid = GridSearchCV(
    gb,
    gb_param_grid,
    cv=gkf,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

print("Tuning Random Forest...")
rf_grid.fit(X_train, y_train, groups=groups_train)

print("Tuning Gradient Boosting...")
gb_grid.fit(X_train, y_train, groups=groups_train)

# -----------------------
# Compare best models (engine-level test set)
# -----------------------

best_models = {
    "Random Forest": rf_grid.best_estimator_,
    "Gradient Boosting": gb_grid.best_estimator_
}

results = []

for name, model in best_models.items():
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append((name, mae, rmse))

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE"]
).sort_values("RMSE")

print("\nFinal Comparison (Engine-level split):")
print(results_df)

# -----------------------
# Select best model
# -----------------------

best_model_name = results_df.iloc[0]["Model"]
final_model = best_models[best_model_name]

print(f"\nSelected Final Model: {best_model_name}")

# -----------------------
# Retrain on FULL dataset
# -----------------------

final_model.fit(X, y)

# -----------------------
# Save model + metadata
# -----------------------

joblib.dump(final_model, "final_model.joblib")

metadata = {
    "model_type": best_model_name,
    "best_params": final_model.get_params(),
    "features": feature_columns
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nModel and metadata saved successfully.")