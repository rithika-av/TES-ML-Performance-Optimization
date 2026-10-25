import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/TES_100000_requested_variables_only.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("XGBOOST - TES TEMPERATURE PREDICTION")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. INPUT FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["temperature_profile_C"])
y = df["temperature_profile_C"]

print("\nInput features:", X.shape)
print("Target:", y.shape)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. CREATE XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost model...")
print("Please wait...")


# ============================================================
# 6. TRAIN MODEL
# ============================================================

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.6f} °C")
print(f"RMSE : {rmse:.6f} °C")
print(f"R²   : {r2:.6f}")


# ============================================================
# 9. SAVE MODEL
# ============================================================

MODEL_PATH = "results/xgboost_model.pkl"
SCALER_PATH = "results/xgboost_scaler.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nScaler saved to:")
print(SCALER_PATH)


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("XGBoost - Actual vs Predicted Temperature")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/xgboost_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. TEMPERATURE PREDICTION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

sample_count = 200

plt.plot(
    range(sample_count),
    y_test.iloc[:sample_count].values,
    label="Actual"
)

plt.plot(
    range(sample_count),
    y_pred[:sample_count],
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Temperature (°C)")
plt.title("XGBoost - Temperature Prediction")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/xgboost_temperature_prediction.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETED")
print("=" * 60)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"MAE              : {mae:.6f} °C")
print(f"RMSE             : {rmse:.6f} °C")
print(f"R²               : {r2:.6f}")

print("\nGenerated files:")
print("1. results/xgboost_model.pkl")
print("2. results/xgboost_scaler.pkl")
print("3. results/xgboost_actual_vs_predicted.png")
print("4. results/xgboost_temperature_prediction.png")