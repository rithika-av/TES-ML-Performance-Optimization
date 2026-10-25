import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/TES_100000_requested_variables_only.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("RANDOM FOREST vs XGBOOST COMPARISON")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["temperature_profile_C"])
y = df["temperature_profile_C"]


# ============================================================
# 3. SAME TRAIN / TEST SPLIT
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
# 4. LOAD RANDOM FOREST
# ============================================================

rf_model = joblib.load(
    "results/random_forest_model.pkl"
)

rf_scaler = joblib.load(
    "results/scaler.pkl"
)


# ============================================================
# 5. LOAD XGBOOST
# ============================================================

xgb_model = joblib.load(
    "results/xgboost_model.pkl"
)

xgb_scaler = joblib.load(
    "results/xgboost_scaler.pkl"
)


# ============================================================
# 6. SCALE DATA
# ============================================================

X_test_rf = rf_scaler.transform(X_test)

X_test_xgb = xgb_scaler.transform(X_test)


# ============================================================
# 7. PREDICTIONS
# ============================================================

print("\nGenerating Random Forest predictions...")

rf_pred = rf_model.predict(X_test_rf)

print("Generating XGBoost predictions...")

xgb_pred = xgb_model.predict(X_test_xgb)


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_pred)
)

rf_r2 = r2_score(
    y_test,
    rf_pred
)


xgb_mae = mean_absolute_error(y_test, xgb_pred)

xgb_rmse = np.sqrt(
    mean_squared_error(y_test, xgb_pred)
)

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print("\nRandom Forest")
print("-" * 30)
print(f"MAE  : {rf_mae:.6f} °C")
print(f"RMSE : {rf_rmse:.6f} °C")
print(f"R²   : {rf_r2:.6f}")

print("\nXGBoost")
print("-" * 30)
print(f"MAE  : {xgb_mae:.6f} °C")
print(f"RMSE : {xgb_rmse:.6f} °C")
print(f"R²   : {xgb_r2:.6f}")


# ============================================================
# 10. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        rf_rmse,
        xgb_rmse
    ],
    "R2": [
        rf_r2,
        xgb_r2
    ]
})

print("\n")
print(comparison.to_string(index=False))


# ============================================================
# 11. SAVE COMPARISON TABLE
# ============================================================

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison table saved:")
print("results/model_comparison.csv")


# ============================================================
# 12. METRIC COMPARISON GRAPH
# ============================================================

metrics = ["MAE", "RMSE", "R2"]

rf_values = [
    rf_mae,
    rf_rmse,
    rf_r2
]

xgb_values = [
    xgb_mae,
    xgb_rmse,
    xgb_r2
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    rf_values,
    width,
    label="Random Forest"
)

plt.bar(
    x + width / 2,
    xgb_values,
    width,
    label="XGBoost"
)

plt.xticks(x, metrics)

plt.ylabel("Metric Value")
plt.title("Random Forest vs XGBoost")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/rf_vs_xgboost_metrics.png",
    dpi=300
)

plt.close()

print("Metric comparison graph saved:")
print("results/rf_vs_xgboost_metrics.png")


# ============================================================
# 13. ACTUAL VS PREDICTED COMPARISON
# ============================================================

sample_count = 200

plt.figure(figsize=(12, 6))

plt.plot(
    range(sample_count),
    y_test.iloc[:sample_count].values,
    label="Actual"
)

plt.plot(
    range(sample_count),
    rf_pred[:sample_count],
    label="Random Forest"
)

plt.plot(
    range(sample_count),
    xgb_pred[:sample_count],
    label="XGBoost"
)

plt.xlabel("Test Sample")
plt.ylabel("Temperature (°C)")

plt.title(
    "Temperature Prediction: Random Forest vs XGBoost"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/rf_vs_xgboost_prediction.png",
    dpi=300
)

plt.close()

print("Prediction comparison graph saved:")
print("results/rf_vs_xgboost_prediction.png")


# ============================================================
# 14. FINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("1. results/model_comparison.csv")
print("2. results/rf_vs_xgboost_metrics.png")
print("3. results/rf_vs_xgboost_prediction.png")