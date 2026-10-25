import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# 1. Load dataset
df = pd.read_csv(
    "dataset/TES_100000_requested_variables_only.csv"
)

# 2. Input and target
X = df.drop(columns=["temperature_profile_C"])
y = df["temperature_profile_C"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 6. Prediction
y_pred = model.predict(X_test_scaled)


# =====================================================
# GRAPH 1: ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Random Forest: Actual vs Predicted Temperature")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/random_forest_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 2: TEMPERATURE PROFILE
# =====================================================

plt.figure(figsize=(10, 6))

# Show first 200 test samples
plt.plot(
    range(200),
    y_test.iloc[:200].values,
    label="Actual"
)

plt.plot(
    range(200),
    y_pred[:200],
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Temperature (°C)")
plt.title("Random Forest Temperature Prediction")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/random_forest_temperature_prediction.png",
    dpi=300
)

plt.show()

print("\nVisualization completed successfully!")

print("\nSaved files:")
print("results/random_forest_actual_vs_predicted.png")
print("results/random_forest_temperature_prediction.png")