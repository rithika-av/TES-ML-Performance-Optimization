import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv(
    "dataset/TES_100000_requested_variables_only.csv"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# 2. Separate input and target
X = df.drop(columns=["temperature_profile_C"])
y = df["temperature_profile_C"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling completed!")


# 5. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# 6. Train model
print("\nTraining Random Forest...")
model.fit(X_train_scaled, y_train)
joblib.dump(
    model,
    "results/random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "results/scaler.pkl"
)

print("\nModel saved successfully!")

print("Training completed!")


# 7. Prediction
y_pred = model.predict(X_test_scaled)


# 8. Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========== RANDOM FOREST RESULTS ==========")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)


# 9. Show actual vs predicted
results = pd.DataFrame({
    "Actual_Temperature_C": y_test.values,
    "Predicted_Temperature_C": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))