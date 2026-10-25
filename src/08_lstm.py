import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "dataset/TES_100000_requested_variables_only.csv"

MODEL_PATH = "results/lstm_model.keras"
SCALER_PATH = "results/lstm_scaler.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("LSTM - TES TEMPERATURE PREDICTION")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. SORT DATA
# ============================================================

# Each TES case contains 100 time steps.
# The dataset is ordered case-by-case.

df = df.sort_values(
    ["inlet_temperature_C",
     "mass_flow_rate_kg_s",
     "initial_PCM_TES_temperature_C",
     "PCM_density_kg_m3",
     "PCM_specific_heat_J_kgK",
     "PCM_latent_heat_J_kg",
     "PCM_melting_temperature_C",
     "PCM_thermal_conductivity_W_mK",
     "fin_length_m",
     "tank_diameter_m",
     "time_s"]
).reset_index(drop=True)


# ============================================================
# 4. INPUT FEATURES
# ============================================================

feature_columns = [
    "inlet_temperature_C",
    "mass_flow_rate_kg_s",
    "initial_PCM_TES_temperature_C",
    "PCM_density_kg_m3",
    "PCM_specific_heat_J_kgK",
    "PCM_latent_heat_J_kg",
    "PCM_melting_temperature_C",
    "PCM_thermal_conductivity_W_mK",
    "fin_length_m",
    "tank_diameter_m",
    "time_s"
]

target_column = "temperature_profile_C"


# ============================================================
# 5. CREATE CASE ID
# ============================================================

# Every 100 rows represent one TES simulation case.

df["case_id"] = np.arange(len(df)) // 100

number_of_cases = df["case_id"].nunique()

print("\nNumber of TES cases:", number_of_cases)
print("Time steps per case:", 100)


# ============================================================
# 6. CREATE SEQUENCES
# ============================================================

X_sequences = []
y_sequences = []

for case_id in range(number_of_cases):

    case_data = df[df["case_id"] == case_id]

    X_case = case_data[feature_columns].values
    y_case = case_data[target_column].values

    X_sequences.append(X_case)
    y_sequences.append(y_case)


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)


print("\nSequence shape:")
print("X:", X_sequences.shape)
print("y:", y_sequences.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT BY CASE
# ============================================================

# 80% cases for training
# 20% cases for testing

train_cases = int(number_of_cases * 0.80)

X_train = X_sequences[:train_cases]
y_train = y_sequences[:train_cases]

X_test = X_sequences[train_cases:]
y_test = y_sequences[train_cases:]


print("\nTraining cases:", X_train.shape[0])
print("Testing cases :", X_test.shape[0])


# ============================================================
# 8. SCALE INPUT FEATURES
# ============================================================

n_train_cases = X_train.shape[0]
n_test_cases = X_test.shape[0]
time_steps = X_train.shape[1]
n_features = X_train.shape[2]


scaler = StandardScaler()

# Convert 3D → 2D for scaler

X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_test_2d = X_test.reshape(
    -1,
    n_features
)


X_train_scaled = scaler.fit_transform(
    X_train_2d
)

X_test_scaled = scaler.transform(
    X_test_2d
)


# Convert back to 3D

X_train_scaled = X_train_scaled.reshape(
    n_train_cases,
    time_steps,
    n_features
)

X_test_scaled = X_test_scaled.reshape(
    n_test_cases,
    time_steps,
    n_features
)


# ============================================================
# 9. BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")


model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(time_steps, n_features)
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=True
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nModel summary:")

model.summary()


# ============================================================
# 11. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 12. TRAIN LSTM
# ============================================================

print("\n" + "=" * 70)
print("STARTING LSTM TRAINING")
print("=" * 70)

history = model.fit(

    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=30,

    batch_size=16,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 13. PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_scaled,
    verbose=0
)


# ============================================================
# 14. RESHAPE PREDICTIONS
# ============================================================

y_pred = y_pred.reshape(
    y_test.shape
)


# ============================================================
# 15. FLATTEN FOR METRICS
# ============================================================

y_test_flat = y_test.flatten()

y_pred_flat = y_pred.flatten()


# ============================================================
# 16. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_flat,
    y_pred_flat
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_flat,
        y_pred_flat
    )
)

r2 = r2_score(
    y_test_flat,
    y_pred_flat
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LSTM RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.6f} °C")
print(f"RMSE : {rmse:.6f} °C")
print(f"R²   : {r2:.6f}")


# ============================================================
# 18. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\nModel saved:")
print(MODEL_PATH)

print("\nScaler saved:")
print(SCALER_PATH)


# ============================================================
# 19. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_flat,
    y_pred_flat,
    alpha=0.3
)

minimum = min(
    y_test_flat.min(),
    y_pred_flat.min()
)

maximum = max(
    y_test_flat.max(),
    y_pred_flat.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")

plt.title(
    "LSTM - Actual vs Predicted Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/lstm_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. TEMPERATURE PROFILE
# ============================================================

# First test case

case_number = 0

plt.figure(figsize=(10, 6))

plt.plot(
    range(time_steps),
    y_test[case_number],
    label="Actual"
)

plt.plot(
    range(time_steps),
    y_pred[case_number],
    label="LSTM Predicted"
)

plt.xlabel("Time Step")

plt.ylabel("Temperature (°C)")

plt.title(
    "LSTM - TES Temperature Profile"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/lstm_temperature_prediction.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "LSTM Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/lstm_training_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETED")
print("=" * 70)

print(f"Training cases : {X_train.shape[0]}")
print(f"Testing cases  : {X_test.shape[0]}")
print(f"Time steps     : {time_steps}")
print(f"Features       : {n_features}")

print(f"\nMAE  : {mae:.6f} °C")
print(f"RMSE : {rmse:.6f} °C")
print(f"R²   : {r2:.6f}")

print("\nGenerated files:")

print("1. results/lstm_model.keras")
print("2. results/lstm_scaler.pkl")
print("3. results/lstm_actual_vs_predicted.png")
print("4. results/lstm_temperature_prediction.png")
print("5. results/lstm_training_loss.png")