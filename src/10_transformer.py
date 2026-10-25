import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "dataset/TES_100000_requested_variables_only.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING TES DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURES AND TARGET
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
# 4. CREATE CASE IDs
# ============================================================

# Dataset contains 100 time steps per TES case
df["case_id"] = np.arange(len(df)) // 100

df = df.sort_values(
    ["case_id", "time_s"]
).reset_index(drop=True)

print("Number of cases:", df["case_id"].nunique())
print(
    "Time steps per case:",
    df.groupby("case_id").size().iloc[0]
)


# ============================================================
# 5. CREATE SEQUENCES
# ============================================================

time_steps = 100
n_features = len(feature_columns)

X_sequences = []
y_sequences = []

for case_id in df["case_id"].unique():

    case_data = df[df["case_id"] == case_id]

    X_case = case_data[feature_columns].values
    y_case = case_data[target_column].values

    if len(case_data) == time_steps:

        X_sequences.append(X_case)
        y_sequences.append(y_case)


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print()
print("Sequence data:")
print("X shape:", X_sequences.shape)
print("y shape:", y_sequences.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

n_cases = len(X_sequences)

train_cases = int(0.8 * n_cases)

X_train = X_sequences[:train_cases]
X_test = X_sequences[train_cases:]

y_train = y_sequences[:train_cases]
y_test = y_sequences[train_cases:]

print()
print("Training cases:", len(X_train))
print("Testing cases :", len(X_test))


# ============================================================
# 7. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

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

X_train_scaled = X_train_scaled.reshape(
    X_train.shape[0],
    time_steps,
    n_features
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape[0],
    time_steps,
    n_features
)

joblib.dump(
    scaler,
    os.path.join(
        RESULTS_DIR,
        "transformer_scaler.pkl"
    )
)


# ============================================================
# 8. TRANSFORMER BLOCK
# ============================================================

def transformer_block(
    inputs,
    num_heads=4,
    key_dim=32,
    ff_dim=64,
    dropout_rate=0.1
):

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate
    )(
        inputs,
        inputs
    )

    # Residual connection
    attention_output = Add()(
        [inputs, attention_output]
    )

    # Layer normalization
    attention_output = LayerNormalization()(
        attention_output
    )

    # Feed-forward network
    ff_output = Dense(
        ff_dim,
        activation="relu"
    )(attention_output)

    ff_output = Dropout(
        dropout_rate
    )(ff_output)

    ff_output = Dense(
        inputs.shape[-1]
    )(ff_output)

    # Second residual connection
    output = Add()(
        [attention_output, ff_output]
    )

    # Second normalization
    output = LayerNormalization()(
        output
    )

    return output


# ============================================================
# 9. BUILD TRANSFORMER MODEL
# ============================================================

inputs = Input(
    shape=(time_steps, n_features)
)

# Project input features into a larger representation
x = Dense(
    64
)(inputs)

# Transformer block
x = transformer_block(
    x,
    num_heads=4,
    key_dim=16,
    ff_dim=128,
    dropout_rate=0.1
)

# Second Transformer block
x = transformer_block(
    x,
    num_heads=4,
    key_dim=16,
    ff_dim=128,
    dropout_rate=0.1
)

# Final processing
x = Dense(
    32,
    activation="relu"
)(x)

x = Dropout(
    0.1
)(x)

# One temperature prediction for every time step
outputs = Dense(
    1
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse"
)


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRANSFORMER MODEL")
print("=" * 70)

model.summary()


# ============================================================
# 12. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 13. TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRANSFORMER TRAINING")
print("=" * 70)

history = model.fit(

    X_train_scaled,
    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=16,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 14. PREDICTION
# ============================================================

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled,
    verbose=0
)

print("Prediction shape:", y_pred.shape)


# ============================================================
# 15. CALCULATE METRICS
# ============================================================

y_test_flat = y_test.reshape(-1)

y_pred_flat = y_pred.reshape(-1)

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
# 16. RESULTS
# ============================================================

print()
print("=" * 70)
print("TRANSFORMER RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.6f} °C")
print(f"RMSE : {rmse:.6f} °C")
print(f"R²   : {r2:.6f}")


# ============================================================
# 17. SAVE MODEL
# ============================================================

model.save(
    os.path.join(
        RESULTS_DIR,
        "transformer_model.keras"
    )
)

print()
print("Model saved:")
print("results/transformer_model.keras")


# ============================================================
# 18. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test_flat,
    y_pred_flat,
    alpha=0.3
)

plt.plot(
    [
        y_test_flat.min(),
        y_test_flat.max()
    ],
    [
        y_test_flat.min(),
        y_test_flat.max()
    ],
    linestyle="--"
)

plt.xlabel(
    "Actual Temperature (°C)"
)

plt.ylabel(
    "Predicted Temperature (°C)"
)

plt.title(
    "Transformer: Actual vs Predicted Temperature"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "transformer_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. TEMPERATURE PROFILE
# ============================================================

sample_index = 0

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    y_test[sample_index],
    label="Actual"
)

plt.plot(
    y_pred[sample_index, :, 0],
    label="Predicted"
)

plt.xlabel(
    "Time Step"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.title(
    "Transformer Temperature Profile"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "transformer_temperature_prediction.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.title(
    "Transformer Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "transformer_training_loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 21. COMPLETED
# ============================================================

print()
print("=" * 70)
print("TRANSFORMER TRAINING COMPLETED")
print("=" * 70)

print("Saved files:")
print("1. results/transformer_model.keras")
print("2. results/transformer_scaler.pkl")
print("3. results/transformer_actual_vs_predicted.png")
print("4. results/transformer_temperature_prediction.png")
print("5. results/transformer_training_loss.png")