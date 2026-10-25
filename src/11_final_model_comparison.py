import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. FINAL MODEL RESULTS
# ============================================================
#
# These are the measured results from the completed models.
#
# RF and XGBoost:
# obtained from results/model_comparison.csv
#
# LSTM, CNN and Transformer:
# obtained from their respective training runs.
#
# ============================================================

results = [

    {
        "Model": "Random Forest",
        "MAE": 0.196444,
        "RMSE": 0.346689,
        "R2": 0.999573
    },

    {
        "Model": "XGBoost",
        "MAE": 0.227936,
        "RMSE": 0.299196,
        "R2": 0.999682
    },

    {
        "Model": "LSTM",
        "MAE": 12.940176,
        "RMSE": 14.846173,
        "R2": 0.133345
    },

    {
        "Model": "1D-CNN",
        "MAE": 5.676392,
        "RMSE": 7.227829,
        "R2": 0.823433
    },

    {
        "Model": "Transformer",
        "MAE": 0.658201,
        "RMSE": 0.923595,
        "R2": 0.997117
    }
]


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)

print("=" * 75)
print("FINAL TES MODEL COMPARISON")
print("=" * 75)

print(
    df.to_string(index=False)
)


# ============================================================
# 4. SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "final_model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print()
print("Saved:")
print(csv_path)


# ============================================================
# 5. MODEL METRICS COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)


# ---------------- MAE ----------------

axes[0].bar(
    df["Model"],
    df["MAE"]
)

axes[0].set_title(
    "MAE Comparison"
)

axes[0].set_ylabel(
    "MAE (°C)"
)

axes[0].tick_params(
    axis="x",
    rotation=45
)


# ---------------- RMSE ----------------

axes[1].bar(
    df["Model"],
    df["RMSE"]
)

axes[1].set_title(
    "RMSE Comparison"
)

axes[1].set_ylabel(
    "RMSE (°C)"
)

axes[1].tick_params(
    axis="x",
    rotation=45
)


# ---------------- R2 ----------------

axes[2].bar(
    df["Model"],
    df["R2"]
)

axes[2].set_title(
    "R² Comparison"
)

axes[2].set_ylabel(
    "R²"
)

axes[2].tick_params(
    axis="x",
    rotation=45
)


plt.tight_layout()

metrics_path = os.path.join(
    RESULTS_DIR,
    "final_model_metrics_comparison.png"
)

plt.savefig(
    metrics_path,
    dpi=300
)

plt.close()


# ============================================================
# 6. ERROR COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["Model"],
    df["RMSE"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "RMSE (°C)"
)

plt.title(
    "TES Temperature Prediction Error Comparison"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

error_path = os.path.join(
    RESULTS_DIR,
    "final_model_error_comparison.png"
)

plt.savefig(
    error_path,
    dpi=300
)

plt.close()


# ============================================================
# 7. R2 COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["Model"],
    df["R2"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R²"
)

plt.title(
    "TES Temperature Prediction R² Comparison"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

r2_path = os.path.join(
    RESULTS_DIR,
    "final_model_r2_comparison.png"
)

plt.savefig(
    r2_path,
    dpi=300
)

plt.close()


# ============================================================
# 8. COMPLETION
# ============================================================

print()
print("=" * 75)
print("FINAL MODEL COMPARISON COMPLETED")
print("=" * 75)

print()
print("Generated files:")

print(
    "1. results/final_model_comparison.csv"
)

print(
    "2. results/final_model_metrics_comparison.png"
)

print(
    "3. results/final_model_error_comparison.png"
)

print(
    "4. results/final_model_r2_comparison.png"
)