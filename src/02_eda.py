import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_csv(
    "dataset/TES_100000_requested_variables_only.csv"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ---------------------------------------
# 2. Temperature vs Time
# ---------------------------------------

plt.figure(figsize=(10, 6))

# Take first TES case (first 100 time points)
first_case = df.iloc[:100]

plt.plot(
    first_case["time_s"],
    first_case["temperature_profile_C"]
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("TES Temperature Profile vs Time")

plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------
# 3. Inlet Temperature vs Temperature Profile
# ---------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["inlet_temperature_C"].iloc[::100],
    df["temperature_profile_C"].iloc[::100],
    alpha=0.5
)

plt.xlabel("Inlet Temperature (°C)")
plt.ylabel("Temperature Profile (°C)")
plt.title("Inlet Temperature vs Temperature Profile")

plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------
# 4. Mass Flow Rate vs Temperature
# ---------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["mass_flow_rate_kg_s"].iloc[::100],
    df["temperature_profile_C"].iloc[::100],
    alpha=0.5
)

plt.xlabel("Mass Flow Rate (kg/s)")
plt.ylabel("Temperature Profile (°C)")
plt.title("Mass Flow Rate vs Temperature Profile")

plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------

plt.figure(figsize=(12, 9))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


print("\nEDA completed successfully!")