import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "TES_100000_requested_variables_only.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==============================================")
print("TES PERFORMANCE ANALYSIS")
print("==============================================")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# STATIC FEATURES
# ============================================================

STATIC_FEATURES = [
    "inlet_temperature_C",
    "mass_flow_rate_kg_s",
    "initial_PCM_TES_temperature_C",
    "PCM_density_kg_m3",
    "PCM_specific_heat_J_kgK",
    "PCM_latent_heat_J_kg",
    "PCM_melting_temperature_C",
    "PCM_thermal_conductivity_W_mK",
    "fin_length_m",
    "tank_diameter_m"
]


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    STATIC_FEATURES + ["time_s"]
).reset_index(drop=True)


# ============================================================
# CREATE CASE ID
# ============================================================

df["case_id"] = (
    df.groupby(STATIC_FEATURES, sort=False)
    .ngroup()
)

print("Number of TES cases:", df["case_id"].nunique())


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

performance_results = []


for case_id, group in df.groupby("case_id"):

    group = group.sort_values("time_s").reset_index(drop=True)

    time = group["time_s"].values
    temperature = group["temperature_profile_C"].values

    initial_temperature = temperature[0]
    final_temperature = temperature[-1]

    maximum_temperature = np.max(temperature)
    minimum_temperature = np.min(temperature)

    temperature_rise = (
        final_temperature - initial_temperature
    )

    total_time = (
        time[-1] - time[0]
    )

    # --------------------------------------------------------
    # AVERAGE HEATING RATE
    # --------------------------------------------------------

    if total_time > 0:
        average_heating_rate = (
            temperature_rise / total_time
        )
    else:
        average_heating_rate = np.nan

    # --------------------------------------------------------
    # MAXIMUM INSTANTANEOUS HEATING RATE
    # --------------------------------------------------------

    dt = np.diff(time)
    dT = np.diff(temperature)

    valid = dt > 0

    if np.any(valid):

        heating_rates = (
            dT[valid] / dt[valid]
        )

        maximum_heating_rate = np.max(
            heating_rates
        )

    else:
        maximum_heating_rate = np.nan

    # --------------------------------------------------------
    # TIME TO TARGET TEMPERATURE
    # --------------------------------------------------------

    def time_to_temperature(target):

        indices = np.where(
            temperature >= target
        )[0]

        if len(indices) == 0:
            return np.nan

        return time[indices[0]]

    time_to_50 = time_to_temperature(50)
    time_to_60 = time_to_temperature(60)
    time_to_70 = time_to_temperature(70)
    time_to_80 = time_to_temperature(80)

    # --------------------------------------------------------
    # AREA UNDER TEMPERATURE-TIME CURVE
    # --------------------------------------------------------

    temperature_time_area = np.trapezoid(
        temperature,
        time
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    first_row = group.iloc[0]

    performance_results.append({

        "case_id": case_id,

        "inlet_temperature_C":
            first_row["inlet_temperature_C"],

        "mass_flow_rate_kg_s":
            first_row["mass_flow_rate_kg_s"],

        "initial_PCM_TES_temperature_C":
            first_row["initial_PCM_TES_temperature_C"],

        "PCM_density_kg_m3":
            first_row["PCM_density_kg_m3"],

        "PCM_specific_heat_J_kgK":
            first_row["PCM_specific_heat_J_kgK"],

        "PCM_latent_heat_J_kg":
            first_row["PCM_latent_heat_J_kg"],

        "PCM_melting_temperature_C":
            first_row["PCM_melting_temperature_C"],

        "PCM_thermal_conductivity_W_mK":
            first_row["PCM_thermal_conductivity_W_mK"],

        "fin_length_m":
            first_row["fin_length_m"],

        "tank_diameter_m":
            first_row["tank_diameter_m"],

        "initial_temperature_C":
            initial_temperature,

        "final_temperature_C":
            final_temperature,

        "minimum_temperature_C":
            minimum_temperature,

        "maximum_temperature_C":
            maximum_temperature,

        "temperature_rise_C":
            temperature_rise,

        "total_time_s":
            total_time,

        "average_heating_rate_C_per_s":
            average_heating_rate,

        "maximum_heating_rate_C_per_s":
            maximum_heating_rate,

        "time_to_50C_s":
            time_to_50,

        "time_to_60C_s":
            time_to_60,

        "time_to_70C_s":
            time_to_70,

        "time_to_80C_s":
            time_to_80,

        "temperature_time_area_Cs":
            temperature_time_area
    })


# ============================================================
# CREATE PERFORMANCE DATAFRAME
# ============================================================

performance_df = pd.DataFrame(
    performance_results
)


# ============================================================
# SAVE PERFORMANCE DATA
# ============================================================

output_csv = os.path.join(
    RESULTS_DIR,
    "tes_performance_analysis.csv"
)

performance_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n==============================================")
print("TES PERFORMANCE RESULTS")
print("==============================================")

print(
    performance_df[
        [
            "case_id",
            "initial_temperature_C",
            "final_temperature_C",
            "temperature_rise_C",
            "average_heating_rate_C_per_s",
            "time_to_70C_s"
        ]
    ].head(10)
)


# ============================================================
# OVERALL STATISTICS
# ============================================================

print("\n==============================================")
print("OVERALL PERFORMANCE STATISTICS")
print("==============================================")

print(
    "\nAverage final temperature:",
    performance_df[
        "final_temperature_C"
    ].mean()
)

print(
    "Average temperature rise:",
    performance_df[
        "temperature_rise_C"
    ].mean()
)

print(
    "Average heating rate:",
    performance_df[
        "average_heating_rate_C_per_s"
    ].mean()
)

print(
    "Average maximum heating rate:",
    performance_df[
        "maximum_heating_rate_C_per_s"
    ].mean()
)


# ============================================================
# PLOT 1
# TEMPERATURE RISE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    performance_df["temperature_rise_C"],
    bins=30
)

plt.xlabel(
    "Temperature Rise (°C)"
)

plt.ylabel(
    "Number of TES Cases"
)

plt.title(
    "TES Temperature Rise Distribution"
)

plt.grid(True)

plt.tight_layout()

plot1 = os.path.join(
    RESULTS_DIR,
    "tes_temperature_rise_distribution.png"
)

plt.savefig(
    plot1,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2
# FINAL TEMPERATURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    performance_df["final_temperature_C"],
    bins=30
)

plt.xlabel(
    "Final Temperature (°C)"
)

plt.ylabel(
    "Number of TES Cases"
)

plt.title(
    "TES Final Temperature Distribution"
)

plt.grid(True)

plt.tight_layout()

plot2 = os.path.join(
    RESULTS_DIR,
    "tes_final_temperature_distribution.png"
)

plt.savefig(
    plot2,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3
# HEATING RATE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    performance_df[
        "average_heating_rate_C_per_s"
    ].dropna(),
    bins=30
)

plt.xlabel(
    "Average Heating Rate (°C/s)"
)

plt.ylabel(
    "Number of TES Cases"
)

plt.title(
    "TES Average Heating Rate Distribution"
)

plt.grid(True)

plt.tight_layout()

plot3 = os.path.join(
    RESULTS_DIR,
    "tes_heating_rate_distribution.png"
)

plt.savefig(
    plot3,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4
# TIME TO TARGET TEMPERATURE
# ============================================================

target_columns = [
    "time_to_50C_s",
    "time_to_60C_s",
    "time_to_70C_s",
    "time_to_80C_s"
]

target_means = performance_df[
    target_columns
].mean()

plt.figure(figsize=(10, 6))

plt.bar(
    ["50°C", "60°C", "70°C", "80°C"],
    target_means.values
)

plt.xlabel(
    "Target Temperature"
)

plt.ylabel(
    "Average Time (s)"
)

plt.title(
    "Average Time to Reach Target Temperature"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plot4 = os.path.join(
    RESULTS_DIR,
    "tes_time_to_target_temperature.png"
)

plt.savefig(
    plot4,
    dpi=300
)

plt.close()


# ============================================================
# SAMPLE TEMPERATURE PROFILES
# ============================================================

sample_cases = performance_df[
    "case_id"
].head(5).tolist()

plt.figure(figsize=(10, 6))

for case_id in sample_cases:

    case_data = df[
        df["case_id"] == case_id
    ].sort_values("time_s")

    plt.plot(
        case_data["time_s"],
        case_data["temperature_profile_C"],
        label=f"Case {case_id}"
    )

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.title(
    "Sample TES Temperature Profiles"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plot5 = os.path.join(
    RESULTS_DIR,
    "tes_sample_temperature_profiles.png"
)

plt.savefig(
    plot5,
    dpi=300
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("ANALYSIS COMPLETED")
print("==============================================")

print("\nGenerated files:")

print(
    "1.",
    output_csv
)

print(
    "2.",
    plot1
)

print(
    "3.",
    plot2
)

print(
    "4.",
    plot3
)

print(
    "5.",
    plot4
)

print(
    "6.",
    plot5
)

print("\nNext step: integrate these TES performance metrics")
print("into the Streamlit dashboard.")