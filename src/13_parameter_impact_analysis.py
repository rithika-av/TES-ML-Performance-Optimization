import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "TES_100000_requested_variables_only.csv"
)

PERFORMANCE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tes_performance_analysis.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TES PARAMETER IMPACT ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

performance = pd.read_csv(
    PERFORMANCE_PATH
)

print("\nDataset loaded:", df.shape)
print("Performance data loaded:", performance.shape)


# ============================================================
# INPUT PARAMETERS
# ============================================================

parameters = [
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
# PERFORMANCE TARGETS
# ============================================================

targets = [
    "final_temperature_C",
    "temperature_rise_C",
    "average_heating_rate_C_per_s"
]


# ============================================================
# REMOVE DUPLICATE TIME ROWS
# ============================================================

case_parameters = (
    df.groupby(parameters, as_index=False)
    .first()
)


# ============================================================
# MERGE PARAMETERS + PERFORMANCE
# ============================================================

analysis_df = performance.merge(
    case_parameters,
    on=parameters,
    how="left",
    suffixes=("", "_dataset")
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

correlation_columns = (
    parameters +
    targets
)

correlation_matrix = (
    analysis_df[correlation_columns]
    .corr()
)


print("\n" + "=" * 70)
print("CORRELATION WITH TES PERFORMANCE")
print("=" * 70)


for target in targets:

    print("\nTarget:", target)

    correlations = (
        correlation_matrix[target]
        .drop(target)
        .sort_values(
            key=lambda x: abs(x),
            ascending=False
        )
    )

    print(
        correlations.to_string()
    )


# ============================================================
# SAVE CORRELATION MATRIX
# ============================================================

correlation_path = os.path.join(
    RESULTS_DIR,
    "tes_parameter_correlation.csv"
)

correlation_matrix.to_csv(
    correlation_path
)


# ============================================================
# PLOT 1
# PARAMETER VS FINAL TEMPERATURE
# ============================================================

for parameter in parameters:

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        analysis_df[parameter],
        analysis_df["final_temperature_C"],
        alpha=0.4
    )

    plt.xlabel(
        parameter
    )

    plt.ylabel(
        "Final Temperature (°C)"
    )

    plt.title(
        f"{parameter} vs Final TES Temperature"
    )

    plt.grid(True)

    plt.tight_layout()

    safe_name = parameter.replace(
        "/",
        "_"
    )

    output_path = os.path.join(
        RESULTS_DIR,
        f"impact_{safe_name}_final_temperature.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# PARAMETER IMPACT SUMMARY
# ============================================================

impact_rows = []

for target in targets:

    correlations = (
        correlation_matrix[target]
        .drop(target)
    )

    for parameter in parameters:

        impact_rows.append({

            "Target": target,

            "Parameter": parameter,

            "Correlation":
                correlations[parameter],

            "Absolute_Correlation":
                abs(correlations[parameter])
        })


impact_df = pd.DataFrame(
    impact_rows
)


impact_df = impact_df.sort_values(
    [
        "Target",
        "Absolute_Correlation"
    ],
    ascending=[
        True,
        False
    ]
)


impact_path = os.path.join(
    RESULTS_DIR,
    "tes_parameter_impact_summary.csv"
)

impact_df.to_csv(
    impact_path,
    index=False
)


# ============================================================
# PRINT TOP PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("TOP PARAMETERS FOR EACH PERFORMANCE METRIC")
print("=" * 70)


for target in targets:

    print(
        f"\n{target}:"
    )

    top_parameters = (
        impact_df[
            impact_df["Target"] == target
        ]
        .head(5)
    )

    print(
        top_parameters[
            [
                "Parameter",
                "Correlation"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# HEATMAP
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index
)

plt.title(
    "TES Parameter Correlation Matrix"
)

plt.tight_layout()

heatmap_path = os.path.join(
    RESULTS_DIR,
    "tes_parameter_correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PARAMETER IMPACT ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "1.",
    correlation_path
)

print(
    "2.",
    impact_path
)

print(
    "3.",
    heatmap_path
)

print(
    "\nIndividual parameter plots were also saved in:"
)

print(
    RESULTS_DIR
)