# ============================================================
# STEP 14: GENETIC ALGORITHM OPTIMIZATION FOR TES
# ============================================================
# Project:
# Machine-Learning-Enabled Prediction and Optimization
# of Thermal Energy Storage (TES) Performance
#
# Purpose:
# Use Genetic Algorithm (GA) to search for TES parameter
# combinations that improve thermal performance.
#
# Dataset:
# TES_100000_requested_variables_only.csv
#
# Output:
# 1. ga_optimization_result.csv
# 2. ga_convergence.png
# 3. ga_before_after_comparison.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "TES_100000_requested_variables_only.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("GENETIC ALGORITHM OPTIMIZATION FOR TES")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. PARAMETERS USED BY GA
# ============================================================

parameter_columns = [
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
# 4. CREATE TES CASES
# ============================================================
# Each TES case contains:
# - 10 design/operating parameters
# - 100 temperature/time records
#
# We identify cases using the 10 static parameters.
# ============================================================

print("\nCreating TES cases...")

static_columns = parameter_columns

case_df = (
    df.groupby(static_columns, as_index=False)
      .agg(
          initial_temperature_C=("temperature_profile_C", "first"),
          final_temperature_C=("temperature_profile_C", "last"),
          minimum_temperature_C=("temperature_profile_C", "min"),
          maximum_temperature_C=("temperature_profile_C", "max"),
          total_time_s=("time_s", "max")
      )
)

# Temperature rise
case_df["temperature_rise_C"] = (
    case_df["final_temperature_C"]
    - case_df["initial_temperature_C"]
)

# Average heating rate
case_df["average_heating_rate_C_per_s"] = np.where(
    case_df["total_time_s"] > 0,
    case_df["temperature_rise_C"] / case_df["total_time_s"],
    0
)

print("Number of TES cases:", len(case_df))


# ============================================================
# 5. TIME TO 80°C
# ============================================================
# For each TES case, determine the first time at which
# temperature reaches 80°C.
#
# If 80°C is not reached, charging time is marked NaN.
# ============================================================

print("\nCalculating time to 80°C...")

time_to_80 = []

for _, group in df.groupby(static_columns):

    group = group.sort_values("time_s")

    reached = group[group["temperature_profile_C"] >= 80.0]

    if len(reached) > 0:
        time_value = reached["time_s"].iloc[0]
    else:
        time_value = np.nan

    row = dict(zip(static_columns, group.iloc[0][static_columns]))
    row["time_to_80_C_s"] = time_value

    time_to_80.append(row)

time_to_80_df = pd.DataFrame(time_to_80)

case_df = case_df.merge(
    time_to_80_df,
    on=static_columns,
    how="left"
)

# Replace cases that never reach 80°C with maximum observed time.
max_time = case_df["total_time_s"].max()

case_df["time_to_80_C_s_fitness"] = (
    case_df["time_to_80_C_s"]
    .fillna(max_time)
)


# ============================================================
# 6. NORMALIZATION FUNCTION
# ============================================================

def normalize(series):
    """
    Min-max normalization.
    Result is between 0 and 1.
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


# ============================================================
# 7. CALCULATE PERFORMANCE SCORE
# ============================================================
#
# Higher:
#   - Final temperature
#   - Heating rate
#
# Lower:
#   - Time required to reach 80°C
#
# Fitness:
#
# 50% final temperature
# 30% heating rate
# 20% charging time
#
# ============================================================

case_df["final_temperature_norm"] = normalize(
    case_df["final_temperature_C"]
)

case_df["heating_rate_norm"] = normalize(
    case_df["average_heating_rate_C_per_s"]
)

case_df["charging_time_norm"] = normalize(
    case_df["time_to_80_C_s_fitness"]
)

case_df["performance_score"] = (
    0.50 * case_df["final_temperature_norm"]
    + 0.30 * case_df["heating_rate_norm"]
    + 0.20 * (1 - case_df["charging_time_norm"])
)


# ============================================================
# 8. GENETIC ALGORITHM SETTINGS
# ============================================================

POPULATION_SIZE = 40
GENERATIONS = 50

MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.80

TOURNAMENT_SIZE = 3

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 9. PARAMETER BOUNDS
# ============================================================

parameter_bounds = {}

for column in parameter_columns:

    parameter_bounds[column] = (
        case_df[column].min(),
        case_df[column].max()
    )


# ============================================================
# 10. NORMALIZED DATA FOR DISTANCE CALCULATION
# ============================================================

parameter_min = case_df[parameter_columns].min()
parameter_max = case_df[parameter_columns].max()

parameter_range = parameter_max - parameter_min

parameter_range = parameter_range.replace(
    0,
    1
)

normalized_cases = (
    (case_df[parameter_columns] - parameter_min)
    / parameter_range
).to_numpy()


# ============================================================
# 11. FIND CLOSEST REAL TES CASE
# ============================================================
#
# The GA creates continuous parameter combinations.
#
# Because our dataset contains actual generated TES cases,
# each GA candidate is mapped to the closest available
# TES case in the dataset.
# ============================================================

def find_closest_case(candidate):

    candidate_array = np.array(candidate)

    normalized_candidate = (
        candidate_array - parameter_min.to_numpy()
    ) / parameter_range.to_numpy()

    distances = np.sqrt(
        np.sum(
            (normalized_cases - normalized_candidate) ** 2,
            axis=1
        )
    )

    index = np.argmin(distances)

    return index


# ============================================================
# 12. FITNESS FUNCTION
# ============================================================

def fitness(candidate):

    index = find_closest_case(candidate)

    return case_df.iloc[index]["performance_score"]


# ============================================================
# 13. CREATE RANDOM INDIVIDUAL
# ============================================================

def create_individual():

    individual = []

    for column in parameter_columns:

        minimum, maximum = parameter_bounds[column]

        value = rng.uniform(
            minimum,
            maximum
        )

        individual.append(value)

    return np.array(individual)


# ============================================================
# 14. CREATE INITIAL POPULATION
# ============================================================

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        population.append(
            create_individual()
        )

    return np.array(population)


# ============================================================
# 15. TOURNAMENT SELECTION
# ============================================================

def tournament_selection(
    population,
    fitness_values
):

    selected_indices = rng.choice(
        len(population),
        size=TOURNAMENT_SIZE,
        replace=False
    )

    best_index = selected_indices[
        np.argmax(
            fitness_values[selected_indices]
        )
    ]

    return population[best_index].copy()


# ============================================================
# 16. CROSSOVER
# ============================================================

def crossover(parent1, parent2):

    if rng.random() > CROSSOVER_RATE:

        return parent1.copy(), parent2.copy()

    point = rng.integers(
        1,
        len(parameter_columns)
    )

    child1 = np.concatenate(
        [
            parent1[:point],
            parent2[point:]
        ]
    )

    child2 = np.concatenate(
        [
            parent2[:point],
            parent1[point:]
        ]
    )

    return child1, child2


# ============================================================
# 17. MUTATION
# ============================================================

def mutate(individual):

    mutated = individual.copy()

    for i, column in enumerate(parameter_columns):

        if rng.random() < MUTATION_RATE:

            minimum, maximum = parameter_bounds[column]

            parameter_range_value = maximum - minimum

            noise = rng.normal(
                0,
                0.10 * parameter_range_value
            )

            mutated[i] += noise

            mutated[i] = np.clip(
                mutated[i],
                minimum,
                maximum
            )

    return mutated


# ============================================================
# 18. RUN GENETIC ALGORITHM
# ============================================================

print("\nStarting Genetic Algorithm...")

population = create_population()

best_individual = None
best_fitness = -np.inf

convergence_history = []


for generation in range(GENERATIONS):

    # --------------------------------------------------------
    # Calculate fitness
    # --------------------------------------------------------

    fitness_values = np.array([
        fitness(individual)
        for individual in population
    ])

    # --------------------------------------------------------
    # Find best solution
    # --------------------------------------------------------

    generation_best_index = np.argmax(
        fitness_values
    )

    generation_best_fitness = (
        fitness_values[generation_best_index]
    )

    generation_best_individual = (
        population[generation_best_index].copy()
    )

    # --------------------------------------------------------
    # Update global best
    # --------------------------------------------------------

    if generation_best_fitness > best_fitness:

        best_fitness = generation_best_fitness

        best_individual = (
            generation_best_individual.copy()
        )

    convergence_history.append(
        best_fitness
    )

    print(
        f"Generation {generation + 1:02d}/{GENERATIONS} "
        f"| Best Fitness: {best_fitness:.6f}"
    )

    # --------------------------------------------------------
    # Elitism
    # --------------------------------------------------------

    elite_index = np.argmax(
        fitness_values
    )

    elite = population[
        elite_index
    ].copy()

    new_population = [
        elite
    ]

    # --------------------------------------------------------
    # Create next generation
    # --------------------------------------------------------

    while len(new_population) < POPULATION_SIZE:

        parent1 = tournament_selection(
            population,
            fitness_values
        )

        parent2 = tournament_selection(
            population,
            fitness_values
        )

        child1, child2 = crossover(
            parent1,
            parent2
        )

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(
            child1
        )

        if len(new_population) < POPULATION_SIZE:

            new_population.append(
                child2
            )

    population = np.array(
        new_population
    )


# ============================================================
# 19. GET OPTIMIZED REAL TES CASE
# ============================================================

optimized_index = find_closest_case(
    best_individual
)

optimized_case = case_df.iloc[
    optimized_index
].copy()


# ============================================================
# 20. FIND BASELINE CASE
# ============================================================
#
# Baseline = average parameter values of the dataset.
#
# Find the real TES case closest to the average operating
# condition.
# ============================================================

average_parameters = case_df[
    parameter_columns
].mean().to_numpy()

baseline_index = find_closest_case(
    average_parameters
)

baseline_case = case_df.iloc[
    baseline_index
].copy()


# ============================================================
# 21. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("GENETIC ALGORITHM OPTIMIZATION RESULT")
print("=" * 70)

print("\nOptimized TES Parameters:")

for column in parameter_columns:

    print(
        f"{column:45s}: "
        f"{optimized_case[column]:.6f}"
    )


print("\nOptimized Performance:")

print(
    f"Final Temperature     : "
    f"{optimized_case['final_temperature_C']:.4f} °C"
)

print(
    f"Temperature Rise      : "
    f"{optimized_case['temperature_rise_C']:.4f} °C"
)

print(
    f"Average Heating Rate  : "
    f"{optimized_case['average_heating_rate_C_per_s']:.6f} °C/s"
)

if pd.isna(
    optimized_case["time_to_80_C_s"]
):

    print(
        "Time to 80°C          : "
        "Not reached"
    )

else:

    print(
        f"Time to 80°C          : "
        f"{optimized_case['time_to_80_C_s']:.2f} s"
    )


print(
    f"Performance Score      : "
    f"{optimized_case['performance_score']:.6f}"
)


print("\nBaseline Performance:")

print(
    f"Final Temperature     : "
    f"{baseline_case['final_temperature_C']:.4f} °C"
)

print(
    f"Temperature Rise      : "
    f"{baseline_case['temperature_rise_C']:.4f} °C"
)

print(
    f"Average Heating Rate  : "
    f"{baseline_case['average_heating_rate_C_per_s']:.6f} °C/s"
)

if pd.isna(
    baseline_case["time_to_80_C_s"]
):

    print(
        "Time to 80°C          : "
        "Not reached"
    )

else:

    print(
        f"Time to 80°C          : "
        f"{baseline_case['time_to_80_C_s']:.2f} s"
    )

print(
    f"Performance Score      : "
    f"{baseline_case['performance_score']:.6f}"
)


# ============================================================
# 22. IMPROVEMENT
# ============================================================

baseline_score = (
    baseline_case["performance_score"]
)

optimized_score = (
    optimized_case["performance_score"]
)

if baseline_score != 0:

    improvement = (
        (optimized_score - baseline_score)
        / abs(baseline_score)
    ) * 100

else:

    improvement = 0


print(
    f"\nPerformance Score Improvement: "
    f"{improvement:.2f}%"
)


# ============================================================
# 23. SAVE OPTIMIZATION RESULT
# ============================================================

result_rows = []

for column in parameter_columns:

    result_rows.append(
        {
            "parameter": column,
            "baseline_value": baseline_case[column],
            "optimized_value": optimized_case[column]
        }
    )


result_rows.extend(
    [
        {
            "parameter": "final_temperature_C",
            "baseline_value":
                baseline_case["final_temperature_C"],
            "optimized_value":
                optimized_case["final_temperature_C"]
        },
        {
            "parameter": "temperature_rise_C",
            "baseline_value":
                baseline_case["temperature_rise_C"],
            "optimized_value":
                optimized_case["temperature_rise_C"]
        },
        {
            "parameter": "average_heating_rate_C_per_s",
            "baseline_value":
                baseline_case[
                    "average_heating_rate_C_per_s"
                ],
            "optimized_value":
                optimized_case[
                    "average_heating_rate_C_per_s"
                ]
        },
        {
            "parameter": "time_to_80_C_s",
            "baseline_value":
                baseline_case["time_to_80_C_s"],
            "optimized_value":
                optimized_case["time_to_80_C_s"]
        },
        {
            "parameter": "performance_score",
            "baseline_value":
                baseline_score,
            "optimized_value":
                optimized_score
        },
        {
            "parameter": "score_improvement_percent",
            "baseline_value":
                0,
            "optimized_value":
                improvement
        }
    ]
)

result_df = pd.DataFrame(
    result_rows
)

result_path = os.path.join(
    RESULTS_DIR,
    "ga_optimization_result.csv"
)

result_df.to_csv(
    result_path,
    index=False
)


# ============================================================
# 24. CONVERGENCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(1, GENERATIONS + 1),
    convergence_history,
    linewidth=2
)

plt.xlabel(
    "Generation"
)

plt.ylabel(
    "Best Fitness Score"
)

plt.title(
    "Genetic Algorithm Convergence for TES Optimization"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

convergence_path = os.path.join(
    RESULTS_DIR,
    "ga_convergence.png"
)

plt.savefig(
    convergence_path,
    dpi=300
)

plt.close()


# ============================================================
# 25. BEFORE vs AFTER PERFORMANCE GRAPH
# ============================================================

metrics = [
    "Final Temperature",
    "Temperature Rise",
    "Heating Rate",
    "Performance Score"
]

baseline_values = [
    baseline_case["final_temperature_C"],
    baseline_case["temperature_rise_C"],
    baseline_case[
        "average_heating_rate_C_per_s"
    ],
    baseline_score
]

optimized_values = [
    optimized_case["final_temperature_C"],
    optimized_case["temperature_rise_C"],
    optimized_case[
        "average_heating_rate_C_per_s"
    ],
    optimized_score
]

# Normalize for visual comparison
baseline_plot = []
optimized_plot = []

for b, o in zip(
    baseline_values,
    optimized_values
):

    maximum = max(
        abs(b),
        abs(o)
    )

    if maximum == 0:

        baseline_plot.append(0)
        optimized_plot.append(0)

    else:

        baseline_plot.append(
            b / maximum
        )

        optimized_plot.append(
            o / maximum
        )


x = np.arange(
    len(metrics)
)

width = 0.35

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    baseline_plot,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    optimized_plot,
    width,
    label="Optimized"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel(
    "Normalized Value"
)

plt.title(
    "Baseline vs Optimized TES Performance"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

comparison_path = os.path.join(
    RESULTS_DIR,
    "ga_before_after_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300
)

plt.close()


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("GA OPTIMIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles generated:")

print(
    "1.",
    result_path
)

print(
    "2.",
    convergence_path
)

print(
    "3.",
    comparison_path
)

print("\nStep 14 completed.")