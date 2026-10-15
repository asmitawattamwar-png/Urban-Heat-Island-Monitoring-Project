
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("Urban-Heat-Island-Monitoring-Project\\outputs\\cleaned_uhi_dataset.csv")

np.random.seed(42)

# 2. PREPARE INPUT VARIABLES
temperature = df["Temperature (°C)"].to_numpy(dtype=float)
vegetation = df["Urban Greenness Ratio (%)"].to_numpy(dtype=float)

builtup_mapping = {
    "Water": 10,
    "Green Space": 20,
    "Urban": 80,
    "Industrial": 100
}

builtup = (
    df["Land Cover"].map(builtup_mapping)
    .fillna(50).to_numpy(dtype=float)
)

def normalize(x):
    minimum = np.min(x)
    maximum = np.max(x)

    if maximum == minimum:
        return np.zeros_like(x)

    return (x - minimum) / (maximum - minimum)

# Normalize all inputs to the range 0 to 1
temp = normalize(temperature)
veg = normalize(vegetation)
built = normalize(builtup)


# 3. FUZZY MEMBERSHIP FUNCTIONS
def low_membership(x):
    return np.clip(1 - 2 * x, 0, 1)

def moderate_membership(x):
    return np.clip(1 - 2 * np.abs(x - 0.5), 0, 1)

def high_membership(x):
    return np.clip(2 * x - 1, 0, 1)


# Convert memberships into risk indicators
temp_risk = (
    high_membership(temp)
    + 0.5 * moderate_membership(temp)
)

vegetation_risk = (
    low_membership(veg)
    + 0.5 * moderate_membership(veg)
)

builtup_risk = (
    high_membership(built)
    + 0.5 * moderate_membership(built)
)

# Three input risk indicators
X = np.column_stack([
    temp_risk,
    vegetation_risk,
    builtup_risk
])


# 4. DEFINE REFERENCE TARGET
# Project-defined benchmark, NOT actual ground-truth heat risk.
# Higher weight is assigned to temperature.

reference_weights = np.array([0.50, 0.30, 0.20])

reference_score = 100 * (X @ reference_weights)


# 5. INITIAL UNOPTIMIZED MODEL
# Give equal weight to all three input variables.
initial_weights = np.array([1/3, 1/3, 1/3])

initial_score = 100 * (X @ initial_weights)


# 6. PARTICLE SWARM OPTIMIZATION (PSO)
particles = 40
iterations = 150
dimensions = 3

position = np.random.dirichlet(
    np.ones(dimensions), size=particles
)

velocity = np.zeros_like(position)

personal_best = position.copy()
personal_best_error = np.full(particles, np.inf)

global_best = position[0].copy()
global_best_error = np.inf

inertia = 0.7
cognitive = 1.5
social = 1.5

for iteration in range(iterations):

    # Evaluate each particle
    for i in range(particles):

        predicted = 100 * (X @ position[i])

        error = np.mean(
            (reference_score - predicted) ** 2
        )

        if error < personal_best_error[i]:
            personal_best_error[i] = error
            personal_best[i] = position[i].copy()

        if error < global_best_error:
            global_best_error = error
            global_best = position[i].copy()

    # Update particle velocities and positions
    r1 = np.random.rand(particles, dimensions)
    r2 = np.random.rand(particles, dimensions)

    velocity = (
        inertia * velocity
        + cognitive * r1 * (personal_best - position)
        + social * r2 * (global_best - position)
    )

    position = position + velocity

    # Keep weights non-negative and normalize their sum to 1
    position = np.clip(position, 0.001, None)
    position = position / position.sum(axis=1, keepdims=True)


# 7. CALCULATE OPTIMIZED SCORES
optimized_weights = global_best

optimized_score = 100 * (X @ optimized_weights)


# 8. COMPARE MODEL ERRORS
initial_mse = np.mean(
    (reference_score - initial_score) ** 2
)

optimized_mse = np.mean(
    (reference_score - optimized_score) ** 2
)

print("\nOPTIMIZED PSO WEIGHTS")
print(f"Temperature weight: {optimized_weights[0]:.4f}")
print(f"Vegetation weight:  {optimized_weights[1]:.4f}")
print(f"Built-up weight:    {optimized_weights[2]:.4f}")

print("\nMODEL COMPARISON")
print(f"Initial MSE:   {initial_mse:.6f}")
print(f"Optimized MSE: {optimized_mse:.6f}")

if initial_mse > 0:
    improvement = (
        (initial_mse - optimized_mse) / initial_mse
    ) * 100
    print(f"Error reduction: {improvement:.2f}%")
else:
    print("Initial model already has zero error.")


# 9. CLASSIFY HEAT RISK
def categorize(score):
    if score < 35:
        return "Low"
    elif score < 65:
        return "Moderate"
    else:
        return "High"


df["Initial Heat Risk"] = initial_score
df["PSO Optimized Heat Risk"] = optimized_score
df["Reference Score"] = reference_score

df["Initial Category"] = [
    categorize(s) for s in initial_score
]

df["PSO Optimized Category"] = [
    categorize(s) for s in optimized_score
]


# 10. SAVE RESULTS
df.to_csv("pso_heat_risk_results.csv", index=False)

print("\nFirst 10 results:")
print(df[[
    "City Name",
    "Initial Heat Risk",
    "PSO Optimized Heat Risk",
    "Initial Category",
    "PSO Optimized Category"
]].head(10).round(2).to_string(index=False))

print("\nResults saved to pso_heat_risk_results.csv")


# 11. VISUALIZE MODEL COMPARISON
plt.figure(figsize=(8, 5))

plt.bar(
    ["Initial Model", "PSO Optimized"],
    [initial_mse, optimized_mse]
)

plt.ylabel("Mean Squared Error (MSE)")
plt.title("Initial Model vs PSO Optimized Model")
plt.tight_layout()
plt.show()


# 12. COMPARE HEAT-RISK SCORES
plt.figure(figsize=(9, 5))

plt.plot(
    reference_score[:50],
    label="Reference Score",
    marker="o"
)

plt.plot(
    initial_score[:50],
    label="Initial Model",
    marker="x"
)

plt.plot(
    optimized_score[:50],
    label="PSO Optimized Model",
    marker="s"
)

plt.xlabel("Record Index")
plt.ylabel("Relative Heat-Risk Score")
plt.title("Heat-Risk Score Comparison")
plt.legend()
plt.tight_layout()
plt.show()
