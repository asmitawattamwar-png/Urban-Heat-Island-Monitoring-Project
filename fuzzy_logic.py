
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("Urban-Heat-Island-Monitoring-Project\\outputs\\cleaned_uhi_dataset.csv")

# 2. PREPARE INPUT VARIABLES
# Temperature is used as a proxy for LST.
lst = df["Temperature (°C)"].to_numpy()

# Higher greenness means more vegetation.
vegetation = df["Urban Greenness Ratio (%)"].to_numpy()

# Estimate built-up intensity from land-cover categories.
builtup_mapping = {
    "Water": 10,
    "Green Space": 20,
    "Urban": 80,
    "Industrial": 100
}

builtup = (
    df["Land Cover"].map(builtup_mapping)
    .fillna(50).to_numpy()
)

# 3. FUZZY MEMBERSHIP FUNCTIONS
def trimf(x, a, b, c):
    """Triangular membership function."""
    x = np.asarray(x, dtype=float)
    left = np.divide(
        x - a, b - a,
        out=np.zeros_like(x), where=(b != a)
    )
    right = np.divide(
        c - x, c - b,
        out=np.zeros_like(x), where=(c != b)
    )
    return np.clip(np.minimum(left, right), 0, 1)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    x = np.asarray(x, dtype=float)
    return np.maximum(
        np.minimum(
            np.minimum(
                np.divide(
                    x - a, b - a,
                    out=np.ones_like(x), where=(b != a)
                ),
                1
            ),
            np.divide(
                d - x, d - c,
                out=np.ones_like(x), where=(d != c)
            )
        ),
        0
    )


# LST / temperature memberships (degrees Celsius)
lst_low = trapmf(lst, 0, 0, 18, 25)
lst_moderate = trimf(lst, 18, 25, 32)
lst_high = trapmf(lst, 25, 32, 45, 45)

# Vegetation memberships (percentage)
veg_low = trapmf(vegetation, 0, 0, 20, 35)
veg_moderate = trimf(vegetation, 20, 40, 60)
veg_high = trapmf(vegetation, 45, 60, 100, 100)

# Built-up intensity memberships (0 to 100)
built_low = trapmf(builtup, 0, 0, 20, 45)
built_moderate = trimf(builtup, 20, 50, 80)
built_high = trapmf(builtup, 55, 80, 100, 100)


# 4. DEFINE OUTPUT MEMBERSHIP FUNCTIONS
risk_universe = np.linspace(0, 100, 501)

risk_low = trapmf(risk_universe, 0, 0, 20, 40)
risk_moderate = trimf(risk_universe, 25, 50, 75)
risk_high = trapmf(risk_universe, 60, 80, 100, 100)


# 5. FUZZY IF-THEN RULES
# Each rule has:
# (rule strength, output membership function)

rules = [
    # High temperature, low vegetation, high built-up
    (np.minimum.reduce([lst_high, veg_low, built_high]),
     risk_high),

    # High temperature and low vegetation
    (np.minimum(lst_high, veg_low), risk_high),

    # High temperature and high built-up
    (np.minimum(lst_high, built_high), risk_high),

    # Moderate temperature and high built-up
    (np.minimum(lst_moderate, built_high), risk_moderate),

    # Moderate temperature and low vegetation
    (np.minimum(lst_moderate, veg_low), risk_moderate),

    # High vegetation reduces relative heat risk
    (np.minimum(lst_high, veg_high), risk_moderate),

    # Low temperature generally indicates low risk
    (lst_low, risk_low),

    # Moderate temperature, high vegetation, low built-up
    (np.minimum.reduce([
        lst_moderate, veg_high, built_low
    ]), risk_low)
]


# 6. FUZZY INFERENCE AND DEFUZZIFICATION
scores = []

for i in range(len(df)):
    aggregated = np.zeros_like(risk_universe)

    for strength, output_mf in rules:
        # Clip output membership using rule strength.
        clipped = np.minimum(strength[i], output_mf)

        # Aggregate outputs using the maximum operator.
        aggregated = np.maximum(aggregated, clipped)

    # Centroid method converts fuzzy output to a crisp score.
    if aggregated.sum() == 0:
        score = 50
    else:
        score = np.sum(
            risk_universe * aggregated
        ) / np.sum(aggregated)

    scores.append(score)


# 7. ADD HEAT-RISK RESULTS
df["Built-up Intensity"] = builtup
df["Relative Heat Risk Score"] = np.round(scores, 2)

def categorize_risk(score):
    if score < 35:
        return "Low"
    elif score < 65:
        return "Moderate"
    else:
        return "High"

df["Heat Risk Category"] = (
    df["Relative Heat Risk Score"].apply(categorize_risk)
)


# 8. DISPLAY RESULTS
print("\nHeat-Risk Results:")
print(df[[
    "City Name",
    "Temperature (°C)",
    "Urban Greenness Ratio (%)",
    "Built-up Intensity",
    "Relative Heat Risk Score",
    "Heat Risk Category"
]].head(10).to_string(index=False))

print("\nRisk Category Counts:")
print(df["Heat Risk Category"].value_counts())

# Save the output dataset.
df.to_csv("fuzzy_heat_risk_results.csv", index=False)
print("\nResults saved to fuzzy_heat_risk_results.csv")


# 9. VISUALIZE HEAT-RISK CATEGORIES
colors = {
    "Low": "green",
    "Moderate": "orange",
    "High": "red"
}

plt.figure(figsize=(9, 5))

for category, color in colors.items():
    subset = df[df["Heat Risk Category"] == category]

    plt.scatter(
        subset["Urban Greenness Ratio (%)"],
        subset["Relative Heat Risk Score"],
        color=color,
        label=category,
        alpha=0.7
    )

plt.xlabel("Urban Greenness Ratio (%)")
plt.ylabel("Relative Heat Risk Score")
plt.title("Fuzzy Logic-Based Heat-Risk Categorization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 10. VISUALIZE SCORE DISTRIBUTION
plt.figure(figsize=(8, 5))
plt.hist(
    df["Relative Heat Risk Score"],
    bins=20,
    edgecolor="black"
)
plt.xlabel("Relative Heat Risk Score")
plt.ylabel("Number of Records")
plt.title("Distribution of Fuzzy Heat-Risk Scores")
plt.tight_layout()
plt.show()
