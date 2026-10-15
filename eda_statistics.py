
# ============================================================
# URBAN HEAT ISLAND (UHI) MONITORING PROJECT
# File 2: EDA and Statistical Analysis
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

INPUT_FILE = os.path.join(
    "outputs", "cleaned_uhi_dataset.csv"
)

OUTPUT_FOLDER = "outputs"
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "eda_plots")

os.makedirs(PLOT_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")


# ------------------------------------------------------------
# 2. LOAD CLEANED DATASET
# ------------------------------------------------------------

try:
    df = pd.read_csv(INPUT_FILE)
    print("Cleaned dataset loaded successfully!")

except FileNotFoundError:
    print("ERROR: Cleaned dataset not found.")
    print("Run data_preprocessing.py first.")
    raise SystemExit(1)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 3. IDENTIFY NUMERICAL COLUMNS
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)
numeric_columns = numeric_df.columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)


# ------------------------------------------------------------
# 4. GENERAL DATASET SUMMARY
# ------------------------------------------------------------

print("\n--- DATASET SUMMARY ---")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistical summary:")
print(numeric_df.describe())


# ------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n--- DESCRIPTIVE STATISTICS ---")

statistics = pd.DataFrame(index=numeric_columns)

statistics["Mean"] = numeric_df.mean()
statistics["Median"] = numeric_df.median()

# Use the first mode if multiple modes exist.
statistics["Mode"] = numeric_df.mode().iloc[0]

statistics["Standard Deviation"] = numeric_df.std()
statistics["Variance"] = numeric_df.var()
statistics["Minimum"] = numeric_df.min()
statistics["Maximum"] = numeric_df.max()

statistics["Range"] = (
    numeric_df.max() - numeric_df.min()
)

statistics["Q1 (25%)"] = numeric_df.quantile(0.25)
statistics["Q2 (50%)"] = numeric_df.quantile(0.50)
statistics["Q3 (75%)"] = numeric_df.quantile(0.75)

statistics["IQR"] = (
    statistics["Q3 (75%)"] - statistics["Q1 (25%)"]
)

statistics["Skewness"] = numeric_df.skew()
statistics["Kurtosis"] = numeric_df.kurtosis()

statistics = statistics.round(3)

print(statistics)

statistics.to_csv(
    os.path.join(
        OUTPUT_FOLDER, "descriptive_statistics.csv"
    )
)


# ------------------------------------------------------------
# 6. UNIVARIATE ANALYSIS - HISTOGRAMS
# ------------------------------------------------------------

print("\n--- HISTOGRAM ANALYSIS ---")

for column in numeric_columns:

    values = numeric_df[column].dropna()

    if values.empty:
        continue

    plt.figure(figsize=(8, 5))

    sns.histplot(
        values,
        bins=20,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = (
        column.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        + "_histogram.png"
    )

    plt.savefig(
        os.path.join(PLOT_FOLDER, filename),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 7. UNIVARIATE ANALYSIS - BOXPLOTS
# ------------------------------------------------------------

print("\n--- BOXPLOT ANALYSIS ---")

for column in numeric_columns:

    values = numeric_df[column].dropna()

    if values.empty:
        continue

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=values)

    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)
    plt.tight_layout()

    filename = (
        column.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        + "_boxplot.png"
    )

    plt.savefig(
        os.path.join(PLOT_FOLDER, filename),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 8. TEMPERATURE DISTRIBUTION
# ------------------------------------------------------------

temperature_column = "Temperature (°C)"

if temperature_column in numeric_df.columns:

    temperature = numeric_df[temperature_column].dropna()

    if not temperature.empty:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            temperature,
            bins=20,
            kde=True
        )

        plt.axvline(
            temperature.mean(),
            linestyle="--",
            label="Mean"
        )

        plt.axvline(
            temperature.median(),
            linestyle=":",
            label="Median"
        )

        plt.title("Distribution of City Temperatures")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOT_FOLDER,
                "temperature_distribution.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


# ------------------------------------------------------------
# 9. BIVARIATE ANALYSIS - TEMPERATURE RELATIONSHIPS
# ------------------------------------------------------------
if temperature_column in numeric_df.columns:

    # Calculate correlation of each numerical variable
    # with temperature
    correlations = (
        numeric_df.corr()[temperature_column]
        .drop(temperature_column)
        .dropna()
        .sort_values()
    )

    # Save correlation values as a CSV file
    correlations.to_frame(
        name="Correlation with Temperature"
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "temperature_correlations.csv")
    )

    # Create one horizontal bar chart
    if not correlations.empty:
        plt.figure(figsize=(10, 7))

        colors = [
            "steelblue" if value >= 0 else "indianred"
            for value in correlations.values
        ]

        plt.barh(
            correlations.index,
            correlations.values,
            color=colors
        )

        plt.axvline(
            0,
            color="black",
            linewidth=1
        )

        plt.xlim(-1, 1)
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Variables")
        plt.title("Correlation of City Variables with Temperature")
        plt.tight_layout()

        plt.savefig(
            os.path.join(PLOT_FOLDER, "temperature_correlation.png"),
            dpi=300
        )

        plt.close()

    print("\nCorrelation with Temperature:")
    print(correlations.round(3))

# ------------------------------------------------------------
# 10. MULTIVARIATE ANALYSIS - PAIR PLOT
# ------------------------------------------------------------

print("\n--- MULTIVARIATE ANALYSIS ---")

pairplot_columns = [
    "Temperature (°C)",
    "Urban Greenness Ratio (%)",
    "Population Density (people/km²)",
    "Energy Consumption (kWh)",
    "Air Quality Index (AQI)"
]

available_columns = [
    column for column in pairplot_columns
    if column in numeric_df.columns
]

if len(available_columns) >= 2:

    pairplot_data = numeric_df[available_columns].dropna()

    if len(pairplot_data) >= 2:

        if len(pairplot_data) > 300:
            pairplot_data = pairplot_data.sample(
                n=300, random_state=42
            )

        pair_plot = sns.pairplot(
            pairplot_data,
            diag_kind="hist",
            corner=True
        )

        pair_plot.fig.suptitle(
            "Pairwise Relationships Between UHI Variables",
            y=1.02
        )

        pair_plot.savefig(
            os.path.join(
                PLOT_FOLDER, "multivariate_pairplot.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close("all")

print("EDA AND STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")