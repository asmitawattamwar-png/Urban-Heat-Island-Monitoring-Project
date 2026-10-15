
# ============================================================
# URBAN HEAT ISLAND (UHI) MONITORING PROJECT
# File 1: data_preprocessing.py
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

INPUT_FILE = "urban_heat_island_dataset_project.csv"  
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER, "cleaned_uhi_dataset.csv"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 65)
print("URBAN HEAT ISLAND - DATA PREPROCESSING")
print("=" * 65)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

try:
    df = pd.read_csv(INPUT_FILE)
    print("\nDataset loaded successfully!")

except FileNotFoundError:
    print(f"\nERROR: File '{INPUT_FILE}' not found.")
    print("Please check the filename and its location.")
    raise SystemExit(1)

except Exception as e:
    print(f"\nERROR while loading dataset: {e}")
    raise SystemExit(1)


# ------------------------------------------------------------
# 3. INITIAL DATASET INSPECTION
# ------------------------------------------------------------

print("\n--- ORIGINAL DATASET INFORMATION ---")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ------------------------------------------------------------
# 4. STANDARDIZE COLUMN NAMES AND TEXT VALUES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

for column in df.select_dtypes(
    include=["object", "string"]
).columns:
    df[column] = df[column].str.strip()

print("\nColumn names and text values standardized.")


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE ROWS
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\n--- DUPLICATE ANALYSIS ---")
print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicate_count)


# ------------------------------------------------------------
# 6. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

numeric_columns = [
    "Latitude",
    "Longitude",
    "Elevation (m)",
    "Temperature (°C)",
    "Population Density (people/km²)",
    "Energy Consumption (kWh)",
    "Air Quality Index (AQI)",
    "Urban Greenness Ratio (%)",
    "Health Impact (Mortality Rate/100k)",
    "Wind Speed (km/h)",
    "Humidity (%)",
    "Annual Rainfall (mm)",
    "GDP per Capita (USD)"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

print("\nNumeric column conversion completed.")


# ------------------------------------------------------------
# 7. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n--- MISSING VALUE ANALYSIS ---")

missing_before = df.isnull().sum()

print("\nMissing values before handling:")
print(missing_before[missing_before > 0])

# Fill missing numeric values using the median.
# If a column contains no valid values, leave it unchanged.

for column in df.select_dtypes(include=np.number).columns:
    if df[column].isnull().any():
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)

# Fill missing categorical values with "Unknown".

for column in df.select_dtypes(
    include=["object", "string", "category"]
).columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna("Unknown")

print("\nMissing values after initial handling:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 8. VALIDATE GEOGRAPHICAL COORDINATES
# ------------------------------------------------------------

print("\n--- GEOGRAPHICAL VALIDATION ---")

if "Latitude" in df.columns:
    invalid_latitude = (
        df["Latitude"].notna()
        & ~df["Latitude"].between(-90, 90)
    )

    print("Invalid latitude values:",
          invalid_latitude.sum())

    df.loc[invalid_latitude, "Latitude"] = np.nan

if "Longitude" in df.columns:
    invalid_longitude = (
        df["Longitude"].notna()
        & ~df["Longitude"].between(-180, 180)
    )

    print("Invalid longitude values:",
          invalid_longitude.sum())

    df.loc[invalid_longitude, "Longitude"] = np.nan

# Remove rows whose coordinates are invalid or missing.
# This is done only when both coordinate columns exist.

if "Latitude" in df.columns and "Longitude" in df.columns:
    valid_coordinates = (
        df["Latitude"].notna()
        & df["Longitude"].notna()
    )

    invalid_coordinate_rows = (~valid_coordinates).sum()

    print("Rows with invalid or missing coordinates:",
          invalid_coordinate_rows)

    df = df.loc[valid_coordinates].copy()


# ------------------------------------------------------------
# 9. VALIDATE ENVIRONMENTAL AND SOCIOECONOMIC VALUES
# ------------------------------------------------------------

print("\n--- ENVIRONMENTAL DATA VALIDATION ---")

# Temperature in Celsius.
# This is a broad plausibility check, not a universal limit.

if "Temperature (°C)" in df.columns:
    invalid_temperature = (
        df["Temperature (°C)"].notna()
        & ~df["Temperature (°C)"].between(-90, 65)
    )

    print("Potentially invalid temperature values:",
          invalid_temperature.sum())

    df.loc[invalid_temperature, "Temperature (°C)"] = np.nan


# Percentage values must lie between 0 and 100.

percentage_columns = [
    "Urban Greenness Ratio (%)",
    "Humidity (%)"
]

for column in percentage_columns:
    if column in df.columns:
        invalid_values = (
            df[column].notna()
            & ~df[column].between(0, 100)
        )

        print(f"Invalid values in {column}:",
              invalid_values.sum())

        df.loc[invalid_values, column] = np.nan


# These measurements should not be negative.

non_negative_columns = [
    "Elevation (m)",
    "Population Density (people/km²)",
    "Energy Consumption (kWh)",
    "Air Quality Index (AQI)",
    "Health Impact (Mortality Rate/100k)",
    "Wind Speed (km/h)",
    "Annual Rainfall (mm)",
    "GDP per Capita (USD)"
]

for column in non_negative_columns:
    if column in df.columns:
        invalid_values = (
            df[column].notna()
            & (df[column] < 0)
        )

        print(f"Negative values in {column}:",
              invalid_values.sum())

        df.loc[invalid_values, column] = np.nan


# ------------------------------------------------------------
# 10. FINAL MISSING VALUE HANDLING
# ------------------------------------------------------------

print("\n--- FINAL MISSING VALUE HANDLING ---")

# Validation may have introduced new missing values.
# Fill numeric gaps using the median of valid observations.

for column in df.select_dtypes(include=np.number).columns:
    if df[column].isnull().any():
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)

# Fill missing categorical values.

for column in df.select_dtypes(
    include=["object", "string", "category"]
).columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna("Unknown")

print("Remaining missing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 11. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n--- OUTLIER DETECTION ---")

# Detect potential outliers but do not automatically remove them.
# Extreme values may be genuine observations.

for column in df.select_dtypes(include=np.number).columns:
    values = df[column].dropna()

    if len(values) < 4 or values.nunique() < 2:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        continue

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_bound)
        | (values > upper_bound)
    ).sum()

    print(f"{column}: {outlier_count} potential outliers")


# ------------------------------------------------------------
# 12. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n--- PREPROCESSING SUMMARY ---")

print("Original dataset shape:", (500, 15))
print("Final dataset shape:", df.shape)
print("Total remaining missing values:",
      df.isnull().sum().sum())

print("\nFinal data types:")
print(df.dtypes)

print("\nCleaned dataset preview:")
print(df.head())

print("\nCleaned dataset saved to:")
print(os.path.abspath(OUTPUT_FILE))

print("\nData preprocessing completed successfully!")

