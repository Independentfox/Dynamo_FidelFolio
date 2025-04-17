# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Optional: For visualization and debugging (commented out)
# import matplotlib.pyplot as plt
# import seaborn as sns

# ------------------------------
# 1. Load and Inspect Dataset
# ------------------------------

# Load dataset
df = pd.read_csv('Dataset/FidelFolio_Dataset.csv')

# Define feature and target columns
features = [f'Feature{i}' for i in range(1, 29)]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']

# Columns originally stored as object with commas
cols_to_convert = [f"Feature{i}" for i in [4, 5, 6, 7, 9]] + targets

# Convert string with commas to float
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort by company and year for consistent time series processing
df = df.sort_values(by=["Company", "Year"])

# Ensure target columns are numeric
df[targets] = df[targets].apply(pd.to_numeric, errors='coerce')

# ------------------------------
# 2. Fill Missing Values
# ------------------------------

# Fill missing values in targets with company-wise mean, fallback to global mean
for target in targets:
    company_mean = df.groupby('Company')[target].transform(lambda x: x.fillna(x.mean()))
    df[target] = company_mean.fillna(df[target].mean())

# Fill missing values in features similarly
for feature in features:
    company_mean = df.groupby('Company')[feature].transform(lambda x: x.fillna(x.mean()))
    df[feature] = company_mean.fillna(df[feature].mean())

# ------------------------------
# 3. Handle Outliers (Winsorization)
# ------------------------------

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df[features] = df[features].apply(cap_outliers)

# ------------------------------
# 4. Normalize Features
# ------------------------------

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ------------------------------
# 5. Save Cleaned Dataset
# ------------------------------

df.to_csv('Dataset/FidelFolio_Dataset_Cleaned.csv', index=False)
print("Cleaned dataset saved to Dataset/FidelFolio_Dataset_Cleaned.csv")
