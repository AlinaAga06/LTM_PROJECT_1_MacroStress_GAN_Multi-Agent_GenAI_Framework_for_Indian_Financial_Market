import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# Paths
# =========================================================

INPUT_FILE = Path("data/processed/macro_stress_5vars_processed.csv")

OUTPUT_DIR = Path("data/timegan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "timegan_features.csv"
SCALER_FILE = OUTPUT_DIR / "timegan_scaler.npy"

# =========================================================
# Configuration
# =========================================================

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

# =========================================================
# Load validated master dataset
# =========================================================

print("=" * 70)
print("TIMEGAN DATA PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("\nInput dataset shape:")
print(df.shape)

print("\nDate range:")
print(df["Date"].min().date(), "→", df["Date"].max().date())

# =========================================================
# Select TimeGAN features
# =========================================================

data = df[["Date"] + FEATURES].copy()

print("\nSelected features:")
print(FEATURES)

# =========================================================
# Check missing values
# =========================================================

print("\nMissing values:")
print(data.isnull().sum())

if data[FEATURES].isnull().any().any():
    raise ValueError("Missing values detected in TimeGAN features.")

# =========================================================
# Check infinite values
# =========================================================

if np.isinf(data[FEATURES].values).any():
    raise ValueError("Infinite values detected in TimeGAN features.")

# =========================================================
# Display feature statistics BEFORE scaling
# =========================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS BEFORE SCALING")
print("=" * 70)

print(data[FEATURES].describe())

# =========================================================
# Normalize features
# =========================================================

scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler.fit_transform(data[FEATURES])

scaled_df = pd.DataFrame(
    data_scaled,
    columns=FEATURES
)

scaled_df.insert(0, "Date", data["Date"].values)

# =========================================================
# Validate scaled data
# =========================================================

print("\n" + "=" * 70)
print("SCALING VALIDATION")
print("=" * 70)

print("\nMinimum values:")
print(scaled_df[FEATURES].min())

print("\nMaximum values:")
print(scaled_df[FEATURES].max())

print("\nMissing values after scaling:")
print(scaled_df.isnull().sum())

# =========================================================
# Save scaled dataset
# =========================================================

scaled_df.to_csv(OUTPUT_FILE, index=False)

# Save scaler parameters
np.save(
    SCALER_FILE,
    {
        "min": scaler.min_,
        "scale": scaler.scale_,
        "data_min": scaler.data_min_,
        "data_max": scaler.data_max_
    },
    allow_pickle=True
)

# =========================================================
# Final validation
# =========================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nOutput shape:")
print(scaled_df.shape)

print("\nOutput date range:")
print(
    scaled_df["Date"].min().date(),
    "→",
    scaled_df["Date"].max().date()
)

print("\nOutput columns:")
print(scaled_df.columns.tolist())

print("\nFirst 5 rows:")
print(scaled_df.head())

print("\nLast 5 rows:")
print(scaled_df.tail())

print("\nSaved TimeGAN features:")
print(OUTPUT_FILE)

print("\nSaved scaler:")
print(SCALER_FILE)

print("\n" + "=" * 70)
print("TIMEGAN DATA PREPARATION COMPLETED")
print("=" * 70)