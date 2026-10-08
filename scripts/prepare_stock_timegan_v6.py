"""
MacroStress-GAN
Stock Market TimeGAN V6
Preprocessing

V6 changes:
- StandardScaler instead of QuantileTransformer
- Standardized feature representation
- Separate V6 files
- V1-V5 are NOT modified
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "stock_5vars.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "v6"
)

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

SEQ_LEN = 30
RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V6 PREPROCESSING")
print("=" * 78)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RAW STOCK DATA
# ============================================================

print("\n" + "=" * 78)
print("1. LOADING STOCK DATA")
print("=" * 78)

print(f"\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"\nRaw shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# DATE HANDLING
# ============================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# CREATE TIMEGAN FEATURES
# ============================================================

print("\n" + "=" * 78)
print("2. CREATING TIMEGAN FEATURES")
print("=" * 78)

required_columns = [
    "NIFTY50",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_VIX",
    "INDIA_10Y_YIELD",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


feature_df = pd.DataFrame()

if "Date" in df.columns:
    feature_df["Date"] = df["Date"]


feature_df["NIFTY50_Return"] = df["NIFTY50"].pct_change()
feature_df["CRUDE_OIL_Return"] = df["CRUDE_OIL"].pct_change()
feature_df["USD_INR_Return"] = df["USD_INR"].pct_change()

feature_df["INDIA_VIX_Change"] = df["INDIA_VIX"].diff()

feature_df["INDIA_10Y_YIELD_Change"] = (
    df["INDIA_10Y_YIELD"].diff()
)


# ============================================================
# CLEAN DATA
# ============================================================

feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_df = feature_df.dropna().reset_index(drop=True)

print(f"\nFeature dataframe shape: {feature_df.shape}")

print("\nMissing values:")
print(feature_df[FEATURES].isna().sum())

print("\nInfinite values:")
print(
    np.isinf(
        feature_df[FEATURES].to_numpy()
    ).sum()
)


# ============================================================
# SAVE REAL FEATURE DATA
# ============================================================

feature_file = OUTPUT_DIR / "stock_v6_features.csv"

feature_df.to_csv(
    feature_file,
    index=False
)

print(f"\nSaved:")
print(feature_file)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 78)
print("3. REAL FEATURE STATISTICS")
print("=" * 78)

for feature in FEATURES:

    values = feature_df[feature].to_numpy(
        dtype=np.float32
    )

    print(f"\n{feature}")
    print(f"  Mean : {values.mean():.8f}")
    print(f"  Std  : {values.std():.8f}")
    print(f"  Min  : {values.min():.8f}")
    print(f"  Max  : {values.max():.8f}")


# ============================================================
# STANDARD SCALER
# ============================================================

print("\n" + "=" * 78)
print("4. FITTING STANDARD SCALER")
print("=" * 78)

X = feature_df[FEATURES].to_numpy(
    dtype=np.float32
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled.astype(np.float32)


print("\nScaled statistics:")

for i, feature in enumerate(FEATURES):

    values = X_scaled[:, i]

    print(f"\n{feature}")
    print(f"  Mean : {values.mean():.8f}")
    print(f"  Std  : {values.std():.8f}")
    print(f"  Min  : {values.min():.8f}")
    print(f"  Max  : {values.max():.8f}")


# ============================================================
# VALIDATE SCALED DATA
# ============================================================

if np.isnan(X_scaled).any():
    raise ValueError(
        "NaN detected after StandardScaler."
    )

if np.isinf(X_scaled).any():
    raise ValueError(
        "Inf detected after StandardScaler."
    )


# ============================================================
# SAVE SCALER
# ============================================================

scaler_file = OUTPUT_DIR / "stock_v6_standard_scaler.pkl"

joblib.dump(
    scaler,
    scaler_file
)

print(f"\nScaler saved:")
print(scaler_file)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print("\n" + "=" * 78)
print("5. CREATING 30-DAY SEQUENCES")
print("=" * 78)

num_samples = len(X_scaled) - SEQ_LEN + 1

if num_samples <= 0:
    raise ValueError(
        "Not enough observations to create sequences."
    )


sequences = np.zeros(
    (
        num_samples,
        SEQ_LEN,
        len(FEATURES)
    ),
    dtype=np.float32
)


for i in range(num_samples):

    sequences[i] = X_scaled[
        i:i + SEQ_LEN
    ]


print(f"\nSequence shape: {sequences.shape}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Feature dimension: {len(FEATURES)}")

print(f"\nSequence minimum: {sequences.min():.8f}")
print(f"Sequence maximum: {sequences.max():.8f}")


# ============================================================
# SAVE SEQUENCES
# ============================================================

sequence_file = (
    OUTPUT_DIR
    / "stock_v6_timegan_sequences.npy"
)

np.save(
    sequence_file,
    sequences
)

print(f"\nSequences saved:")
print(sequence_file)


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

metadata = pd.DataFrame(
    {
        "feature_index": range(len(FEATURES)),
        "feature_name": FEATURES,
        "scaler": ["StandardScaler"] * len(FEATURES),
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }
)

metadata_file = (
    OUTPUT_DIR
    / "stock_v6_feature_metadata.csv"
)

metadata.to_csv(
    metadata_file,
    index=False
)

print(f"\nMetadata saved:")
print(metadata_file)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_file = (
    OUTPUT_DIR
    / "stock_v6_config.txt"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MacroStress-GAN Stock Market TimeGAN V6\n"
    )

    f.write(
        "=========================================\n"
    )

    f.write(
        f"SEQ_LEN={SEQ_LEN}\n"
    )

    f.write(
        f"FEATURE_DIM={len(FEATURES)}\n"
    )

    f.write(
        "SCALER=StandardScaler\n"
    )

    f.write(
        f"RANDOM_STATE={RANDOM_STATE}\n"
    )

    f.write(
        "DATA_TYPE=REAL\n"
    )

    f.write(
        "INSTITUTION=Stock Market\n"
    )

    f.write(
        "VERSION=TimeGAN_Stock_V6\n"
    )

    f.write(
        "\nFEATURES:\n"
    )

    for i, feature in enumerate(FEATURES):
        f.write(
            f"{i}: {feature}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("V6 PREPROCESSING COMPLETED")
print("=" * 78)

print("\nReal observations:")
print(len(feature_df))

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nSequence shape:")
print(sequences.shape)

print("\nScaler:")
print("  StandardScaler")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCreated files:")
print(f"  1. {feature_file.name}")
print(f"  2. {scaler_file.name}")
print(f"  3. {sequence_file.name}")
print(f"  4. {metadata_file.name}")
print(f"  5. {config_file.name}")

print("\nV1-V5 files were not modified.")

print("\n" + "=" * 78)