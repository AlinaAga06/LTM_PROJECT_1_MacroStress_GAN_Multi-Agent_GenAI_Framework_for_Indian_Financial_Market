"""
MacroStress-GAN
V3.1 TimeGAN Data Preparation

Purpose:
    Prepare financial return/change features for TimeGAN using
    QuantileTransformer instead of MinMaxScaler.

Why QuantileTransformer:
    The corrected crude-oil return contains -2 and +2 observations
    caused by the 2020 negative WTI price event.

    MinMaxScaler compresses the majority of normal observations
    into a very small region around 0.5 because -2 and +2 define
    the global range.

    QuantileTransformer spreads the empirical distribution across
    [0, 1], allowing TimeGAN to learn the distribution more effectively.

Features:
    NIFTY50_Return
    CRUDE_OIL_Return
    USD_INR_Return
    INDIA_VIX_Change
    INDIA_10Y_YIELD_Change
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_stress_5vars_processed.csv"
)

OUTPUT_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_sequences.npy"
)

OUTPUT_SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_scaler.pkl"
)

OUTPUT_TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_training_data.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30

# QuantileTransformer configuration
N_QUANTILES = 1000
RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MacroStress-GAN - V3.1 Data Preparation")
print("QuantileTransformer Financial Return Scaling")
print("=" * 70)

print()
print(f"Project root       : {PROJECT_ROOT}")
print(f"Input dataset      : {INPUT_PATH}")
print(f"Sequence output    : {OUTPUT_SEQUENCE_PATH}")
print(f"Scaler output      : {OUTPUT_SCALER_PATH}")
print(f"Training data      : {OUTPUT_TRAINING_DATA_PATH}")

print()
print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print()
print(f"Sequence length    : {SEQUENCE_LENGTH}")
print(f"Quantiles          : {N_QUANTILES}")
print(f"Random state       : {RANDOM_STATE}")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading processed financial dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Required features are missing:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )


# ============================================================
# DATE HANDLING
# ============================================================

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

missing_before = df[FEATURES].isna().sum().sum()

print()
print(
    f"Missing feature values before cleaning: "
    f"{missing_before}"
)

if missing_before > 0:

    df[FEATURES] = (
        df[FEATURES]
        .ffill()
        .bfill()
    )


# Remove any remaining rows containing NaN
df = df.dropna(
    subset=FEATURES
).reset_index(drop=True)


missing_after = df[FEATURES].isna().sum().sum()

print(
    f"Missing feature values after cleaning : "
    f"{missing_after}"
)


# ============================================================
# DISPLAY RAW FEATURE STATISTICS
# ============================================================

print()
print("-" * 70)
print("RAW FEATURE STATISTICS")
print("-" * 70)

for feature in FEATURES:

    values = df[feature].values

    print()
    print(feature)
    print(
        f"  Mean : {values.mean():.8f}"
    )
    print(
        f"  Std  : {values.std():.8f}"
    )
    print(
        f"  Min  : {values.min():.8f}"
    )
    print(
        f"  Max  : {values.max():.8f}"
    )


# ============================================================
# EXTRACT FEATURE MATRIX
# ============================================================

feature_data = df[
    FEATURES
].values.astype(
    np.float32
)

print()
print(
    f"Feature matrix shape: "
    f"{feature_data.shape}"
)


# ============================================================
# QUANTILE TRANSFORMER
# ============================================================

print()
print("-" * 70)
print("FITTING QUANTILE TRANSFORMER")
print("-" * 70)

n_quantiles_actual = min(
    N_QUANTILES,
    len(feature_data)
)

scaler = QuantileTransformer(
    n_quantiles=n_quantiles_actual,
    output_distribution="uniform",
    random_state=RANDOM_STATE
)


# ============================================================
# FIT + TRANSFORM
# ============================================================

scaled_data = scaler.fit_transform(
    feature_data
).astype(
    np.float32
)


# ============================================================
# SCALING VALIDATION
# ============================================================

print()
print(
    f"Actual number of quantiles: "
    f"{n_quantiles_actual}"
)

print(
    f"Scaled data shape: "
    f"{scaled_data.shape}"
)

print(
    f"Global scaled minimum: "
    f"{scaled_data.min():.8f}"
)

print(
    f"Global scaled maximum: "
    f"{scaled_data.max():.8f}"
)


# ============================================================
# FEATURE-WISE SCALED STATISTICS
# ============================================================

print()
print("-" * 70)
print("SCALED FEATURE STATISTICS")
print("-" * 70)

for i, feature in enumerate(FEATURES):

    values = scaled_data[:, i]

    print()
    print(feature)

    print(
        f"  Mean : {values.mean():.8f}"
    )

    print(
        f"  Std  : {values.std():.8f}"
    )

    print(
        f"  Min  : {values.min():.8f}"
    )

    print(
        f"  Max  : {values.max():.8f}"
    )

    print(
        "  Quantiles:"
    )

    q = np.quantile(
        values,
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    print(
        f"    1%  = {q[0]:.6f}"
    )

    print(
        f"    5%  = {q[1]:.6f}"
    )

    print(
        f"    25% = {q[2]:.6f}"
    )

    print(
        f"    50% = {q[3]:.6f}"
    )

    print(
        f"    75% = {q[4]:.6f}"
    )

    print(
        f"    95% = {q[5]:.6f}"
    )

    print(
        f"    99% = {q[6]:.6f}"
    )


# ============================================================
# CRUDE-OIL SPECIFIC DIAGNOSTIC
# ============================================================

crude_index = FEATURES.index(
    "CRUDE_OIL_Return"
)

crude_scaled = scaled_data[
    :, crude_index
]

print()
print("-" * 70)
print("CRUDE OIL SCALING DIAGNOSTIC")
print("-" * 70)

print(
    f"Crude scaled mean : "
    f"{crude_scaled.mean():.8f}"
)

print(
    f"Crude scaled std  : "
    f"{crude_scaled.std():.8f}"
)

print(
    f"Crude scaled min  : "
    f"{crude_scaled.min():.8f}"
)

print(
    f"Crude scaled max  : "
    f"{crude_scaled.max():.8f}"
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print()
print("-" * 70)
print("CREATING TIMEGAN SEQUENCES")
print("-" * 70)

sequences = []

for start_idx in range(
    len(scaled_data) - SEQUENCE_LENGTH + 1
):

    end_idx = (
        start_idx
        + SEQUENCE_LENGTH
    )

    sequence = scaled_data[
        start_idx:end_idx
    ]

    sequences.append(
        sequence
    )


sequences = np.asarray(
    sequences,
    dtype=np.float32
)


# ============================================================
# SEQUENCE VALIDATION
# ============================================================

print()
print(
    f"Sequence shape: "
    f"{sequences.shape}"
)

if sequences.ndim != 3:

    raise ValueError(
        f"Expected 3D sequence array, "
        f"got shape {sequences.shape}"
    )


expected_features = len(FEATURES)

if sequences.shape[1] != SEQUENCE_LENGTH:

    raise ValueError(
        f"Expected sequence length "
        f"{SEQUENCE_LENGTH}, "
        f"got {sequences.shape[1]}"
    )


if sequences.shape[2] != expected_features:

    raise ValueError(
        f"Expected {expected_features} features, "
        f"got {sequences.shape[2]}"
    )


nan_count = np.isnan(
    sequences
).sum()

inf_count = np.isinf(
    sequences
).sum()


print(
    f"NaN count: {nan_count}"
)

print(
    f"Inf count: {inf_count}"
)

if nan_count > 0 or inf_count > 0:

    raise ValueError(
        "Sequences contain NaN or Inf values."
    )


if sequences.min() < 0:

    raise ValueError(
        "Scaled sequences contain values below 0."
    )


if sequences.max() > 1:

    raise ValueError(
        "Scaled sequences contain values above 1."
    )


# ============================================================
# SAVE SEQUENCES
# ============================================================

OUTPUT_SEQUENCE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_SEQUENCE_PATH,
    sequences
)


# ============================================================
# SAVE QUANTILE TRANSFORMER
# ============================================================

joblib.dump(
    scaler,
    OUTPUT_SCALER_PATH
)


# ============================================================
# SAVE TRAINING DATA
# ============================================================

training_columns = [
    "Date"
] + FEATURES

training_df = df[
    training_columns
].copy()


training_df.to_csv(
    OUTPUT_TRAINING_DATA_PATH,
    index=False
)


# ============================================================
# INVERSE TRANSFORM TEST
# ============================================================

print()
print("-" * 70)
print("INVERSE TRANSFORM TEST")
print("-" * 70)

sample_scaled = scaled_data[
    :5
]

sample_restored = scaler.inverse_transform(
    sample_scaled
)


for i, feature in enumerate(FEATURES):

    original_values = feature_data[
        :5,
        i
    ]

    restored_values = sample_restored[
        :5,
        i
    ]

    max_error = np.max(
        np.abs(
            original_values
            - restored_values
        )
    )

    print(
        f"{feature}: "
        f"max reconstruction error = "
        f"{max_error:.12f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V3.1 DATA PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Original dataset      : "
    f"{len(df)} rows"
)

print(
    f"Features              : "
    f"{len(FEATURES)}"
)

print(
    f"Sequence length       : "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Training sequences    : "
    f"{len(sequences)}"
)

print(
    f"Sequence shape        : "
    f"{sequences.shape}"
)

print(
    f"Scaled minimum        : "
    f"{sequences.min():.8f}"
)

print(
    f"Scaled maximum        : "
    f"{sequences.max():.8f}"
)

print()
print(
    "Scaler: QuantileTransformer"
)

print(
    f"Quantiles used        : "
    f"{n_quantiles_actual}"
)

print()
print(
    f"Saved sequences       : "
    f"{OUTPUT_SEQUENCE_PATH}"
)

print(
    f"Saved scaler          : "
    f"{OUTPUT_SCALER_PATH}"
)

print(
    f"Saved training data   : "
    f"{OUTPUT_TRAINING_DATA_PATH}"
)

print()
print("=" * 70)