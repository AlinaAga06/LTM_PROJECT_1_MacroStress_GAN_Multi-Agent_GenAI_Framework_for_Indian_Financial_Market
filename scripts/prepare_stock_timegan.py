from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import QuantileTransformer


# ============================================================
# PATHS
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
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TIMEGAN CONFIGURATION
# ============================================================

SEQ_LEN = 30

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STOCK MARKET TIMEGAN PREPROCESSING")
print("=" * 70)

print(f"\nInput:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print(f"\nRaw rows: {len(df):,}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")


# ============================================================
# CREATE TIME-SERIES FEATURES
# ============================================================

print("\nCreating TimeGAN features...")


# Price variables → percentage returns
df["NIFTY50_Return"] = (
    df["NIFTY50"]
    .pct_change()
)

df["CRUDE_OIL_Return"] = (
    df["CRUDE_OIL"]
    .pct_change()
)

df["USD_INR_Return"] = (
    df["USD_INR"]
    .pct_change()
)


# Volatility → daily change
df["INDIA_VIX_Change"] = (
    df["INDIA_VIX"]
    .diff()
)


# Government bond yield → daily change
df["INDIA_10Y_YIELD_Change"] = (
    df["INDIA_10Y_YIELD"]
    .diff()
)


# ============================================================
# REMOVE INITIAL NaN
# ============================================================

df = df.dropna(
    subset=FEATURE_NAMES
).reset_index(drop=True)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURE_NAMES
).reset_index(drop=True)


print(f"\nRows after feature creation: {len(df):,}")


# ============================================================
# DISPLAY FEATURE STATISTICS
# ============================================================

print("\nFeature statistics:")
print(
    df[FEATURE_NAMES]
    .describe()
    .round(6)
    .to_string()
)


# ============================================================
# SAVE TRANSFORMED DATA
# ============================================================

transformed_file = (
    OUTPUT_DIR
    / "stock_timegan_features.csv"
)

df[
    ["Date"] + FEATURE_NAMES
].to_csv(
    transformed_file,
    index=False
)

print(
    f"\nSaved transformed features:\n"
    f"{transformed_file}"
)


# ============================================================
# EXTRACT NUMPY ARRAY
# ============================================================

X = df[
    FEATURE_NAMES
].values.astype(np.float32)


# ============================================================
# QUANTILE TRANSFORM
# ============================================================

print("\nFitting QuantileTransformer...")

n_quantiles = min(
    1000,
    len(X)
)

scaler = QuantileTransformer(
    n_quantiles=n_quantiles,
    output_distribution="uniform",
    random_state=42
)

X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled.astype(
    np.float32
)


# ============================================================
# SAVE SCALER
# ============================================================

scaler_file = (
    OUTPUT_DIR
    / "stock_timegan_scaler.pkl"
)

with open(
    scaler_file,
    "wb"
) as f:
    pickle.dump(
        scaler,
        f
    )

print(
    f"Scaler saved:\n"
    f"{scaler_file}"
)


# ============================================================
# CREATE 30-DAY SEQUENCES
# ============================================================

print(
    f"\nCreating {SEQ_LEN}-day sequences..."
)

sequences = []

sequence_dates = []

for i in range(
    len(X_scaled) - SEQ_LEN + 1
):

    sequence = X_scaled[
        i:i + SEQ_LEN
    ]

    sequences.append(
        sequence
    )

    sequence_dates.append(
        df["Date"].iloc[
            i:i + SEQ_LEN
        ].iloc[-1]
    )


sequences = np.asarray(
    sequences,
    dtype=np.float32
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TIMEGAN DATA VALIDATION")
print("=" * 70)

print(
    f"Sequences      : {len(sequences):,}"
)

print(
    f"Sequence shape : {sequences.shape}"
)

print(
    f"Expected shape : "
    f"(N, {SEQ_LEN}, {len(FEATURE_NAMES)})"
)

print(
    f"Min scaled     : {sequences.min():.6f}"
)

print(
    f"Max scaled     : {sequences.max():.6f}"
)

print(
    f"NaN count      : {np.isnan(sequences).sum()}"
)

print(
    f"Inf count      : {np.isinf(sequences).sum()}"
)


# ============================================================
# SAVE SEQUENCES
# ============================================================

sequence_file = (
    OUTPUT_DIR
    / "stock_timegan_sequences.npy"
)

np.save(
    sequence_file,
    sequences
)

print(
    f"\nSequences saved:\n"
    f"{sequence_file}"
)


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

metadata = pd.DataFrame(
    {
        "feature_index": range(
            len(FEATURE_NAMES)
        ),
        "feature_name": FEATURE_NAMES,
        "transformation": [
            "percentage_return",
            "percentage_return",
            "percentage_return",
            "daily_difference",
            "daily_difference",
        ],
    }
)

metadata_file = (
    OUTPUT_DIR
    / "stock_timegan_feature_metadata.csv"
)

metadata.to_csv(
    metadata_file,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_file = (
    OUTPUT_DIR
    / "stock_timegan_config.txt"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Stock Market TimeGAN Configuration\n"
    )

    f.write(
        "==================================\n\n"
    )

    f.write(
        f"SEQ_LEN={SEQ_LEN}\n"
    )

    f.write(
        f"FEATURE_DIM={len(FEATURE_NAMES)}\n"
    )

    f.write(
        "MODEL_VERSION=TimeGAN_Stock_v1\n\n"
    )

    f.write(
        "FEATURES:\n"
    )

    for i, feature in enumerate(
        FEATURE_NAMES
    ):
        f.write(
            f"{i}: {feature}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STOCK TIMEGAN PREPROCESSING COMPLETE")
print("=" * 70)

print("\nCreated files:")

print(
    f"1. {transformed_file}"
)

print(
    f"2. {scaler_file}"
)

print(
    f"3. {sequence_file}"
)

print(
    f"4. {metadata_file}"
)

print(
    f"5. {config_file}"
)

print("\nTimeGAN input:")
print(
    f"Shape = {sequences.shape}"
)

print(
    "\nREADY FOR STOCK TIMEGAN TRAINING"
)