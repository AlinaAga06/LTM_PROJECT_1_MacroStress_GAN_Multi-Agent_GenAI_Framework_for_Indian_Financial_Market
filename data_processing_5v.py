import pandas as pd
from pathlib import Path

# =========================================================
# Paths
# =========================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Function to clean Yahoo Finance files
# =========================================================

def load_yahoo_file(filename, column_name):

    path = RAW_DIR / filename

    df = pd.read_csv(path)

    # Remove Yahoo metadata rows
    df = df[
        ~df["Price"].astype(str).isin(["Ticker", "Date"])
    ].copy()

    # Rename columns
    df.rename(columns={"Price": "Date"}, inplace=True)

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert Close to numeric
    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # Keep required columns
    df = df[["Date", "Close"]].copy()

    # Rename Close
    df.rename(
        columns={"Close": column_name},
        inplace=True
    )

    # Remove invalid rows
    df.dropna(
        subset=["Date", column_name],
        inplace=True
    )

    # Remove duplicate dates
    df.drop_duplicates(
        subset=["Date"],
        keep="first",
        inplace=True
    )

    # Sort chronologically
    df.sort_values("Date", inplace=True)

    return df


# =========================================================
# Zero-crossing-safe symmetric percentage change
# =========================================================
#
# Ordinary percentage change:
#
#     (P_t - P_(t-1)) / P_(t-1)
#
# becomes problematic when a price crosses zero.
#
# This occurred in the WTI crude-oil market in April 2020.
#
# Symmetric percentage change:
#
#     (P_t - P_(t-1))
#     ------------------------------
#     (|P_t| + |P_(t-1)|) / 2
#
# This avoids division by a negative/near-zero previous
# price while preserving the direction and magnitude of
# the price movement.
#
# IMPORTANT:
# This transformation is used ONLY for CRUDE_OIL.
# NIFTY50, USD_INR and INDIA_VIX retain their original
# percentage-change calculations.
# =========================================================

def symmetric_change(series):

    denominator = (
        series.abs() +
        series.shift(1).abs()
    ) / 2

    return (
        series - series.shift(1)
    ) / denominator


# =========================================================
# Load Yahoo Finance datasets
# =========================================================

print("=" * 70)
print("LOADING MARKET DATA")
print("=" * 70)

nifty = load_yahoo_file(
    "NIFTY50.csv",
    "NIFTY50"
)

vix = load_yahoo_file(
    "INDIA_VIX.csv",
    "INDIA_VIX"
)

crude = load_yahoo_file(
    "CRUDE_OIL.csv",
    "CRUDE_OIL"
)

usd = load_yahoo_file(
    "USD_INR.csv",
    "USD_INR"
)

print("NIFTY50:", nifty.shape)
print("INDIA_VIX:", vix.shape)
print("CRUDE_OIL:", crude.shape)
print("USD_INR:", usd.shape)


# =========================================================
# Load India 10-Year Government Bond Yield
# =========================================================

print("\n" + "=" * 70)
print("LOADING INDIA 10-YEAR GOVERNMENT BOND YIELD")
print("=" * 70)

bond_path = RAW_DIR / "INDIA_10Y_YIELD.csv"

bond = pd.read_csv(bond_path)

bond["Date"] = pd.to_datetime(
    bond["Date"],
    errors="coerce"
)

bond["INDIA_10Y_YIELD"] = pd.to_numeric(
    bond["INDIA_10Y_YIELD"],
    errors="coerce"
)

bond = bond[
    ["Date", "INDIA_10Y_YIELD"]
].copy()

bond.dropna(
    subset=["Date", "INDIA_10Y_YIELD"],
    inplace=True
)

bond.drop_duplicates(
    subset=["Date"],
    keep="first",
    inplace=True
)

bond.sort_values(
    "Date",
    inplace=True
)

print("INDIA_10Y_YIELD:", bond.shape)


# =========================================================
# Merge all five datasets
# =========================================================

print("\n" + "=" * 70)
print("MERGING FIVE VARIABLES")
print("=" * 70)

df = nifty.merge(
    vix,
    on="Date",
    how="outer"
)

df = df.merge(
    crude,
    on="Date",
    how="outer"
)

df = df.merge(
    usd,
    on="Date",
    how="outer"
)

df = df.merge(
    bond,
    on="Date",
    how="outer"
)

# Ensure chronological order
df.sort_values(
    "Date",
    inplace=True
)

print("\nCombined shape before missing-value removal:")
print(df.shape)


# =========================================================
# Missing values
# =========================================================

print("\nMissing values before cleaning:")

print(
    df[
        [
            "NIFTY50",
            "INDIA_VIX",
            "CRUDE_OIL",
            "USD_INR",
            "INDIA_10Y_YIELD"
        ]
    ].isnull().sum()
)


# =========================================================
# Keep dates where all five variables are available
# =========================================================

required_columns = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_10Y_YIELD"
]

df.dropna(
    subset=required_columns,
    inplace=True
)

print("\nShape after removing missing dates:")
print(df.shape)


# =========================================================
# Calculate returns / changes
# =========================================================

# NIFTY50 percentage return
df["NIFTY50_Return"] = (
    df["NIFTY50"].pct_change()
)


# CRUDE OIL zero-crossing-safe return
df["CRUDE_OIL_Return"] = symmetric_change(
    df["CRUDE_OIL"]
)


# USD/INR percentage return
df["USD_INR_Return"] = (
    df["USD_INR"].pct_change()
)


# INDIA VIX percentage change
df["INDIA_VIX_Change"] = (
    df["INDIA_VIX"].pct_change()
)


# India 10-Year Government Bond Yield absolute change
df["INDIA_10Y_YIELD_Change"] = (
    df["INDIA_10Y_YIELD"].diff()
)


# =========================================================
# Rolling volatility
# =========================================================

df["NIFTY50_Volatility_30D"] = (
    df["NIFTY50_Return"]
    .rolling(30)
    .std()
)

df["CRUDE_OIL_Volatility_30D"] = (
    df["CRUDE_OIL_Return"]
    .rolling(30)
    .std()
)

df["USD_INR_Volatility_30D"] = (
    df["USD_INR_Return"]
    .rolling(30)
    .std()
)


# =========================================================
# Remove NaN values generated by calculations
# =========================================================

df.dropna(
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# =========================================================
# Save final processed dataset
# =========================================================

output_file = (
    PROCESSED_DIR /
    "macro_stress_5vars_processed.csv"
)

df.to_csv(
    output_file,
    index=False
)


# =========================================================
# Final validation
# =========================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)

print("\nFinal shape:")
print(df.shape)

print("\nDate range:")
print(
    df["Date"].min().date(),
    "→",
    df["Date"].max().date()
)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# =========================================================
# Validate crude-oil extreme events
# =========================================================

print("\n" + "=" * 70)
print("CRUDE OIL RETURN VALIDATION")
print("=" * 70)

extreme_crude = df.loc[
    df["CRUDE_OIL_Return"].abs().nlargest(10).index,
    [
        "Date",
        "CRUDE_OIL",
        "CRUDE_OIL_Return"
    ]
].sort_values("Date")

print("\nLargest absolute crude-oil changes:")
print(extreme_crude.to_string(index=False))


# Specifically inspect the April 2020 WTI event
april_2020 = df[
    df["Date"].astype(str).str.startswith("2020-04")
][
    [
        "Date",
        "CRUDE_OIL",
        "CRUDE_OIL_Return"
    ]
]

print("\nApril 2020 crude-oil observations:")
print(april_2020.to_string(index=False))


print("\nSaved to:")
print(output_file)

print("\n" + "=" * 70)
print("PROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)