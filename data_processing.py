import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "NIFTY50": RAW_DIR / "NIFTY50.csv",
    "INDIA_VIX": RAW_DIR / "INDIA_VIX.csv",
    "CRUDE_OIL": RAW_DIR / "CRUDE_OIL.csv",
    "USD_INR": RAW_DIR / "USD_INR.csv"
}


# ============================================================
# Function to clean Yahoo Finance CSV
# ============================================================

def clean_yahoo_file(file_path, column_name):

    print("\n" + "=" * 60)
    print(f"Processing: {file_path}")
    print("=" * 60)

    # Read CSV
    df = pd.read_csv(file_path)

    print("Original shape:", df.shape)

    # --------------------------------------------------------
    # Remove Yahoo Finance extra rows
    # --------------------------------------------------------

    # Keep only rows where Price is NOT "Ticker" or "Date"
    df = df[
        ~df["Price"].isin(["Ticker", "Date"])
    ].copy()

    # Rename Price column to Date
    df = df.rename(columns={"Price": "Date"})

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    # --------------------------------------------------------
    # Convert Close to numeric
    # --------------------------------------------------------

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # Remove rows where Close is missing
    df = df.dropna(subset=["Close"])

    # --------------------------------------------------------
    # Keep only Date and Close
    # --------------------------------------------------------

    df = df[["Date", "Close"]]

    # Rename Close
    df = df.rename(
        columns={"Close": column_name}
    )

    # Sort by date
    df = df.sort_values("Date")

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["Date"]
    )

    print("Cleaned shape:", df.shape)

    print(
        "Date range:",
        df["Date"].min(),
        "to",
        df["Date"].max()
    )

    print("Missing values:")
    print(df.isnull().sum())

    return df


# ============================================================
# Process each dataset
# ============================================================

cleaned_data = {}

for name, file_path in FILES.items():

    cleaned_data[name] = clean_yahoo_file(
        file_path,
        name
    )


# ============================================================
# Merge all datasets
# ============================================================

print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)

combined = cleaned_data["NIFTY50"]

for name in [
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR"
]:

    combined = pd.merge(
        combined,
        cleaned_data[name],
        on="Date",
        how="outer"
    )


# Sort by date
combined = combined.sort_values("Date")


print("\nCombined dataset shape:")
print(combined.shape)

print("\nCombined date range:")
print(combined["Date"].min(), "to", combined["Date"].max())


# ============================================================
# Check missing values
# ============================================================

print("\nMissing values before cleaning:")

print(combined.isnull().sum())


# ============================================================
# Remove rows where any required variable is missing
# ============================================================

combined_clean = combined.dropna().copy()


print("\nShape after removing missing dates:")
print(combined_clean.shape)


print("\nMissing values after cleaning:")

print(combined_clean.isnull().sum())


# ============================================================
# Calculate daily percentage returns
# ============================================================

combined_clean["NIFTY50_Return"] = (
    combined_clean["NIFTY50"]
    .pct_change()
)

combined_clean["CRUDE_OIL_Return"] = (
    combined_clean["CRUDE_OIL"]
    .pct_change()
)

combined_clean["USD_INR_Return"] = (
    combined_clean["USD_INR"]
    .pct_change()
)

combined_clean["INDIA_VIX_Change"] = (
    combined_clean["INDIA_VIX"]
    .pct_change()
)


# Remove first row created by pct_change()
combined_clean = combined_clean.dropna()


# ============================================================
# Calculate rolling volatility
# ============================================================

combined_clean["NIFTY50_Volatility_30D"] = (
    combined_clean["NIFTY50_Return"]
    .rolling(window=30)
    .std()
)

combined_clean["CRUDE_OIL_Volatility_30D"] = (
    combined_clean["CRUDE_OIL_Return"]
    .rolling(window=30)
    .std()
)

combined_clean["USD_INR_Volatility_30D"] = (
    combined_clean["USD_INR_Return"]
    .rolling(window=30)
    .std()
)


# ============================================================
# Remove rows created by rolling calculation
# ============================================================

combined_clean = combined_clean.dropna()


# ============================================================
# Save processed dataset
# ============================================================

output_file = (
    PROCESSED_DIR /
    "macro_stress_processed.csv"
)

combined_clean.to_csv(
    output_file,
    index=False
)


# ============================================================
# Final information
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)

print("\nFinal dataset:")
print(output_file)

print("\nFinal shape:")
print(combined_clean.shape)

print("\nFinal columns:")
print(combined_clean.columns.tolist())

print("\nFirst 5 rows:")
print(combined_clean.head())

print("\nLast 5 rows:")
print(combined_clean.tail())

print("\nFinal missing-value check:")
print(combined_clean.isnull().sum())