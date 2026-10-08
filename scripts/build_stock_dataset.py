from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "institutions" / "stock"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

FILES = {
    "NIFTY50": RAW_DIR / "NIFTY50.csv",
    "INDIA_VIX": RAW_DIR / "INDIA_VIX.csv",
    "USD_INR": RAW_DIR / "USD_INR.csv",
    "CRUDE_OIL": RAW_DIR / "CRUDE_OIL.csv",
    "INDIA_10Y_YIELD": RAW_DIR / "INDIA_10Y_YIELD.csv",
}


# ============================================================
# READ YAHOO FINANCE FILE
# ============================================================

def read_yahoo_file(path: Path, variable_name: str) -> pd.DataFrame:

    print(f"\nReading: {path.name}")

    df = pd.read_csv(path)

    # Yahoo Finance CSV structure:
    #
    # Row 0 -> Ticker
    # Row 1 -> Date
    # Row 2 onward -> actual data
    #
    # Example:
    # Price | Adj Close | Close | ...
    # Ticker | ^NSEI    | ^NSEI
    # Date   | NaN      | NaN
    # 2008-01-01 | ...
    #
    # Remove ticker/date metadata rows.

    if len(df) >= 2:
        first_value = str(df.iloc[0, 0]).strip().lower()
        second_value = str(df.iloc[1, 0]).strip().lower()

        if first_value == "ticker" and second_value == "date":
            df = df.iloc[2:].copy()

    # First column contains the actual date
    date_column = df.columns[0]

    # Convert date
    df["Date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    # Prefer Close
    if "Close" in df.columns:
        value_column = "Close"
    elif "Adj Close" in df.columns:
        value_column = "Adj Close"
    else:
        raise ValueError(
            f"No Close or Adj Close column found in {path.name}. "
            f"Columns: {list(df.columns)}"
        )

    # Convert numeric
    df[variable_name] = pd.to_numeric(
        df[value_column],
        errors="coerce"
    )

    # Keep required columns
    df = df[["Date", variable_name]].copy()

    # Remove invalid rows
    df = df.dropna(subset=["Date", variable_name])

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"])

    # Sort
    df = df.sort_values("Date")

    print(f"  Variable : {variable_name}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Start    : {df['Date'].min().date()}")
    print(f"  End      : {df['Date'].max().date()}")

    return df


# ============================================================
# READ INDIA 10Y YIELD
# ============================================================

def read_india_10y(path: Path) -> pd.DataFrame:

    print(f"\nReading: {path.name}")

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["INDIA_10Y_YIELD"] = pd.to_numeric(
        df["INDIA_10Y_YIELD"],
        errors="coerce"
    )

    df = df[
        ["Date", "INDIA_10Y_YIELD"]
    ].copy()

    df = df.dropna(
        subset=["Date", "INDIA_10Y_YIELD"]
    )

    df = df.drop_duplicates(
        subset=["Date"]
    )

    df = df.sort_values("Date")

    print(f"  Variable : INDIA_10Y_YIELD")
    print(f"  Rows     : {len(df):,}")
    print(f"  Start    : {df['Date'].min().date()}")
    print(f"  End      : {df['Date'].max().date()}")

    return df


# ============================================================
# LOAD DATA
# ============================================================

nifty = read_yahoo_file(
    FILES["NIFTY50"],
    "NIFTY50"
)

vix = read_yahoo_file(
    FILES["INDIA_VIX"],
    "INDIA_VIX"
)

usdinr = read_yahoo_file(
    FILES["USD_INR"],
    "USD_INR"
)

crude = read_yahoo_file(
    FILES["CRUDE_OIL"],
    "CRUDE_OIL"
)

yield_10y = read_india_10y(
    FILES["INDIA_10Y_YIELD"]
)


# ============================================================
# MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATASETS")
print("=" * 70)

df = nifty.copy()

df = df.merge(
    vix,
    on="Date",
    how="inner"
)

df = df.merge(
    usdinr,
    on="Date",
    how="inner"
)

df = df.merge(
    crude,
    on="Date",
    how="inner"
)

df = df.merge(
    yield_10y,
    on="Date",
    how="inner"
)


# ============================================================
# SORT AND CLEAN
# ============================================================

df = df.sort_values("Date")

df = df.drop_duplicates(
    subset=["Date"]
)

required_columns = [
    "Date",
    "NIFTY50",
    "INDIA_VIX",
    "USD_INR",
    "CRUDE_OIL",
    "INDIA_10Y_YIELD",
]

df = df[required_columns]


# Remove infinite values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows with missing values
df = df.dropna(
    subset=required_columns
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(f"Rows          : {len(df):,}")
print(f"Columns       : {len(df.columns)}")
print(f"Start Date    : {df['Date'].min().date()}")
print(f"End Date      : {df['Date'].max().date()}")
print(f"Missing Values: {df.isna().sum().sum()}")
print(f"Duplicate Dates: {df['Date'].duplicated().sum()}")

print("\nColumns:")
for column in df.columns:
    print(f"  ✓ {column}")


# ============================================================
# BASIC DATA QUALITY CHECK
# ============================================================

print("\nValue ranges:")

for column in required_columns[1:]:
    print(
        f"  {column:20s} "
        f"min={df[column].min():.4f} "
        f"max={df[column].max():.4f}"
    )


# ============================================================
# SAVE
# ============================================================

output_file = OUTPUT_DIR / "stock_5vars.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("STOCK DATASET CREATED")
print("=" * 70)

print(f"\nSaved to:")
print(output_file)

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nLast 5 rows:")
print(df.tail().to_string(index=False))

print("\nSUCCESS")