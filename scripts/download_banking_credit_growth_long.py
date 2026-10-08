"""
MacroStress-GAN
Long-History RBI Bank Credit Growth Extraction

Source:
RBI DBIE
Table:
financial_sector.r689_business_of_scheduled_banks

Selected series:
Row 20 -> 7) Bank credit

Method:
1. Download RBI DBIE full-history table
2. Extract Reporting Friday dates from row 6
3. Extract Bank Credit from row 20
4. Convert RBI Indian-number format to numeric
5. Convert fortnightly observations to month-end
   using the last available observation in each month
6. Calculate YoY Bank Credit Growth
7. Validate coverage, duplicates and missing months
8. Save a separate candidate file

IMPORTANT:
This script does NOT overwrite the existing CREDIT_GROWTH.csv.
"""

from pathlib import Path
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

URL = (
    "https://data-api.dbie.rbihub.in/"
    "api/tables/financial_sector/"
    "r689_business_of_scheduled_banks/rows"
)

OUTPUT_DIR = Path("data/raw/banking")
OUTPUT_FILE = OUTPUT_DIR / "CREDIT_GROWTH_LONG.csv"

TIMEOUT = 60

# RBI DBIE row identified from inspection
DATE_ROW_NO = 6
BANK_CREDIT_ROW_NO = 20


# ============================================================
# HELPERS
# ============================================================

def clean_indian_number(value):
    """
    Convert RBI Indian-formatted numbers such as:

        '2,23,29,258'
        '27,75,549'

    into floats.

    Empty / invalid values become NaN.
    """

    if value is None:
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    # Remove commas used in Indian number formatting
    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_rbi_date(value):
    """
    Parse RBI dates such as:

        15-Sep-26
        31-Aug-26
        01-Jan-21

    """

    if value is None:
        return pd.NaT

    value = str(value).strip()

    if not value:
        return pd.NaT

    return pd.to_datetime(
        value,
        format="%d-%b-%y",
        errors="coerce"
    )


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 90)
print("MACROSTRESS-GAN")
print("LONG-HISTORY RBI BANK CREDIT EXTRACTION")
print("=" * 90)

print()
print("Source:")
print(URL)

print()
print("Downloading RBI DBIE data...")

response = requests.get(
    URL,
    timeout=TIMEOUT
)

response.raise_for_status()

data = response.json()

print("HTTP Status :", response.status_code)


# ============================================================
# READ TABLE
# ============================================================

columns = data["columns"]
rows = data["rows"]

print("Total rows  :", len(rows))
print("Total cols  :", len(columns))


# Convert rows into dictionaries
records = [
    dict(zip(columns, row))
    for row in rows
]

df = pd.DataFrame(records)

print()
print("DataFrame shape:", df.shape)


# ============================================================
# FIND DATE ROW
# ============================================================

date_rows = df[
    df["row_no"].astype(str) == str(DATE_ROW_NO)
]

if date_rows.empty:
    raise RuntimeError(
        f"Could not find date row {DATE_ROW_NO}"
    )

date_row = date_rows.iloc[0]

print()
print("Date row:")
print("ROW NO :", date_row["row_no"])
print("LABEL  :", date_row["c1"])


# ============================================================
# FIND BANK CREDIT ROW
# ============================================================

credit_rows = df[
    df["row_no"].astype(str) == str(BANK_CREDIT_ROW_NO)
]

if credit_rows.empty:
    raise RuntimeError(
        f"Could not find Bank Credit row {BANK_CREDIT_ROW_NO}"
    )

credit_row = credit_rows.iloc[0]

print()
print("Bank Credit row:")
print("ROW NO :", credit_row["row_no"])
print("LABEL  :", credit_row["c1"])


# ============================================================
# EXTRACT DATE / VALUE PAIRS
# ============================================================

print()
print("Extracting date/value pairs...")

output = []

for col in columns:

    # Only data columns c1, c2, ...
    if not col.startswith("c"):
        continue

    # Skip c1 because it contains labels
    if col == "c1":
        continue

    date_value = date_row[col]
    credit_value = credit_row[col]

    if date_value in (None, ""):
        continue

    date = parse_rbi_date(date_value)

    if pd.isna(date):
        continue

    credit = clean_indian_number(credit_value)

    if pd.isna(credit):
        continue

    output.append(
        {
            "Date": date,
            "BANK_CREDIT": credit,
        }
    )


credit_df = pd.DataFrame(output)


# ============================================================
# VALIDATE RAW SERIES
# ============================================================

if credit_df.empty:
    raise RuntimeError(
        "No Bank Credit observations were extracted."
    )

credit_df = credit_df.sort_values("Date")

credit_df = credit_df.drop_duplicates(
    subset=["Date"],
    keep="last"
)

credit_df = credit_df.reset_index(drop=True)


print()
print("=" * 90)
print("RAW BANK CREDIT SERIES")
print("=" * 90)

print("Rows        :", len(credit_df))
print(
    "Date range  :",
    credit_df["Date"].min().date(),
    "→",
    credit_df["Date"].max().date()
)

print(
    "Missing     :",
    int(credit_df["BANK_CREDIT"].isna().sum())
)

print(
    "Duplicates  :",
    int(credit_df["Date"].duplicated().sum())
)


# ============================================================
# MONTHLY CONVERSION
# ============================================================

print()
print("Converting fortnightly observations to monthly...")

credit_df["Month"] = (
    credit_df["Date"]
    .dt.to_period("M")
)

monthly = (
    credit_df
    .sort_values("Date")
    .groupby("Month", as_index=False)
    .tail(1)
    .copy()
)

monthly = monthly.sort_values("Date")

monthly["Month"] = monthly["Date"].dt.to_period("M")


# Month-end date
monthly["Date"] = (
    monthly["Month"]
    .dt.to_timestamp("M")
)


# ============================================================
# YOY BANK CREDIT GROWTH
# ============================================================

monthly["CREDIT_GROWTH"] = (
    monthly["BANK_CREDIT"]
    .pct_change(periods=12)
    * 100.0
)


# ============================================================
# SOURCE METADATA
# ============================================================

monthly["source"] = "RBI DBIE"
monthly["source_table"] = (
    "financial_sector.r689_business_of_scheduled_banks"
)
monthly["source_frequency"] = "Fortnightly"
monthly["source_row"] = "7) Bank credit"
monthly["aggregation"] = (
    "Last available RBI observation in each calendar month"
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

monthly = monthly[
    [
        "Date",
        "BANK_CREDIT",
        "CREDIT_GROWTH",
        "source",
        "source_table",
        "source_frequency",
        "source_row",
        "aggregation",
    ]
]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 90)
print("MONTHLY DATASET VALIDATION")
print("=" * 90)

print(
    "Monthly rows :",
    len(monthly)
)

print(
    "Date range   :",
    monthly["Date"].min().date(),
    "→",
    monthly["Date"].max().date()
)

print(
    "Credit min   :",
    round(monthly["BANK_CREDIT"].min(), 4)
)

print(
    "Credit max   :",
    round(monthly["BANK_CREDIT"].max(), 4)
)

print(
    "Credit mean  :",
    round(monthly["BANK_CREDIT"].mean(), 4)
)

print(
    "YoY valid    :",
    int(monthly["CREDIT_GROWTH"].notna().sum())
)

print(
    "YoY missing  :",
    int(monthly["CREDIT_GROWTH"].isna().sum())
)

print(
    "Duplicates   :",
    int(monthly["Date"].duplicated().sum())
)


# ============================================================
# CHECK MONTH CONTINUITY
# ============================================================

expected_months = pd.date_range(
    start=monthly["Date"].min(),
    end=monthly["Date"].max(),
    freq="ME"
)

actual_months = monthly["Date"]

missing_months = expected_months[
    ~expected_months.isin(actual_months)
]

print(
    "Missing months:",
    len(missing_months)
)

if len(missing_months) > 0:
    print("Missing month examples:")
    print(missing_months[:20])


# ============================================================
# SAMPLE
# ============================================================

print()
print("=" * 90)
print("FIRST 10 MONTHLY OBSERVATIONS")
print("=" * 90)

print(
    monthly[
        [
            "Date",
            "BANK_CREDIT",
            "CREDIT_GROWTH"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 90)
print("LAST 10 MONTHLY OBSERVATIONS")
print("=" * 90)

print(
    monthly[
        [
            "Date",
            "BANK_CREDIT",
            "CREDIT_GROWTH"
        ]
    ]
    .tail(10)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

monthly.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(
    "Output:",
    OUTPUT_FILE
)

print()
print("IMPORTANT:")
print("Existing CREDIT_GROWTH.csv was NOT modified.")

print()
print("LONG RBI BANK CREDIT EXTRACTION COMPLETED")