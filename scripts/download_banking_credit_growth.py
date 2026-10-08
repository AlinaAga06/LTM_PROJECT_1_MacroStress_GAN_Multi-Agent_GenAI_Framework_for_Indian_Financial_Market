# ==============================================================================
# MACROSTRESS-GAN
# RBI BANK CREDIT GROWTH DOWNLOADER — FIXED
# ==============================================================================
#
# Source:
#   RBI Database on Indian Economy (DBIE)
#
# Table:
#   financial_sector.r152_deployment_of_bank_credit_by_major_sectors
#
# RBI table structure:
#
#   row_no = 7  -> observation/reporting dates
#   row_no = 8  -> I. Bank Credit (II+III)
#
# The RBI CSV uses columns c1, c2, c3, ...
# and Indian-number-format strings such as:
#
#   1,01,05,176
#
# Therefore those values must be cleaned before numeric conversion.
#
# NO:
#   - synthetic data
#   - interpolation
#   - extrapolation
#   - fabricated observations
#
# ==============================================================================

from pathlib import Path
from io import StringIO
import sys

import numpy as np
import pandas as pd
import requests


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "CREDIT_GROWTH.csv"
)


# ==============================================================================
# RBI DBIE URL
# ==============================================================================

TABLE_URL = (
    "https://data-api.dbie.rbihub.in/"
    "api/tables/financial_sector/"
    "r152_deployment_of_bank_credit_by_major_sectors"
)

CSV_URL = (
    TABLE_URL
    + "/csv"
)


# ==============================================================================
# SETTINGS
# ==============================================================================

TIMEOUT = 60


# ==============================================================================
# PRINT HELPERS
# ==============================================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ==============================================================================
# DOWNLOAD RBI CSV
# ==============================================================================

section(
    "RBI DBIE BANK CREDIT TABLE"
)

print(
    "Table:"
)

print(
    "financial_sector."
    "r152_deployment_of_bank_credit_by_major_sectors"
)

print()
print(
    "CSV URL:"
)

print(CSV_URL)


section(
    "DOWNLOADING RBI CSV"
)

try:

    response = requests.get(
        CSV_URL,
        timeout=TIMEOUT,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/154.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

except Exception as exc:

    print(
        "ERROR downloading RBI data:"
    )

    print(exc)

    sys.exit(1)


print(
    f"Downloaded characters: "
    f"{len(response.text):,}"
)


# ==============================================================================
# READ CSV
# ==============================================================================

try:

    raw = pd.read_csv(
        StringIO(response.text),
        dtype=str,
    )

except Exception as exc:

    print(
        "ERROR reading RBI CSV:"
    )

    print(exc)

    sys.exit(1)


print()
print(
    f"Rows    : {len(raw):,}"
)

print(
    f"Columns : {len(raw.columns):,}"
)


# ==============================================================================
# CONVERT ROW NUMBER
# ==============================================================================

raw["row_no_numeric"] = pd.to_numeric(
    raw["row_no"],
    errors="coerce",
)


# ==============================================================================
# IDENTIFY REQUIRED ROWS
# ==============================================================================

date_rows = raw[
    raw["row_no_numeric"] == 7
].copy()

credit_rows = raw[
    raw["row_no_numeric"] == 8
].copy()


print()
print(
    f"Date rows   : {len(date_rows)}"
)

print(
    f"Credit rows : {len(credit_rows)}"
)


if date_rows.empty:

    print(
        "ERROR: RBI date row not found."
    )

    sys.exit(1)


if credit_rows.empty:

    print(
        "ERROR: RBI Bank Credit row not found."
    )

    sys.exit(1)


# ==============================================================================
# SELECT ROW
# ==============================================================================

date_row = date_rows.iloc[0]

credit_row = credit_rows.iloc[0]


print()
print(
    "RBI date row selected : row_no 7"
)

print(
    "RBI credit row selected : row_no 8"
)


# ==============================================================================
# OBSERVATION COLUMNS
# ==============================================================================
#
# The RBI table has:
#
# c1 = serial number
# c2 = sector
# c3 onward = observations
#
# We therefore only inspect c3 onward.
# ==============================================================================

observation_columns = [
    column
    for column in raw.columns
    if str(column).startswith("c")
    and str(column)[1:].isdigit()
    and int(str(column)[1:]) >= 3
]


print()
print(
    f"Observation columns: "
    f"{len(observation_columns)}"
)

print(
    f"First observation column: "
    f"{observation_columns[0]}"
)

print(
    f"Last observation column: "
    f"{observation_columns[-1]}"
)


# ==============================================================================
# EXTRACT OBSERVATIONS
# ==============================================================================

section(
    "EXTRACTING REAL RBI BANK CREDIT OBSERVATIONS"
)

records = []


for column in observation_columns:

    # --------------------------------------------------------------------------
    # DATE
    # --------------------------------------------------------------------------

    date_text = date_row.get(
        column,
        None,
    )

    if pd.isna(date_text):

        continue

    date_text = str(
        date_text
    ).strip()

    if not date_text:

        continue

    parsed_date = pd.to_datetime(
        date_text,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(parsed_date):

        continue


    # --------------------------------------------------------------------------
    # BANK CREDIT
    # --------------------------------------------------------------------------

    credit_text = credit_row.get(
        column,
        None,
    )

    if pd.isna(credit_text):

        continue

    credit_text = str(
        credit_text
    ).strip()


    if not credit_text:

        continue


    # --------------------------------------------------------------------------
    # REMOVE INDIAN NUMBER FORMAT COMMAS
    #
    # Example:
    #
    #   "1,01,05,176"
    #
    # becomes:
    #
    #   "10105176"
    # --------------------------------------------------------------------------

    credit_clean = (
        credit_text
        .replace(",", "")
        .replace(" ", "")
    )


    numeric_credit = pd.to_numeric(
        credit_clean,
        errors="coerce",
    )


    if pd.isna(numeric_credit):

        continue


    records.append(
        {
            "Source_Date": parsed_date,
            "BANK_CREDIT_CRORE": float(
                numeric_credit
            ),
        }
    )


# ==============================================================================
# CHECK EXTRACTION
# ==============================================================================

credit = pd.DataFrame(
    records
)


print()
print(
    f"Extracted RBI observations: "
    f"{len(credit):,}"
)


if credit.empty:

    print()
    print(
        "ERROR: No RBI bank-credit observations "
        "were extracted."
    )

    sys.exit(1)


# ==============================================================================
# CLEAN
# ==============================================================================

credit["Source_Date"] = pd.to_datetime(
    credit["Source_Date"],
    errors="coerce",
)

credit = credit.dropna(
    subset=[
        "Source_Date",
        "BANK_CREDIT_CRORE",
    ]
)


credit = (
    credit
    .sort_values("Source_Date")
    .drop_duplicates(
        subset=["Source_Date"],
        keep="last",
    )
    .reset_index(drop=True)
)


# ==============================================================================
# DISPLAY RAW COVERAGE
# ==============================================================================

section(
    "RAW RBI CREDIT COVERAGE"
)

print(
    f"Observations : {len(credit):,}"
)

print(
    f"First date   : "
    f"{credit['Source_Date'].min().date()}"
)

print(
    f"Last date    : "
    f"{credit['Source_Date'].max().date()}"
)


# ==============================================================================
# CONVERT TO MONTHLY
# ==============================================================================
#
# RBI observations are reporting dates, generally Fridays.
#
# For our monthly MacroStress-GAN dataset:
#
#   one observation per month
#   = last available RBI reporting observation in that month
#
# We do NOT interpolate between reporting dates.
# ==============================================================================

section(
    "CONVERTING RBI REPORTING DATA TO MONTHLY"
)

credit["Month"] = (
    credit["Source_Date"]
    .dt.to_period("M")
)


monthly = (
    credit
    .sort_values("Source_Date")
    .groupby(
        "Month",
        as_index=False,
    )
    .last()
)


# Month-end date for alignment with CPI etc.

monthly["Date"] = (
    monthly["Month"]
    .dt.to_timestamp("M")
)


monthly = monthly[
    [
        "Date",
        "Source_Date",
        "BANK_CREDIT_CRORE",
    ]
].copy()


monthly = (
    monthly
    .sort_values("Date")
    .reset_index(drop=True)
)


print(
    f"Monthly observations: "
    f"{len(monthly):,}"
)

print(
    f"Monthly range:"
)

print(
    f"{monthly['Date'].min().date()}"
    f" → "
    f"{monthly['Date'].max().date()}"
)


# ==============================================================================
# YEAR-ON-YEAR MATCH
# ==============================================================================
#
# We calculate:
#
# CREDIT_GROWTH =
#
# ((Current Bank Credit /
#   Previous-Year Bank Credit) - 1) * 100
#
# The previous-year value must be a REAL RBI observation.
# ==============================================================================

section(
    "CALCULATING YEAR-ON-YEAR CREDIT GROWTH"
)

monthly["Year"] = (
    monthly["Date"].dt.year
)

monthly["Month_Number"] = (
    monthly["Date"].dt.month
)


# Lookup table.

previous_year_lookup = monthly[
    [
        "Year",
        "Month_Number",
        "BANK_CREDIT_CRORE",
    ]
].copy()


previous_year_lookup = (
    previous_year_lookup
    .rename(
        columns={
            "Year": "Previous_Year",
            "BANK_CREDIT_CRORE":
                "BANK_CREDIT_PREVIOUS_YEAR",
        }
    )
)


monthly["Previous_Year"] = (
    monthly["Year"] - 1
)


monthly = monthly.merge(
    previous_year_lookup[
        [
            "Previous_Year",
            "Month_Number",
            "BANK_CREDIT_PREVIOUS_YEAR",
        ]
    ],
    on=[
        "Previous_Year",
        "Month_Number",
    ],
    how="left",
)


# ==============================================================================
# CALCULATE GROWTH
# ==============================================================================

monthly["CREDIT_GROWTH"] = np.nan


valid = (
    monthly[
        "BANK_CREDIT_PREVIOUS_YEAR"
    ].notna()
    &
    (
        monthly[
            "BANK_CREDIT_PREVIOUS_YEAR"
        ] > 0
    )
)


monthly.loc[
    valid,
    "CREDIT_GROWTH",
] = (
    (
        monthly.loc[
            valid,
            "BANK_CREDIT_CRORE",
        ]
        /
        monthly.loc[
            valid,
            "BANK_CREDIT_PREVIOUS_YEAR",
        ]
    )
    - 1.0
) * 100.0


# ==============================================================================
# FINAL DATASET
# ==============================================================================

monthly = monthly[
    [
        "Date",
        "Source_Date",
        "BANK_CREDIT_CRORE",
        "BANK_CREDIT_PREVIOUS_YEAR",
        "CREDIT_GROWTH",
    ]
].copy()


# ==============================================================================
# VALIDATION
# ==============================================================================

section(
    "FINAL RBI CREDIT GROWTH VALIDATION"
)

print(
    f"Rows                     : "
    f"{len(monthly):,}"
)

print(
    f"Date range               : "
    f"{monthly['Date'].min().date()}"
    f" → "
    f"{monthly['Date'].max().date()}"
)

print(
    f"Valid YoY observations   : "
    f"{monthly['CREDIT_GROWTH'].notna().sum():,}"
)

print(
    f"Missing YoY observations : "
    f"{monthly['CREDIT_GROWTH'].isna().sum():,}"
)

print(
    f"Duplicate dates          : "
    f"{monthly['Date'].duplicated().sum()}"
)


# ==============================================================================
# MONTH GAP CHECK
# ==============================================================================

expected_months = pd.date_range(
    start=monthly["Date"].min(),
    end=monthly["Date"].max(),
    freq="ME",
)

actual_months = pd.DatetimeIndex(
    monthly["Date"]
)

missing_months = (
    expected_months
    .difference(actual_months)
)


print(
    f"Missing calendar months  : "
    f"{len(missing_months)}"
)


if len(missing_months) > 0:

    for date in missing_months:

        print(
            "  Missing:",
            date.strftime(
                "%Y-%m-%d"
            ),
        )


# ==============================================================================
# GROWTH STATISTICS
# ==============================================================================

valid_growth = monthly[
    "CREDIT_GROWTH"
].dropna()


print()
print(
    "Credit growth statistics:"
)

print(
    f"Minimum : "
    f"{valid_growth.min():.4f}%"
)

print(
    f"Maximum : "
    f"{valid_growth.max():.4f}%"
)

print(
    f"Mean    : "
    f"{valid_growth.mean():.4f}%"
)


# ==============================================================================
# FIRST OBSERVATIONS
# ==============================================================================

section(
    "FIRST 15 MONTHLY OBSERVATIONS"
)

print(
    monthly.head(15).to_string(
        index=False
    )
)


# ==============================================================================
# LAST OBSERVATIONS
# ==============================================================================

section(
    "LAST 15 MONTHLY OBSERVATIONS"
)

print(
    monthly.tail(15).to_string(
        index=False
    )
)


# ==============================================================================
# SAVE
# ==============================================================================

section(
    "SAVING RBI CREDIT GROWTH DATASET"
)

monthly.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)


# ==============================================================================
# FINAL STATUS
# ==============================================================================

print()
print("=" * 80)
print(
    "RBI CREDIT GROWTH DOWNLOAD COMPLETED SUCCESSFULLY"
)
print("=" * 80)

print(
    f"Monthly rows : {len(monthly):,}"
)

print(
    f"Range        : "
    f"{monthly['Date'].min().date()}"
    f" → "
    f"{monthly['Date'].max().date()}"
)

print(
    f"Valid YoY    : "
    f"{valid_growth.shape[0]:,}"
)

print(
    f"Output       : "
    f"{OUTPUT_FILE}"
)

print("=" * 80)