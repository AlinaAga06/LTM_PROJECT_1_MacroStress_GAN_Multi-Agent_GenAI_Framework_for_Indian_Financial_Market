"""
MACROSTRESS-GAN
BANKING MONTHLY DATASET BUILDER — CONTIGUOUS REAL-DATA VERSION

Purpose
-------
Build a continuous monthly banking dataset from REAL data only.

Variables
---------
BANK_NIFTY
REPO_RATE
CPI_INFLATION
CREDIT_GROWTH
USD_INR

Important methodology
---------------------
- No interpolation
- No synthetic observations
- No forward-fill of missing core variables
- No backward-fill
- Detects real calendar gaps
- Automatically selects the LONGEST continuous common block
- Model cutoff = 2025-12-31
- Previous banking_monthly_dataset.csv is not overwritten

TimeGAN
-------
Sequence length = 30 months
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "banking"

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "institutions"
    / "banking"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# INPUT FILES
# =============================================================================

BANK_NIFTY_FILE = RAW_DIR / "BANK_NIFTY.csv"
USD_INR_FILE = RAW_DIR / "USD_INR.csv"
REPO_FILE = RAW_DIR / "REPO_RATE_FINAL_MONTHLY.csv"
CPI_FILE = RAW_DIR / "CPI_INFLATION.csv"
CREDIT_FILE = RAW_DIR / "CREDIT_GROWTH_LONG.csv"


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_FILE = (
    PROCESSED_DIR
    / "banking_monthly_dataset_long.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_END_DATE = pd.Timestamp("2025-12-31")

SEQ_LEN = 30


# =============================================================================
# HELPERS
# =============================================================================

def check_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


def normalize_month_end(series):

    dates = pd.to_datetime(
        series,
        errors="coerce"
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found."
        )

    return (
        dates
        .dt
        .to_period("M")
        .dt
        .to_timestamp("M")
    )


def validate_monthly_series(
    df,
    date_col,
    value_col,
    name
):

    print("\n" + "-" * 80)
    print(f"VALIDATING: {name}")
    print("-" * 80)

    if date_col not in df.columns:

        raise ValueError(
            f"{name}: missing {date_col}"
        )

    if value_col not in df.columns:

        raise ValueError(
            f"{name}: missing {value_col}"
        )

    df = df.copy()

    df[date_col] = normalize_month_end(
        df[date_col]
    )

    duplicate_count = int(
        df[date_col].duplicated().sum()
    )

    if duplicate_count > 0:

        print(
            f"WARNING: {duplicate_count} duplicates"
        )

        df = (
            df
            .sort_values(date_col)
            .drop_duplicates(
                date_col,
                keep="last"
            )
        )

    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce"
    )

    missing = int(
        df[value_col].isna().sum()
    )

    print(
        f"Rows           : {len(df)}"
    )

    print(
        f"Date range     : "
        f"{df[date_col].min().date()} → "
        f"{df[date_col].max().date()}"
    )

    print(
        f"Missing values : {missing}"
    )

    print(
        f"Duplicates     : {duplicate_count}"
    )

    return (
        df[
            [
                date_col,
                value_col
            ]
        ]
        .sort_values(date_col)
        .reset_index(drop=True)
    )


# =============================================================================
# BANK NIFTY
# =============================================================================

def load_bank_nifty():

    check_file(BANK_NIFTY_FILE)

    df = pd.read_csv(
        BANK_NIFTY_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if "Date" not in df.columns:

        raise ValueError(
            "BANK_NIFTY.csv missing Date."
        )

    close_col = None

    for candidate in [
        "Adj Close",
        "Close"
    ]:

        if candidate in df.columns:

            close_col = candidate
            break

    if close_col is None:

        raise ValueError(
            "BANK_NIFTY.csv has no Close/Adj Close."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["BANK_NIFTY"] = pd.to_numeric(
        df[close_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "BANK_NIFTY"
        ]
    )

    df["Month"] = (
        df["Date"]
        .dt
        .to_period("M")
    )

    monthly = (
        df
        .sort_values("Date")
        .groupby("Month", as_index=False)
        .tail(1)
        [
            [
                "Date",
                "BANK_NIFTY"
            ]
        ]
        .copy()
    )

    monthly["Date"] = normalize_month_end(
        monthly["Date"]
    )

    return validate_monthly_series(
        monthly,
        "Date",
        "BANK_NIFTY",
        "BANK_NIFTY"
    )


# =============================================================================
# USD INR
# =============================================================================

def load_usd_inr():

    check_file(USD_INR_FILE)

    df = pd.read_csv(
        USD_INR_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if "Date" not in df.columns:

        raise ValueError(
            "USD_INR.csv missing Date."
        )

    close_col = None

    for candidate in [
        "Adj Close",
        "Close"
    ]:

        if candidate in df.columns:

            close_col = candidate
            break

    if close_col is None:

        raise ValueError(
            "USD_INR.csv has no Close/Adj Close."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["USD_INR"] = pd.to_numeric(
        df[close_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "USD_INR"
        ]
    )

    df["Month"] = (
        df["Date"]
        .dt
        .to_period("M")
    )

    monthly = (
        df
        .sort_values("Date")
        .groupby("Month", as_index=False)
        .tail(1)
        [
            [
                "Date",
                "USD_INR"
            ]
        ]
        .copy()
    )

    monthly["Date"] = normalize_month_end(
        monthly["Date"]
    )

    return validate_monthly_series(
        monthly,
        "Date",
        "USD_INR",
        "USD_INR"
    )


# =============================================================================
# REPO RATE
# =============================================================================

def load_repo():

    check_file(REPO_FILE)

    df = pd.read_csv(
        REPO_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print("\nREPO FILE COLUMNS:")
    print(list(df.columns))

    if "Date" not in df.columns:

        raise ValueError(
            "REPO file missing Date."
        )

    if "repo_rate" not in df.columns:

        raise ValueError(
            "REPO file missing repo_rate."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["repo_rate"] = pd.to_numeric(
        df["repo_rate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "repo_rate"
        ]
    )

    df["Date"] = normalize_month_end(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            "Date",
            keep="last"
        )
    )

    return (
        validate_monthly_series(
            df,
            "Date",
            "repo_rate",
            "REPO_RATE"
        )
        .rename(
            columns={
                "repo_rate":
                "REPO_RATE"
            }
        )
    )


# =============================================================================
# CPI
# =============================================================================

def load_cpi():

    check_file(CPI_FILE)

    df = pd.read_csv(
        CPI_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if "Date" not in df.columns:

        raise ValueError(
            "CPI file missing Date."
        )

    if "CPI_INFLATION" not in df.columns:

        raise ValueError(
            "CPI file missing CPI_INFLATION."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["CPI_INFLATION"] = pd.to_numeric(
        df["CPI_INFLATION"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df["Date"] = normalize_month_end(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            "Date",
            keep="last"
        )
    )

    return validate_monthly_series(
        df,
        "Date",
        "CPI_INFLATION",
        "CPI_INFLATION"
    )


# =============================================================================
# CREDIT GROWTH
# =============================================================================

def load_credit_growth():

    check_file(CREDIT_FILE)

    df = pd.read_csv(
        CREDIT_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print("\nCREDIT FILE COLUMNS:")
    print(list(df.columns))

    required = [
        "Date",
        "BANK_CREDIT",
        "CREDIT_GROWTH"
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"CREDIT_GROWTH_LONG.csv "
                f"missing {col}"
            )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["BANK_CREDIT"] = pd.to_numeric(
        df["BANK_CREDIT"],
        errors="coerce"
    )

    df["CREDIT_GROWTH"] = pd.to_numeric(
        df["CREDIT_GROWTH"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df["Date"] = normalize_month_end(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            "Date",
            keep="last"
        )
    )

    print("\nRBI LONG CREDIT SERIES")

    print(
        f"Rows            : {len(df)}"
    )

    print(
        f"Date range      : "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Credit missing  : "
        f"{int(df['BANK_CREDIT'].isna().sum())}"
    )

    print(
        f"Growth missing  : "
        f"{int(df['CREDIT_GROWTH'].isna().sum())}"
    )

    return df[
        [
            "Date",
            "BANK_CREDIT",
            "CREDIT_GROWTH"
        ]
    ].copy()


# =============================================================================
# CONTIGUOUS BLOCK DETECTION
# =============================================================================

def find_contiguous_blocks(df):

    """
    Detect consecutive monthly blocks.

    Example:

    2012-01
    2012-02
    2012-03
    ...
    2013-05

    then gap

    2014-01
    ...

    Each continuous section becomes a block.
    """

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    if len(df) == 0:

        return []

    expected_next = (
        df["Date"]
        .shift(1)
        .dt
        .to_period("M")
        .add(1)
        .dt
        .to_timestamp("M")
    )

    new_block = (
        df["Date"] != expected_next
    )

    new_block.iloc[0] = True

    df["block_id"] = (
        new_block
        .cumsum()
    )

    blocks = []

    for block_id, block in df.groupby(
        "block_id"
    ):

        blocks.append(
            {
                "block_id": int(block_id),
                "start": block["Date"].min(),
                "end": block["Date"].max(),
                "rows": len(block)
            }
        )

    return blocks


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("MACROSTRESS-GAN")
    print("BANKING MONTHLY DATASET BUILDER")
    print("CONTIGUOUS REAL-DATA VERSION")
    print("=" * 80)

    print(
        f"\nProject root : {ROOT}"
    )

    print(
        f"Output       : {OUTPUT_FILE}"
    )

    print(
        f"Model cutoff : {MODEL_END_DATE.date()}"
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    bank_nifty = load_bank_nifty()

    usd_inr = load_usd_inr()

    repo = load_repo()

    cpi = load_cpi()

    credit = load_credit_growth()

    # =========================================================================
    # MERGE
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("BUILDING COMMON MONTHLY DATASET")
    print("=" * 80)

    merged = bank_nifty.merge(
        repo,
        on="Date",
        how="inner"
    )

    merged = merged.merge(
        cpi,
        on="Date",
        how="inner"
    )

    merged = merged.merge(
        credit,
        on="Date",
        how="inner"
    )

    merged = merged.merge(
        usd_inr,
        on="Date",
        how="inner"
    )

    merged = (
        merged
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"\nMerged rows before cutoff : "
        f"{len(merged)}"
    )

    # =========================================================================
    # CUTOFF
    # =========================================================================

    merged = merged[
        merged["Date"] <= MODEL_END_DATE
    ].copy()

    merged = (
        merged
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"Rows after 2025-12 cutoff : "
        f"{len(merged)}"
    )

    # =========================================================================
    # CORE VALIDATION
    # =========================================================================

    core_columns = [
        "BANK_NIFTY",
        "REPO_RATE",
        "CPI_INFLATION",
        "CREDIT_GROWTH",
        "USD_INR"
    ]

    print("\n")
    print("=" * 80)
    print("CORE VARIABLE VALIDATION")
    print("=" * 80)

    for col in core_columns:

        missing = int(
            merged[col].isna().sum()
        )

        print(
            f"{col:<20} "
            f"missing={missing:<5} "
            f"rows={len(merged)}"
        )

        if missing > 0:

            raise ValueError(
                f"Core variable {col} "
                f"contains missing values."
            )

    # =========================================================================
    # DUPLICATES
    # =========================================================================

    duplicates = int(
        merged["Date"].duplicated().sum()
    )

    print(
        f"\nDuplicate dates : {duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate dates found."
        )

    # =========================================================================
    # CONTIGUOUS BLOCK ANALYSIS
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("CONTIGUOUS BLOCK ANALYSIS")
    print("=" * 80)

    blocks = find_contiguous_blocks(
        merged[["Date"]]
    )

    for block in blocks:

        print(
            f"\nBlock {block['block_id']}"
        )

        print(
            f"Start : "
            f"{block['start'].date()}"
        )

        print(
            f"End   : "
            f"{block['end'].date()}"
        )

        print(
            f"Rows  : "
            f"{block['rows']}"
        )

    if not blocks:

        raise RuntimeError(
            "No contiguous blocks found."
        )

    # =========================================================================
    # SELECT LONGEST BLOCK
    # =========================================================================

    selected_block = max(
        blocks,
        key=lambda x: (
            x["rows"],
            x["end"]
        )
    )

    selected_start = (
        selected_block["start"]
    )

    selected_end = (
        selected_block["end"]
    )

    print("\n")
    print("=" * 80)
    print("SELECTED MODELING BLOCK")
    print("=" * 80)

    print(
        f"Start : {selected_start.date()}"
    )

    print(
        f"End   : {selected_end.date()}"
    )

    print(
        f"Rows  : {selected_block['rows']}"
    )

    # =========================================================================
    # FILTER LONGEST BLOCK
    # =========================================================================

    merged = merged[
        (
            merged["Date"]
            >= selected_start
        )
        &
        (
            merged["Date"]
            <= selected_end
        )
    ].copy()

    merged = (
        merged
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # =========================================================================
    # FINAL CONTINUITY CHECK
    # =========================================================================

    expected_dates = pd.date_range(
        start=merged["Date"].min(),
        end=merged["Date"].max(),
        freq="ME"
    )

    actual_dates = pd.DatetimeIndex(
        merged["Date"]
    )

    missing_months = (
        expected_dates
        .difference(actual_dates)
    )

    print("\n")
    print("=" * 80)
    print("CONTINUITY VALIDATION")
    print("=" * 80)

    print(
        f"Expected months : "
        f"{len(expected_dates)}"
    )

    print(
        f"Actual months   : "
        f"{len(actual_dates)}"
    )

    print(
        f"Missing months  : "
        f"{len(missing_months)}"
    )

    if len(missing_months) > 0:

        print(
            "Missing dates:"
        )

        for date in missing_months:

            print(
                f"  {date.date()}"
            )

        raise ValueError(
            "Selected modeling block is not continuous."
        )

    # =========================================================================
    # TIMEGAN FEATURES
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("CREATING TIMEGAN FEATURES")
    print("=" * 80)

    merged["BANK_NIFTY_Return"] = (
        merged["BANK_NIFTY"]
        .pct_change()
        * 100
    )

    merged["REPO_RATE_Change"] = (
        merged["REPO_RATE"]
        .diff()
    )

    merged["CPI_INFLATION_Change"] = (
        merged["CPI_INFLATION"]
        .diff()
    )

    merged["CREDIT_GROWTH_Change"] = (
        merged["CREDIT_GROWTH"]
        .diff()
    )

    merged["USD_INR_Return"] = (
        merged["USD_INR"]
        .pct_change()
        * 100
    )

    feature_columns = [
        "BANK_NIFTY_Return",
        "REPO_RATE_Change",
        "CPI_INFLATION_Change",
        "CREDIT_GROWTH_Change",
        "USD_INR_Return"
    ]

    print("\nDerived feature validation:")

    for col in feature_columns:

        missing = int(
            merged[col].isna().sum()
        )

        print(
            f"{col:<25} "
            f"missing={missing}"
        )

        if missing != 1:

            raise ValueError(
                f"{col} expected exactly "
                f"one initial NaN."
            )

    # =========================================================================
    # REMOVE INITIAL DERIVED FEATURE ROW
    # =========================================================================

    merged = (
        merged
        .dropna(
            subset=feature_columns
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # FINAL CONTINUITY CHECK
    # =========================================================================

    expected_after = pd.date_range(
        start=merged["Date"].min(),
        end=merged["Date"].max(),
        freq="ME"
    )

    actual_after = pd.DatetimeIndex(
        merged["Date"]
    )

    missing_after = (
        expected_after
        .difference(actual_after)
    )

    if len(missing_after) > 0:

        raise ValueError(
            "Missing months after feature creation."
        )

    # =========================================================================
    # FINAL COLUMNS
    # =========================================================================

    final_columns = [
        "Date",

        "BANK_NIFTY",
        "REPO_RATE",
        "CPI_INFLATION",
        "CREDIT_GROWTH",
        "USD_INR",

        "BANK_CREDIT",

        "BANK_NIFTY_Return",
        "REPO_RATE_Change",
        "CPI_INFLATION_Change",
        "CREDIT_GROWTH_Change",
        "USD_INR_Return"
    ]

    merged = merged[
        final_columns
    ]

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("FINAL DATASET VALIDATION")
    print("=" * 80)

    print(
        f"Rows        : {len(merged)}"
    )

    print(
        f"Columns     : {len(merged.columns)}"
    )

    print(
        f"Date range  : "
        f"{merged['Date'].min().date()} → "
        f"{merged['Date'].max().date()}"
    )

    print(
        f"Duplicates  : "
        f"{int(merged['Date'].duplicated().sum())}"
    )

    print(
        f"Missing     : "
        f"{int(merged.isna().sum().sum())}"
    )

    # =========================================================================
    # HARD VALIDATION
    # =========================================================================

    if merged["Date"].duplicated().any():

        raise ValueError(
            "Final dataset contains duplicate dates."
        )

    if merged.isna().any().any():

        raise ValueError(
            "Final dataset contains missing values."
        )

    # =========================================================================
    # TIMEGAN SEQUENCES
    # =========================================================================

    possible_sequences = max(
        0,
        len(merged) - SEQ_LEN + 1
    )

    print("\n")
    print("=" * 80)
    print("TIMEGAN SEQUENCE VALIDATION")
    print("=" * 80)

    print(
        f"Sequence length       : {SEQ_LEN}"
    )

    print(
        f"Usable monthly rows   : {len(merged)}"
    )

    print(
        f"Possible 30M windows  : "
        f"{possible_sequences}"
    )

    expected_minimum = 50

    if possible_sequences >= expected_minimum:

        print(
            "\nSTATUS: PASS"
        )

        print(
            f"At least {expected_minimum} "
            f"30-month sequences available."
        )

    else:

        print(
            "\nSTATUS: WARNING"
        )

        print(
            f"Only {possible_sequences} "
            f"30-month sequences available."
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("CORE VARIABLE SUMMARY")
    print("=" * 80)

    print(
        merged[
            [
                "BANK_NIFTY",
                "REPO_RATE",
                "CPI_INFLATION",
                "CREDIT_GROWTH",
                "USD_INR"
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )

    # =========================================================================
    # FIRST 5
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("FIRST 5 OBSERVATIONS")
    print("=" * 80)

    print(
        merged
        .head(5)
        .to_string(index=False)
    )

    # =========================================================================
    # LAST 5
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("LAST 5 OBSERVATIONS")
    print("=" * 80)

    print(
        merged
        .tail(5)
        .to_string(index=False)
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =========================================================================
    # SUCCESS
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("BANKING DATASET CREATED SUCCESSFULLY")
    print("=" * 80)

    print(
        f"Output file : {OUTPUT_FILE}"
    )

    print(
        f"Rows        : {len(merged)}"
    )

    print(
        f"Columns     : {len(merged.columns)}"
    )

    print(
        f"Date range  : "
        f"{merged['Date'].min().date()} → "
        f"{merged['Date'].max().date()}"
    )

    print(
        f"Missing     : "
        f"{int(merged.isna().sum().sum())}"
    )

    print(
        f"Duplicates  : "
        f"{int(merged['Date'].duplicated().sum())}"
    )

    print(
        f"30M windows : "
        f"{possible_sequences}"
    )

    print(
        "\nPrevious banking_monthly_dataset.csv "
        "was NOT modified."
    )

    print(
        "No interpolation or synthetic observations were used."
    )

    print("\n")


if __name__ == "__main__":
    main()