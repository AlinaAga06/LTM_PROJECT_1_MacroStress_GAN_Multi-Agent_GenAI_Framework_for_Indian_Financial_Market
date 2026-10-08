from pathlib import Path

import pandas as pd
import yfinance as yf


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

OUTPUT_FILE = OUTPUT_DIR / "USD_INR.csv"

TICKER = "INR=X"

START_DATE = "2008-01-01"
END_DATE = "2026-01-01"


print("=" * 80)
print("MACROSTRESS-GAN")
print("BANKING DATA COLLECTION — USD/INR")
print("=" * 80)

print(f"Ticker       : {TICKER}")
print(f"Start Date   : {START_DATE}")
print(f"End Date     : {END_DATE}")
print(f"Output       : {OUTPUT_FILE}")
print()


print("Downloading USD/INR historical data...")


df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False,
)


if df.empty:
    raise RuntimeError(
        "USD/INR download returned no data."
    )


# Yahoo Finance can return MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        column[0]
        if isinstance(column, tuple)
        else column
        for column in df.columns
    ]


df = df.reset_index()


if "Date" not in df.columns:
    raise RuntimeError(
        f"Expected Date column. Found: {list(df.columns)}"
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)


required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


available_columns = [
    column
    for column in required_columns
    if column in df.columns
]


df = df[available_columns].copy()


for column in df.columns:

    if column != "Date":

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


df = df.dropna(
    subset=["Date"]
)


df = df.sort_values(
    "Date"
)


df = df.drop_duplicates(
    subset=["Date"],
    keep="last",
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 80)
print("USD/INR DOWNLOAD COMPLETE")
print("=" * 80)

print(
    f"Rows          : {len(df):,}"
)

print(
    f"Columns       : {len(df.columns)}"
)

print(
    f"First Date    : {df['Date'].min().date()}"
)

print(
    f"Last Date     : {df['Date'].max().date()}"
)

print()
print("Missing values:")

print(
    df.isna().sum()
)

print()
print("Saved:")

print(
    OUTPUT_FILE
)

print()
print("First 5 rows:")

print(
    df.head().to_string(index=False)
)

print()
print("Last 5 rows:")

print(
    df.tail().to_string(index=False)
)