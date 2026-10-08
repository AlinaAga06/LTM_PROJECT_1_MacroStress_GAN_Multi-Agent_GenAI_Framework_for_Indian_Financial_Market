import io
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://data-api.dbie.rbihub.in"

TABLE = "real_sector/cpi_av_rn"

CSV_URL = f"{BASE_URL}/api/tables/{TABLE}/csv"

OUTPUT = Path("data/raw/banking/CPI_INFLATION.csv")


print("=" * 80)
print("BANKING — RBI CPI INFLATION")
print("=" * 80)

print("Downloading RBI DBIE CPI table...")
print("URL:", CSV_URL)

response = requests.get(
    CSV_URL,
    timeout=120,
)

print("HTTP Status:", response.status_code)

response.raise_for_status()

raw = response.text

print("Downloaded characters:", len(raw))

if not raw.strip():
    raise RuntimeError("RBI DBIE returned an empty CSV response.")

# ------------------------------------------------------------------
# Read complete DBIE table
# ------------------------------------------------------------------

df = pd.read_csv(io.StringIO(raw))

print("\nRAW TABLE")
print("Rows:", len(df))
print("Columns:")
for col in df.columns:
    print("  -", col)

# ------------------------------------------------------------------
# Find CPI Combined series
# ------------------------------------------------------------------

# DBIE table should contain the series dimension.
series_col = None

for candidate in [
    "type_cpi_rn",
    "TYPE_CPI_RN",
]:
    if candidate in df.columns:
        series_col = candidate
        break

if series_col is None:
    raise RuntimeError(
        "Could not find CPI series-code column. "
        f"Available columns: {list(df.columns)}"
    )

print("\nSeries column:", series_col)

cpi = df[
    df[series_col].astype(str).str.strip()
    == "CPI_COM_AIGI"
].copy()

print("CPI_COM_AIGI rows:", len(cpi))

if cpi.empty:
    print("\nAvailable series codes:")
    print(
        df[series_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(50)
        .to_string(index=False)
    )

    raise RuntimeError(
        "CPI_COM_AIGI was not found in the downloaded DBIE table."
    )

# ------------------------------------------------------------------
# Identify date/value columns
# ------------------------------------------------------------------

date_col = None
value_col = None

for candidate in [
    "time_period",
    "TIME_PERIOD",
    "period",
    "PERIOD",
]:
    if candidate in cpi.columns:
        date_col = candidate
        break

for candidate in [
    "obs_value",
    "OBS_VALUE",
    "value",
    "VALUE",
]:
    if candidate in cpi.columns:
        value_col = candidate
        break

if date_col is None:
    raise RuntimeError(
        f"Could not identify date column. Columns: {list(cpi.columns)}"
    )

if value_col is None:
    raise RuntimeError(
        f"Could not identify observation-value column. "
        f"Columns: {list(cpi.columns)}"
    )

print("Date column:", date_col)
print("Value column:", value_col)

# ------------------------------------------------------------------
# Build clean dataset
# ------------------------------------------------------------------

result = pd.DataFrame()

result["Date"] = pd.to_datetime(
    cpi[date_col],
    errors="coerce",
)

result["CPI_INFLATION"] = pd.to_numeric(
    cpi[value_col],
    errors="coerce",
)

result["series_code"] = "CPI_COM_AIGI"

# Preserve the unit if available.
if "unit_measure" in cpi.columns:
    result["unit"] = cpi["unit_measure"].astype(str)
else:
    result["unit"] = ""

result = result.dropna(
    subset=["Date", "CPI_INFLATION"]
)

result = result.sort_values("Date")

result = result.drop_duplicates(
    subset=["Date"],
    keep="last",
)

result = result.reset_index(drop=True)

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CPI VALIDATION")
print("=" * 80)

print("Rows:", len(result))
print(
    "Date range:",
    result["Date"].min().date(),
    "→",
    result["Date"].max().date(),
)

print(
    "Missing CPI:",
    result["CPI_INFLATION"].isna().sum(),
)

print(
    "Duplicate dates:",
    result["Date"].duplicated().sum(),
)

print(
    "September 2021 onward:",
    (
        result["Date"]
        >= pd.Timestamp("2021-09-01")
    ).sum(),
    "observations",
)

print("\nLatest observations:")
print(
    result.tail(15).to_string(index=False)
)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print(OUTPUT)
print("Rows:", len(result))
print("Columns:", list(result.columns))