import requests
import pandas as pd
from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

URL = (
    "https://data-api.dbie.rbihub.in/api/tables/"
    "financial_sector/"
    "r152_deployment_of_bank_credit_by_major_sectors/"
    "rows"
)

OUTPUT = Path(
    "data/raw/banking/CREDIT_GROWTH.csv"
)


# ================================================================
# HELPERS
# ================================================================

def clean_number(value):
    """
    Convert RBI formatted numeric strings such as:

        '2,07,54,078'
        '93,32,555'

    into numeric values.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def get_text(value):
    if value is None:
        return ""

    return str(value).strip()


# ================================================================
# DOWNLOAD
# ================================================================

print("=" * 80)
print("BANKING — RBI BANK CREDIT → CREDIT GROWTH")
print("=" * 80)

print("\nDownloading RBI DBIE table...")

response = requests.get(
    URL,
    timeout=120
)

print("HTTP Status:", response.status_code)

response.raise_for_status()

data = response.json()

rows = data["rows"]

print("Rows returned:", len(rows))


# ================================================================
# INSPECT TABLE STRUCTURE
# ================================================================

print("\n" + "=" * 80)
print("RBI TABLE STRUCTURE")
print("=" * 80)

for row in rows:

    if len(row) < 6:
        continue

    row_no = row[3]
    tab = row[1]
    label = row[5]

    if row_no in [5, 6, 7, 8, 9, 10, 11]:

        print(
            f"tab={tab!r:20} "
            f"row_no={row_no!s:4} "
            f"label={label!r}"
        )


# ================================================================
# FIND ALL CANDIDATE DATE ROWS
# ================================================================

date_candidates = []

for row in rows:

    if len(row) < 6:
        continue

    row_no = row[3]
    label = get_text(row[5])

    if row_no == 7:

        date_candidates.append(
            {
                "tab": row[1],
                "row": row,
                "label": label
            }
        )


print("\nDate-row candidates:", len(date_candidates))

for i, candidate in enumerate(date_candidates):

    row = candidate["row"]

    values = row[4:]

    parsed_dates = []

    for value in values:

        date = pd.to_datetime(
            value,
            errors="coerce"
        )

        if not pd.isna(date):
            parsed_dates.append(date)

    if parsed_dates:

        print(
            f"[{i}] "
            f"tab={candidate['tab']} "
            f"range={min(parsed_dates).date()} "
            f"→ {max(parsed_dates).date()} "
            f"observations={len(parsed_dates)}"
        )


# ================================================================
# FIND DATE ROW WITH LATEST COVERAGE
# ================================================================

best_date_row = None
best_date_end = None

for candidate in date_candidates:

    row = candidate["row"]

    parsed_dates = []

    for value in row[4:]:

        date = pd.to_datetime(
            value,
            errors="coerce"
        )

        if not pd.isna(date):
            parsed_dates.append(date)

    if not parsed_dates:
        continue

    candidate_end = max(parsed_dates)

    if best_date_end is None or candidate_end > best_date_end:

        best_date_end = candidate_end
        best_date_row = candidate


if best_date_row is None:

    raise RuntimeError(
        "Could not find an RBI observation-date row."
    )


print("\nSelected date row:")

print(
    "  tab      :",
    best_date_row["tab"]
)

print(
    "  row_no   :",
    best_date_row["row"][3]
)

print(
    "  coverage :",
    best_date_end.date()
)


# ================================================================
# FIND BANK CREDIT ROW IN SAME TAB
# ================================================================

selected_tab = best_date_row["tab"]

credit_candidates = []

for row in rows:

    if len(row) < 6:
        continue

    row_no = row[3]
    tab = row[1]
    label = get_text(row[5])

    if (
        tab == selected_tab
        and row_no == 8
    ):

        credit_candidates.append(row)


print(
    "\nBank-credit candidates in selected tab:",
    len(credit_candidates)
)


# ================================================================
# VALIDATE CREDIT ROW
# ================================================================

if not credit_candidates:

    raise RuntimeError(
        "Could not find Bank Credit row in the same RBI tab "
        "as the selected observation-date row."
    )


# Usually exactly one.
# If multiple exist, choose the one with the largest
# number of numeric observations.

best_credit_row = None
best_credit_count = -1

for row in credit_candidates:

    numeric_count = 0

    for value in row[4:]:

        if clean_number(value) is not None:
            numeric_count += 1

    if numeric_count > best_credit_count:

        best_credit_count = numeric_count
        best_credit_row = row


print("\nSelected credit row:")

print(
    "  tab       :",
    best_credit_row[1]
)

print(
    "  row_no    :",
    best_credit_row[3]
)

print(
    "  label     :",
    best_credit_row[5]
)

print(
    "  numeric observations:",
    best_credit_count
)


# ================================================================
# EXTRACT DATES + BANK CREDIT
# ================================================================

date_values = best_date_row["row"][4:]
credit_values = best_credit_row[4:]

records = []

for date_value, credit_value in zip(
    date_values,
    credit_values
):

    if date_value is None:
        continue

    if not str(date_value).strip():
        continue

    date = pd.to_datetime(
        date_value,
        errors="coerce"
    )

    credit = clean_number(
        credit_value
    )

    if pd.isna(date):
        continue

    if credit is None:
        continue

    records.append(
        {
            "Date": date,
            "BANK_CREDIT_CRORE": credit
        }
    )


df = pd.DataFrame(records)


# ================================================================
# BASIC VALIDATION
# ================================================================

if df.empty:

    raise RuntimeError(
        "No bank-credit observations were extracted."
    )


df = df.sort_values(
    "Date"
)

df = df.drop_duplicates(
    subset=["Date"],
    keep="last"
)

df = df.reset_index(
    drop=True
)


print("\n" + "=" * 80)
print("BANK CREDIT VALIDATION")
print("=" * 80)

print(
    "Observations:",
    len(df)
)

print(
    "Date range:",
    df["Date"].min().date(),
    "→",
    df["Date"].max().date()
)

print(
    "Missing dates:",
    df["Date"].isna().sum()
)

print(
    "Missing credit:",
    df["BANK_CREDIT_CRORE"].isna().sum()
)

print(
    "Duplicate dates:",
    df["Date"].duplicated().sum()
)


# ================================================================
# CREATE YEAR / MONTH
# ================================================================

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month


# ================================================================
# PREVIOUS-YEAR MATCH
# ================================================================

previous = df[
    [
        "Date",
        "BANK_CREDIT_CRORE",
        "Year",
        "Month"
    ]
].copy()

previous = previous.rename(
    columns={
        "Date": "Previous_Date",
        "BANK_CREDIT_CRORE":
            "BANK_CREDIT_PREVIOUS_YEAR",
        "Year":
            "Previous_Year"
    }
)

df["Previous_Year"] = (
    df["Year"] - 1
)


df = df.merge(
    previous[
        [
            "Previous_Date",
            "BANK_CREDIT_PREVIOUS_YEAR",
            "Previous_Year",
            "Month"
        ]
    ],
    on=[
        "Previous_Year",
        "Month"
    ],
    how="left"
)


# ================================================================
# CALCULATE CREDIT GROWTH
# ================================================================

df["CREDIT_GROWTH"] = (

    (
        df["BANK_CREDIT_CRORE"]
        /
        df["BANK_CREDIT_PREVIOUS_YEAR"]
    )
    - 1

) * 100


# ================================================================
# REMOVE OBSERVATIONS WITHOUT PRIOR YEAR
# ================================================================

result = df[
    df["BANK_CREDIT_PREVIOUS_YEAR"].notna()
].copy()


# ================================================================
# KEEP REQUIRED COLUMNS
# ================================================================

result = result[
    [
        "Date",
        "BANK_CREDIT_CRORE",
        "BANK_CREDIT_PREVIOUS_YEAR",
        "CREDIT_GROWTH"
    ]
]


# ================================================================
# FINAL VALIDATION
# ================================================================

print("\n" + "=" * 80)
print("CREDIT GROWTH VALIDATION")
print("=" * 80)

print(
    "Rows:",
    len(result)
)

print(
    "Date range:",
    result["Date"].min().date(),
    "→",
    result["Date"].max().date()
)

print(
    "Missing CREDIT_GROWTH:",
    result["CREDIT_GROWTH"].isna().sum()
)

print(
    "Duplicate dates:",
    result["Date"].duplicated().sum()
)

print(
    "September 2021 onward:",
    (
        result["Date"]
        >= pd.Timestamp("2021-09-01")
    ).sum(),
    "observations"
)


# ================================================================
# CHECK 2021–2025 COVERAGE
# ================================================================

print("\nObservations by year:")

print(
    result[
        result["Date"].dt.year >= 2021
    ]
    .groupby(
        result[
            result["Date"].dt.year >= 2021
        ]["Date"].dt.year
    )
    .size()
    .to_string()
)


# ================================================================
# SHOW RECENT DATA
# ================================================================

print("\nRecent RBI bank-credit observations:")

print(
    result.tail(20).to_string(
        index=False
    )
)


# ================================================================
# SAVE
# ================================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False
)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(
    "File:",
    OUTPUT
)

print(
    "Rows:",
    len(result)
)

print(
    "Columns:",
    list(result.columns)
)

print("=" * 80)