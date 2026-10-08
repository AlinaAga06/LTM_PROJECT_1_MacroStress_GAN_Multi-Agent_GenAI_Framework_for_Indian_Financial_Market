from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = Path("data/raw/banking/REPO_RATE.csv")

OUTPUT_FILE = Path(
    "data/raw/banking/REPO_RATE_MONTHLY.csv"
)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("BANKING — MONTHLY RBI POLICY REPO RATE")
print("=" * 80)


# =============================================================================
# LOAD EXISTING RBI SERIES
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


df["repo_rate"] = pd.to_numeric(
    df["repo_rate"],
    errors="coerce"
)


df = (
    df
    .dropna(subset=["Date", "repo_rate"])
    .sort_values("Date")
    .reset_index(drop=True)
)


print()
print("Weekly RBI observations loaded:", len(df))

print(
    "Weekly date range:",
    df["Date"].min().date(),
    "→",
    df["Date"].max().date()
)


# =============================================================================
# MONTHLY CONVERSION OF EXISTING RBI OBSERVATIONS
#
# Use the LAST AVAILABLE RBI OBSERVATION IN EACH MONTH.
#
# This is appropriate because the policy repo rate is an effective policy
# rate that remains in force until the next change.
# =============================================================================

df["Month"] = df["Date"].dt.to_period("M")


monthly = (
    df
    .sort_values("Date")
    .groupby("Month", as_index=False)
    .last()
)


monthly["Date"] = (
    monthly["Month"]
    .dt.to_timestamp("M")
)


monthly = monthly[
    [
        "Date",
        "repo_rate"
    ]
]


monthly["source"] = "RBI WSS5"


# =============================================================================
# EXTEND WITH OFFICIAL RBI POLICY CHANGES AFTER AUGUST 2024
#
# These are effective-date changes, not fabricated monthly observations.
#
# 2025:
# - 06-Feb-2025: 6.25%
# - 09-Apr-2025: 6.00%
# - 06-Jun-2025: 5.50%
#
# The February 2025 change was the first reduction after the 6.50%
# period represented in our extracted WSS5 data.
# =============================================================================

official_changes = pd.DataFrame(
    [
        {
            "effective_date": "2025-02-06",
            "repo_rate": 6.25,
            "source": "RBI official policy decision"
        },
        {
            "effective_date": "2025-04-09",
            "repo_rate": 6.00,
            "source": "RBI official policy decision"
        },
        {
            "effective_date": "2025-06-06",
            "repo_rate": 5.50,
            "source": "RBI official policy decision"
        },
    ]
)


official_changes["effective_date"] = pd.to_datetime(
    official_changes["effective_date"]
)


# =============================================================================
# BUILD MONTHLY EXTENSION
# =============================================================================

last_existing_date = monthly["Date"].max()


# We need September 2024 through December 2025
future_months = pd.date_range(
    start="2024-09-30",
    end="2025-12-31",
    freq="ME"
)


extension_rows = []


for month_end in future_months:

    applicable = official_changes[
        official_changes["effective_date"] <= month_end
    ]

    if month_end < pd.Timestamp("2025-02-01"):

        # Rate prevailing after the final verified WSS5 observation
        rate = 6.50
        source = "RBI WSS5 carry-forward"

    elif month_end < pd.Timestamp("2025-04-01"):

        rate = 6.25
        source = "RBI official policy decision"

    elif month_end < pd.Timestamp("2025-06-01"):

        rate = 6.00
        source = "RBI official policy decision"

    else:

        rate = 5.50
        source = "RBI official policy decision"


    extension_rows.append(
        {
            "Date": month_end,
            "repo_rate": rate,
            "source": source
        }
    )


extension = pd.DataFrame(extension_rows)


# =============================================================================
# COMBINE
# =============================================================================

combined = pd.concat(
    [
        monthly,
        extension
    ],
    ignore_index=True
)


combined = (
    combined
    .sort_values("Date")
    .drop_duplicates(
        subset=["Date"],
        keep="last"
    )
    .reset_index(drop=True)
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print()
print("=" * 80)
print("MONTHLY VALIDATION")
print("=" * 80)

print()
print("Rows:", len(combined))

print(
    "Date range:",
    combined["Date"].min().date(),
    "→",
    combined["Date"].max().date()
)

print()
print("Unique repo rates:")

for value in sorted(
    combined["repo_rate"].unique()
):
    print(
        f"  {value:.2f}%"
    )


print()
print("2024-2025 extension:")

print(
    combined[
        combined["Date"] >= "2024-09-01"
    ].to_string(index=False)
)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print("File:", OUTPUT_FILE)
print("Rows:", len(combined))
print("Columns:", len(combined.columns))

print()
print("Monthly Repo Rate dataset ready.")