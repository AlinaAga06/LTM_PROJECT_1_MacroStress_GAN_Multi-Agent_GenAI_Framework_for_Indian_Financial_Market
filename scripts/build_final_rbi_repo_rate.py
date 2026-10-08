from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "banking"

HISTORICAL = RAW / "REPO_RATE_HISTORICAL.csv"
EXISTING = RAW / "REPO_RATE.csv"

OUTPUT_EVENTS = RAW / "REPO_RATE_FINAL_EVENTS.csv"
OUTPUT_MONTHLY = RAW / "REPO_RATE_FINAL_MONTHLY.csv"


# =============================================================================
# 1. LOAD HISTORICAL RBI SERIES
# =============================================================================

historical = pd.read_csv(HISTORICAL)

historical["Date"] = pd.to_datetime(
    historical["Date"],
    errors="coerce"
)

historical["repo_rate"] = pd.to_numeric(
    historical["repo_rate"],
    errors="coerce"
)

historical = historical.dropna(
    subset=["Date", "repo_rate"]
)

historical["source"] = (
    "RBI Historical Policy Rate Table"
)


# =============================================================================
# 2. LOAD VERIFIED 2021-2024 WSS5 DATA
# =============================================================================

existing = pd.read_csv(EXISTING)

existing["Date"] = pd.to_datetime(
    existing["Date"],
    errors="coerce"
)

existing["repo_rate"] = pd.to_numeric(
    existing["repo_rate"],
    errors="coerce"
)

existing = existing.dropna(
    subset=["Date", "repo_rate"]
)

# Keep the directly extracted RBI WSS5 observations.
wss5 = existing[
    existing["Date"] <= pd.Timestamp("2024-08-29")
].copy()

wss5["source"] = "RBI WSS5"


# =============================================================================
# 3. ADD VERIFIED 2025 POLICY CHANGES
# =============================================================================

policy_2025 = pd.DataFrame(
    [
        # February 2025
        ("2025-02-07", 6.25, "RBI official policy decision"),

        # April 2025
        ("2025-04-09", 6.00, "RBI official policy decision"),

        # June 2025
        ("2025-06-06", 5.50, "RBI official policy decision"),
    ],
    columns=[
        "Date",
        "repo_rate",
        "source"
    ]
)

policy_2025["Date"] = pd.to_datetime(
    policy_2025["Date"]
)


# =============================================================================
# 4. COMBINE POLICY-EVENT OBSERVATIONS
# =============================================================================

events = pd.concat(
    [
        historical,
        wss5,
        policy_2025
    ],
    ignore_index=True
)

events = events[
    [
        "Date",
        "repo_rate",
        "source"
    ]
]

events = (
    events
    .dropna(subset=["Date", "repo_rate"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# =============================================================================
# 5. REMOVE DUPLICATES / CHECK CONFLICTS
# =============================================================================

print("=" * 80)
print("CHECKING REPO-RATE OVERLAPS")
print("=" * 80)

duplicates = (
    events
    .groupby("Date")["repo_rate"]
    .nunique()
)

conflicts = duplicates[
    duplicates > 1
]

if len(conflicts) > 0:
    print("\nCONFLICTING DATES FOUND:")
    print(conflicts)

    for date in conflicts.index:
        print(
            events[
                events["Date"] == date
            ].to_string(index=False)
        )

    raise ValueError(
        "Conflicting repo-rate values found. "
        "Resolve before continuing."
    )

# Same date/same value duplicates are harmless;
# retain the most authoritative/latest row.
events = (
    events
    .drop_duplicates(
        subset=["Date"],
        keep="last"
    )
    .sort_values("Date")
    .reset_index(drop=True)
)


# =============================================================================
# 6. IMPORTANT: ADD 2023-2024 CONTINUATION
# =============================================================================
#
# The policy repo rate stayed at 6.50% after the 08-Feb-2023 decision
# throughout 2023 and 2024 until the next verified change in Feb-2025.
#
# This is not a fabricated observation.
# It is a policy-rate carry-forward between verified policy decisions.
#
# We explicitly label it as such in the monthly dataset.
# =============================================================================

print("\nVerified event range:")
print(
    f"{events['Date'].min().date()} → "
    f"{events['Date'].max().date()}"
)


# =============================================================================
# 7. BUILD MONTHLY END-OF-MONTH POLICY RATE
# =============================================================================

start_month = pd.Period(
    "2003-03",
    freq="M"
)

end_month = pd.Period(
    "2025-12",
    freq="M"
)

months = pd.period_range(
    start=start_month,
    end=end_month,
    freq="M"
)

monthly = pd.DataFrame(
    {
        "Month": months
    }
)

monthly["Date"] = (
    monthly["Month"]
    .dt.to_timestamp(how="end")
    .dt.normalize()
)


# Merge policy events onto monthly calendar using
# the latest policy event effective on or before month-end.
event_lookup = events[
    ["Date", "repo_rate", "source"]
].sort_values("Date")

monthly = pd.merge_asof(
    monthly.sort_values("Date"),
    event_lookup.sort_values("Date"),
    on="Date",
    direction="backward"
)


# =============================================================================
# 8. VALIDATE THAT EVERY MONTH HAS A RATE
# =============================================================================

missing = monthly["repo_rate"].isna()

if missing.any():

    print("\nERROR: Missing repo-rate months:")

    print(
        monthly.loc[
            missing,
            ["Date"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Some months have no verified repo rate."
    )


# =============================================================================
# 9. LABEL MONTHLY SOURCE
# =============================================================================

monthly["source_type"] = "RBI policy-event carry-forward"

# For months whose month-end date itself corresponds to
# an actual policy event, identify them as direct event months.
event_dates = set(
    events["Date"].dt.to_period("M")
)

monthly.loc[
    monthly["Month"].isin(event_dates),
    "source_type"
] = "RBI policy event / effective rate"


# =============================================================================
# 10. FINAL COLUMNS
# =============================================================================

monthly = monthly[
    [
        "Date",
        "repo_rate",
        "source",
        "source_type"
    ]
]


# =============================================================================
# 11. SAVE
# =============================================================================

events.to_csv(
    OUTPUT_EVENTS,
    index=False
)

monthly.to_csv(
    OUTPUT_MONTHLY,
    index=False
)


# =============================================================================
# 12. VALIDATION OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL RBI REPO RATE SERIES")
print("=" * 80)

print(
    f"Policy events : {len(events)}"
)

print(
    f"Monthly rows  : {len(monthly)}"
)

print(
    f"Date range    : "
    f"{monthly['Date'].min().date()} → "
    f"{monthly['Date'].max().date()}"
)

print(
    f"Missing       : "
    f"{monthly['repo_rate'].isna().sum()}"
)

print(
    f"Duplicates    : "
    f"{monthly['Date'].duplicated().sum()}"
)

print("\n2021-2025 monthly repo rates:")

print(
    monthly[
        monthly["Date"] >= pd.Timestamp("2021-01-01")
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_EVENTS)
print(OUTPUT_MONTHLY)

print("\nDONE.")