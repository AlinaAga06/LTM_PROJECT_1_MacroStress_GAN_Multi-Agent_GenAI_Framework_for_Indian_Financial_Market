from pathlib import Path
import json
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

JS_FILE = Path("rbi_main.js")
OUTPUT_FILE = Path("data/raw/banking/REPO_RATE.csv")


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("RBI POLICY REPO RATE EXTRACTION")
print("=" * 80)


# =============================================================================
# READ RBI JAVASCRIPT
# =============================================================================

if not JS_FILE.exists():
    raise FileNotFoundError(
        f"RBI JavaScript file not found: {JS_FILE.resolve()}"
    )

text = JS_FILE.read_text(
    encoding="utf-8",
    errors="ignore"
)

print("JavaScript file:", JS_FILE)
print("JavaScript size:", f"{len(text):,}", "characters")


# =============================================================================
# LOCATE EMBEDDED WSS5 JSON
# =============================================================================

marker = "yve=JSON.parse('"

start = text.find(marker)

if start == -1:
    raise RuntimeError(
        "Could not find embedded WSS5 JSON marker: yve=JSON.parse('"
    )

json_start = start + len(marker)

print()
print("WSS5 JSON marker found")
print("JSON start position:", json_start)


# =============================================================================
# FIND THE REAL END OF THE JSON STRING
#
# The RBI JavaScript contains:
#
#     yve=JSON.parse('....JSON....')
#
# We cannot simply search for '), because the JSON itself can contain
# escaped characters and JavaScript surrounding the JSON.
#
# Instead, use JSONDecoder.raw_decode() to parse exactly one JSON object
# from the beginning of the extracted text.
# =============================================================================

json_candidate = text[json_start:]

# The JSON is inside a JavaScript single-quoted string.
# Decode the JavaScript escaped quote sequence first only where necessary.
#
# IMPORTANT:
# Do not globally replace all backslashes because that could corrupt
# valid JSON escaping.

# Find the closing JavaScript quote candidates.
# We try progressively larger sections and let json.loads/raw_decode
# determine the exact JSON boundary.

decoder = json.JSONDecoder()

data = None
json_end = None

# The embedded object is large, so first locate the likely closing
# sequence of JSON.parse('...').
#
# Search for the next occurrence of:
#     ')
#
# and test each candidate until valid JSON is obtained.

search_pos = json_start

while True:
    close_pos = text.find("')", search_pos)

    if close_pos == -1:
        break

    candidate = text[json_start:close_pos]

    try:
        # JavaScript string escaping for apostrophes.
        candidate_json = candidate.replace("\\'", "'")

        parsed, consumed = decoder.raw_decode(candidate_json)

        # We require the decoder to consume the complete candidate.
        remaining = candidate_json[consumed:].strip()

        if remaining == "":
            data = parsed
            json_end = close_pos
            break

    except (json.JSONDecodeError, ValueError):
        pass

    search_pos = close_pos + 2


if data is None:
    raise RuntimeError(
        "Could not identify the complete embedded WSS5 JSON object."
    )


print("JSON end position:", json_end)
print("Extracted JSON characters:", f"{json_end - json_start:,}")


# =============================================================================
# VERIFY TOP LEVEL
# =============================================================================

print()
print("Top-level keys:")

if isinstance(data, dict):
    for key in data.keys():
        print("  -", key)
else:
    raise RuntimeError("Extracted object is not a JSON dictionary.")


if "WSS5" not in data:
    raise RuntimeError(
        "WSS5 dataset not found inside extracted RBI JSON."
    )


records = data["WSS5"]

if not isinstance(records, list):
    raise RuntimeError("WSS5 is not a list of records.")


print()
print("Total WSS5 records:", len(records))


# =============================================================================
# FILTER POLICY REPO RATE
# =============================================================================

repo = []

for record in records:

    description = str(
        record.get("wss_desc", "")
    ).strip().lower()

    if description == "policy repo rate":
        repo.append(record)


print("Policy Repo Rate records:", len(repo))


if not repo:
    raise RuntimeError(
        "No records with wss_desc = 'Policy Repo Rate' were found."
    )


# =============================================================================
# BUILD DATAFRAME
# =============================================================================

rows = []

for record in repo:

    rows.append(
        {
            "time_date": record.get("time_date"),
            "year": record.get("time_cal_year"),
            "month": record.get("month_time_date"),
            "repo_rate": record.get("wss_amount"),
            "item_code": record.get("item_code"),
            "description": record.get("wss_desc"),
        }
    )


df = pd.DataFrame(rows)


# =============================================================================
# CLEAN VALUES
# =============================================================================

df["repo_rate"] = pd.to_numeric(
    df["repo_rate"],
    errors="coerce"
)


# RBI time_date is JavaScript Unix timestamp in milliseconds

df["Date"] = pd.to_datetime(
    df["time_date"],
    unit="ms",
    errors="coerce"
)


# Keep only usable records

df = df.dropna(
    subset=["Date", "repo_rate"]
)


df = (
    df
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# =============================================================================
# REORDER COLUMNS
# =============================================================================

df = df[
    [
        "Date",
        "repo_rate",
        "year",
        "month",
        "item_code",
        "description",
    ]
]


# =============================================================================
# VALIDATION
# =============================================================================

print()
print("=" * 80)
print("VALIDATION")
print("=" * 80)

print()
print("Date range:")
print("  Start:", df["Date"].min().date())
print("  End  :", df["Date"].max().date())

print()
print("Rows:", len(df))

print()
print("Unique repo-rate values:")

for value in sorted(df["repo_rate"].unique()):
    print(f"  {value:.2f}")


# =============================================================================
# SAMPLE RECORDS
# =============================================================================

print()
print("First 20 records:")
print(
    df.head(20).to_string(index=False)
)

print()
print("Last 30 records:")
print(
    df.tail(30).to_string(index=False)
)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
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
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("RBI Policy Repo Rate extraction completed successfully.")