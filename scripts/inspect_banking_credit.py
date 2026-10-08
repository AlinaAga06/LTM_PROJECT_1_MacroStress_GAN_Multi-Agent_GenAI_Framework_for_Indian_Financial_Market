import requests
import json

URL = (
    "https://data-api.dbie.rbihub.in/api/tables/"
    "financial_sector/"
    "r152_deployment_of_bank_credit_by_major_sectors/"
    "rows"
)

print("=" * 80)
print("RBI DBIE — BANK CREDIT TABLE INSPECTION")
print("=" * 80)

response = requests.get(URL, timeout=120)

print("HTTP Status:", response.status_code)

response.raise_for_status()

data = response.json()

print("\nTop-level keys:")
for key, value in data.items():
    if isinstance(value, list):
        print(f"  {key}: list ({len(value)} items)")
    elif isinstance(value, dict):
        print(f"  {key}: dict")
    else:
        print(f"  {key}: {value}")

# ------------------------------------------------------------
# Correct API field
# ------------------------------------------------------------

rows = data["rows"]

print("\nRows returned:", len(rows))

print("\nColumns:")
for i, col in enumerate(data.get("columns", [])):
    print(f"  [{i}] {col}")

# ------------------------------------------------------------
# Inspect rows
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST 10 ROWS")
print("=" * 80)

for i, row in enumerate(rows[:10]):
    print(f"\nROW {i}")
    print(row)

# ------------------------------------------------------------
# Inspect table metadata
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TABLE METADATA")
print("=" * 80)

print(json.dumps(
    data.get("table", {}),
    indent=2,
    ensure_ascii=False
))

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)