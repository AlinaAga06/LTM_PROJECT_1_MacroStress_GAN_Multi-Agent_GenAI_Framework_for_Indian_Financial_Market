import yfinance as yf
from pathlib import Path

# Create data/raw folder
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Yahoo Finance tickers
tickers = {
    "NIFTY50": "^NSEI",
    "INDIA_VIX": "^INDIAVIX",
    "CRUDE_OIL": "CL=F",
    "USD_INR": "USDINR=X"
}

start_date = "2008-01-01"
end_date = "2026-01-01"

for name, ticker in tickers.items():

    print(f"\nDownloading {name} ({ticker})...")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=True
    )

    if data.empty:
        print(f"❌ No data found for {name}")
        continue

    file_path = f"data/raw/{name}.csv"

    data.to_csv(file_path)

    print(f"✅ Saved: {file_path}")
    print(f"Rows: {len(data)}")
    print(f"From: {data.index.min()}")
    print(f"To:   {data.index.max()}")

print("\n✅ Download completed!")