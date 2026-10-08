"""
MacroStress-GAN
Real Market Data Collector

Downloads real historical market data for the
institution-specific datasets.

Sources:
    Yahoo Finance:
        NIFTY 50
        Bank NIFTY
        S&P 500
        VIX
        USD/INR
        Crude Oil
        Brent Crude
        Gold
        Silver
        Natural Gas

    FRED:
        US 10-Year Treasury Yield (DGS10)

Official Indian macroeconomic variables such as:
    Repo Rate
    CPI
    Credit Growth
    WPI
    India 10Y Yield

should be supplied from RBI/NSE/official downloaded files
rather than fabricated or substituted.
"""

from pathlib import Path
import io

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

STOCK_DIR = RAW_ROOT / "stock"
BANKING_DIR = RAW_ROOT / "banking"
CORPORATE_DIR = RAW_ROOT / "corporate"
GLOBAL_DIR = RAW_ROOT / "global"
COMMODITY_DIR = RAW_ROOT / "commodity"

for directory in [
    STOCK_DIR,
    BANKING_DIR,
    CORPORATE_DIR,
    GLOBAL_DIR,
    COMMODITY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATE RANGE
# ============================================================

START_DATE = "2008-01-01"
END_DATE = None


# ============================================================
# YAHOO FINANCE TICKERS
# ============================================================

TICKERS = {

    # India
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "USD_INR": "INR=X",

    # India VIX
    "INDIA_VIX": "^INDIAVIX",

    # US
    "SP500": "^GSPC",
    "US_VIX": "^VIX",

    # Energy
    "CRUDE_OIL": "CL=F",
    "BRENT_CRUDE": "BZ=F",

    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "NATURAL_GAS": "NG=F",
}


# ============================================================
# DOWNLOAD ONE SERIES
# ============================================================

def download_series(
    name: str,
    ticker: str,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print(f"Downloading: {name}")
    print(f"Ticker:     {ticker}")
    print("=" * 70)

    try:

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )

    except Exception as exc:

        print(f"ERROR: {exc}")
        return pd.DataFrame()

    if data is None or data.empty:

        print("WARNING: No data returned.")

        return pd.DataFrame()

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if "Close" in data.columns.get_level_values(0):

            close = data["Close"]

            if isinstance(
                close,
                pd.DataFrame,
            ):

                close = close.iloc[:, 0]

        elif "Adj Close" in data.columns.get_level_values(0):

            close = data["Adj Close"]

            if isinstance(
                close,
                pd.DataFrame,
            ):

                close = close.iloc[:, 0]

        else:

            close = data.iloc[:, 0]

    else:

        if "Close" in data.columns:

            close = data["Close"]

        elif "Adj Close" in data.columns:

            close = data["Adj Close"]

        else:

            close = data.iloc[:, 0]

    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                close.index
            ),
            name: pd.to_numeric(
                close.values,
                errors="coerce",
            ),
        }
    )

    result = result.dropna()

    result["Date"] = (
        result["Date"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    result = (
        result
        .drop_duplicates(
            subset=["Date"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"Rows:       {len(result):,}"
    )

    print(
        f"Start:      {result['Date'].min().date()}"
    )

    print(
        f"End:        {result['Date'].max().date()}"
    )

    return result


# ============================================================
# SAVE SERIES
# ============================================================

def save_series(
    dataframe: pd.DataFrame,
    name: str,
):

    if dataframe.empty:

        print(
            f"Skipping {name}: empty dataframe."
        )

        return

    path = RAW_ROOT / "individual"

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        path /
        f"{name.lower()}.csv"
    )

    dataframe.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# DOWNLOAD FRED DGS10
# ============================================================

def download_fred_dgs10():

    print()
    print("=" * 70)
    print("Downloading: US 10Y Treasury Yield")
    print("FRED series: DGS10")
    print("=" * 70)

    url = (
        "https://fred.stlouisfed.org/"
        "graph/fredgraph.csv"
        "?id=DGS10"
    )

    try:

        response = requests.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

    except Exception as exc:

        print(
            f"ERROR downloading FRED DGS10: {exc}"
        )

        return pd.DataFrame()

    data = pd.read_csv(
        io.StringIO(
            response.text
        )
    )

    data["DATE"] = pd.to_datetime(
        data["DATE"],
        errors="coerce",
    )

    data["DGS10"] = pd.to_numeric(
        data["DGS10"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "DATE",
            "DGS10",
        ]
    )

    data = data.rename(
        columns={
            "DATE": "Date",
            "DGS10": "US_10Y_YIELD",
        }
    )

    data = data[
        data["Date"]
        >= pd.Timestamp(START_DATE)
    ]

    data = data.reset_index(
        drop=True
    )

    print(
        f"Rows:       {len(data):,}"
    )

    print(
        f"Start:      {data['Date'].min().date()}"
    )

    print(
        f"End:        {data['Date'].max().date()}"
    )

    output = (
        RAW_ROOT /
        "individual" /
        "us_10y_yield.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    return data


# ============================================================
# DOWNLOAD ALL
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("REAL MARKET DATA COLLECTION")
    print("=" * 78)

    downloaded = {}

    for name, ticker in TICKERS.items():

        data = download_series(
            name,
            ticker,
        )

        downloaded[name] = data

        save_series(
            data,
            name,
        )

    downloaded[
        "US_10Y_YIELD"
    ] = download_fred_dgs10()

    print()
    print("=" * 78)
    print("DOWNLOAD SUMMARY")
    print("=" * 78)

    for name, data in downloaded.items():

        if data is None or data.empty:

            print(
                f"{name:20s} FAILED"
            )

        else:

            print(
                f"{name:20s} "
                f"{len(data):8,d} rows "
                f"{data['Date'].min().date()} "
                f"-> "
                f"{data['Date'].max().date()}"
            )

    print()
    print(
        "Raw individual files are in:"
    )

    print(
        RAW_ROOT / "individual"
    )


if __name__ == "__main__":
    main()