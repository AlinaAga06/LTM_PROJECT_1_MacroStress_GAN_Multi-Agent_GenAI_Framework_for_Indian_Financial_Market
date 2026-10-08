import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================================================
# Paths
# =========================================================

INPUT_FILE = Path("../data/processed/macro_stress_5vars_processed.csv")
PLOT_DIR = Path("plots_5vars")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load dataset
# =========================================================

print("=" * 70)
print("MACRO STRESS - 5 VARIABLE EDA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df["Date"].min().date(), "→", df["Date"].max().date())

print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# Missing values
# =========================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(df.isnull().sum())

# =========================================================
# Descriptive statistics
# =========================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

price_columns = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_10Y_YIELD"
]

print(df[price_columns].describe())

# =========================================================
# 1. NIFTY 50
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["NIFTY50"])
plt.title("NIFTY 50")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_nifty50_price.png", dpi=300)
plt.close()

# =========================================================
# 2. India VIX
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["INDIA_VIX"])
plt.title("India VIX")
plt.xlabel("Date")
plt.ylabel("VIX")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_india_vix.png", dpi=300)
plt.close()

# =========================================================
# 3. Crude Oil
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["CRUDE_OIL"])
plt.title("WTI Crude Oil Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_crude_oil.png", dpi=300)
plt.close()

# =========================================================
# 4. USD/INR
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["USD_INR"])
plt.title("USD/INR Exchange Rate")
plt.xlabel("Date")
plt.ylabel("INR per USD")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_usd_inr.png", dpi=300)
plt.close()

# =========================================================
# 5. India 10-Year Government Bond Yield
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["INDIA_10Y_YIELD"])
plt.title("India 10-Year Government Bond Yield")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_india_10y_yield.png", dpi=300)
plt.close()

# =========================================================
# 6. NIFTY Returns
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["NIFTY50_Return"])
plt.title("NIFTY 50 Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.axhline(0, linewidth=0.8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_nifty_returns.png", dpi=300)
plt.close()

# =========================================================
# 7. Rolling Volatility
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["NIFTY50_Volatility_30D"],
    label="NIFTY 50"
)

plt.plot(
    df["Date"],
    df["CRUDE_OIL_Volatility_30D"],
    label="Crude Oil"
)

plt.plot(
    df["Date"],
    df["USD_INR_Volatility_30D"],
    label="USD/INR"
)

plt.title("30-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_rolling_volatility.png", dpi=300)
plt.close()

# =========================================================
# 8. Correlation Matrix
# =========================================================

correlation_columns = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

correlation_matrix = df[correlation_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix - Macro Stress Variables")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_correlation_matrix.png", dpi=300)
plt.close()

# =========================================================
# 9. NIFTY Return Distribution
# =========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["NIFTY50_Return"],
    bins=100,
    kde=True
)

plt.title("Distribution of NIFTY 50 Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_nifty_return_distribution.png", dpi=300)
plt.close()

# =========================================================
# 10. Crude Return Distribution
# =========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["CRUDE_OIL_Return"],
    bins=100,
    kde=True
)

plt.title("Distribution of Crude Oil Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_DIR / "10_crude_return_distribution.png", dpi=300)
plt.close()

# =========================================================
# 11. India VIX vs NIFTY Returns
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["NIFTY50_Return"],
    df["INDIA_VIX_Change"],
    alpha=0.3
)

plt.title("India VIX Change vs NIFTY 50 Return")
plt.xlabel("NIFTY 50 Return")
plt.ylabel("India VIX Change")
plt.tight_layout()
plt.savefig(PLOT_DIR / "11_vix_vs_nifty.png", dpi=300)
plt.close()

# =========================================================
# 12. 10Y Yield Change vs NIFTY Returns
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["NIFTY50_Return"],
    df["INDIA_10Y_YIELD_Change"],
    alpha=0.3
)

plt.title("10-Year G-Sec Yield Change vs NIFTY 50 Return")
plt.xlabel("NIFTY 50 Return")
plt.ylabel("10Y Yield Change (percentage points)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "12_10y_yield_vs_nifty.png", dpi=300)
plt.close()

# =========================================================
# 13. 10-Year Yield Over Time
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["INDIA_10Y_YIELD"]
)

plt.title("India 10-Year Government Bond Yield Over Time")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "13_10y_yield_history.png", dpi=300)
plt.close()

# =========================================================
# Stress-period statistics
# =========================================================

print("\n" + "=" * 70)
print("STRESS PERIOD ANALYSIS")
print("=" * 70)

# Lowest NIFTY returns
print("\n5 Largest Negative NIFTY Returns:")
print(
    df[
        ["Date", "NIFTY50_Return"]
    ]
    .sort_values("NIFTY50_Return")
    .head(5)
    .to_string(index=False)
)

# Highest VIX
print("\n5 Highest India VIX Observations:")
print(
    df[
        ["Date", "INDIA_VIX"]
    ]
    .sort_values("INDIA_VIX", ascending=False)
    .head(5)
    .to_string(index=False)
)

# Highest crude oil
print("\n5 Highest Crude Oil Prices:")
print(
    df[
        ["Date", "CRUDE_OIL"]
    ]
    .sort_values("CRUDE_OIL", ascending=False)
    .head(5)
    .to_string(index=False)
)

# Highest 10Y yield
print("\n5 Highest India 10Y Yield Observations:")
print(
    df[
        ["Date", "INDIA_10Y_YIELD"]
    ]
    .sort_values("INDIA_10Y_YIELD", ascending=False)
    .head(5)
    .to_string(index=False)
)

# =========================================================
# Final validation
# =========================================================

print("\n" + "=" * 70)
print("FINAL EDA VALIDATION")
print("=" * 70)

print("\nShape:", df.shape)

print(
    "Date:",
    df["Date"].min().date(),
    "→",
    df["Date"].max().date()
)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of plots generated:")
print(len(list(PLOT_DIR.glob("*.png"))))

print("\nPlots saved in:")
print(PLOT_DIR)

print("\n" + "=" * 70)
print("5-VARIABLE EDA COMPLETED")
print("=" * 70)