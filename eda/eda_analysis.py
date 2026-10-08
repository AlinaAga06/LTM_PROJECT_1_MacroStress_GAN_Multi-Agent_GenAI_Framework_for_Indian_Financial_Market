import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("data/processed/macro_stress_processed.csv")
PLOT_DIR = Path("eda/plots")

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load processed dataset
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

print("=" * 70)
print("MACROSTRESS-GAN - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# Basic statistics
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

price_columns = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR"
]

print(df[price_columns].describe())


# ============================================================
# 1. NIFTY 50 Price Trend
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["NIFTY50"]
)

plt.title("NIFTY 50 Price Trend (2008–2025)")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 Index")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "01_nifty50_price.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. India VIX Trend
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["INDIA_VIX"]
)

plt.title("India VIX Trend (2008–2025)")
plt.xlabel("Date")
plt.ylabel("India VIX")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "02_india_vix.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. Crude Oil Price Trend
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["CRUDE_OIL"]
)

plt.title("Crude Oil Price Trend (2008–2025)")
plt.xlabel("Date")
plt.ylabel("Crude Oil Price")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "03_crude_oil.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. USD/INR Trend
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["USD_INR"]
)

plt.title("USD/INR Exchange Rate Trend (2008–2025)")
plt.xlabel("Date")
plt.ylabel("USD/INR")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "04_usd_inr.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. NIFTY Daily Returns
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["NIFTY50_Return"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("NIFTY 50 Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "05_nifty_returns.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. Rolling Volatility
# ============================================================

plt.figure(figsize=(14, 6))

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
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "06_rolling_volatility.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. Correlation Matrix
# ============================================================

return_columns = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change"
]

correlation = df[return_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation)


plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Market Movements")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "07_correlation_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. NIFTY Return Distribution
# ============================================================

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

plt.savefig(
    PLOT_DIR / "08_nifty_return_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. Crude Oil Return Distribution
# ============================================================

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

plt.savefig(
    PLOT_DIR / "09_crude_return_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. India VIX vs NIFTY Returns
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["INDIA_VIX_Change"],
    df["NIFTY50_Return"],
    alpha=0.4
)

plt.title("India VIX Change vs NIFTY 50 Returns")
plt.xlabel("India VIX Change")
plt.ylabel("NIFTY 50 Return")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "10_vix_vs_nifty.png",
    dpi=300
)

plt.show()


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("\nPlots saved in:")
print(PLOT_DIR)

print("\nGenerated plots:")

for file in sorted(PLOT_DIR.glob("*.png")):
    print("-", file.name)