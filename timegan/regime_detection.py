"""
MacroStress-GAN
Historical Market Regime Detection

Purpose:
    Analyze historical Indian financial-market conditions and identify
    interpretable market regimes before designing a regime-conditioned
    TimeGAN.

This script DOES NOT modify:
    - TimeGAN V3.2 model
    - V3.2 scaler
    - V3.2 synthetic data
    - training data

Output:
    outputs/validation/regime_analysis/
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_stress_5vars_processed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "regime_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

WINDOW = 30

# Rolling thresholds
VOLATILITY_QUANTILE = 0.75
STRESS_QUANTILE = 0.90
CRISIS_QUANTILE = 0.97


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MacroStress-GAN - Historical Market Regime Detection")
print("=" * 70)

print("\n[1] Loading historical data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# DATE HANDLING
# ============================================================

date_candidates = [
    "Date",
    "date",
    "DATE",
]

date_column = None

for candidate in date_candidates:
    if candidate in df.columns:
        date_column = candidate
        break

if date_column is None:
    raise ValueError(
        "Could not find a Date column."
    )

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df = df.sort_values(date_column).reset_index(drop=True)


# ============================================================
# VALIDATE FEATURES
# ============================================================

print("\n[2] Validating required features...")

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

df[FEATURES] = df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna(
    subset=FEATURES + [date_column]
).reset_index(drop=True)

print(f"Clean observations: {len(df):,}")


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n[3] Historical feature statistics...")

stats = df[FEATURES].describe().T

stats["abs_mean"] = df[FEATURES].abs().mean()

print(
    stats[
        [
            "mean",
            "std",
            "min",
            "max",
            "abs_mean",
        ]
    ].to_string()
)


# ============================================================
# ROLLING MARKET MEASURES
# ============================================================

print("\n[4] Calculating rolling market conditions...")

returns = df[FEATURES].copy()

# Rolling standard deviation
rolling_vol = returns.rolling(
    WINDOW,
    min_periods=WINDOW
).std()

# Average absolute movement
rolling_abs_move = returns.abs().rolling(
    WINDOW,
    min_periods=WINDOW
).mean()

# Market-wide volatility score
volatility_score = (
    rolling_vol / returns.std()
).mean(axis=1)

# Market-wide absolute movement score
movement_score = (
    rolling_abs_move / returns.abs().mean()
).mean(axis=1)


# ============================================================
# NIFTY DRAWDOWN
# ============================================================

print("\n[5] Calculating NIFTY drawdown...")

nifty_return = df["NIFTY50_Return"]

nifty_equity = (
    1.0 + nifty_return.fillna(0.0)
).cumprod()

nifty_running_max = nifty_equity.cummax()

nifty_drawdown = (
    nifty_equity / nifty_running_max
) - 1.0

rolling_nifty_drawdown = nifty_drawdown.rolling(
    WINDOW,
    min_periods=WINDOW
).min()


# ============================================================
# VIX STRESS
# ============================================================

print("\n[6] Calculating VIX stress...")

vix_change_abs = (
    df["INDIA_VIX_Change"]
    .abs()
)

rolling_vix_stress = vix_change_abs.rolling(
    WINDOW,
    min_periods=WINDOW
).mean()


# ============================================================
# COMPOSITE STRESS SCORE
# ============================================================

print("\n[7] Building composite stress score...")

# Normalize each component using its historical distribution.
def robust_zscore(series):
    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0 or not np.isfinite(mad):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - median) / (
        1.4826 * mad
    )


vol_z = robust_zscore(volatility_score)
move_z = robust_zscore(movement_score)
dd_z = robust_zscore(
    -rolling_nifty_drawdown
)

vix_z = robust_zscore(
    rolling_vix_stress
)

# Composite stress score
stress_score = (
    0.35 * vol_z
    + 0.25 * move_z
    + 0.25 * dd_z
    + 0.15 * vix_z
)


# ============================================================
# THRESHOLDS
# ============================================================

valid_stress = stress_score.dropna()

vol_threshold = valid_stress.quantile(
    VOLATILITY_QUANTILE
)

stress_threshold = valid_stress.quantile(
    STRESS_QUANTILE
)

crisis_threshold = valid_stress.quantile(
    CRISIS_QUANTILE
)

print("\nStress thresholds:")
print(
    f"Elevated volatility threshold : "
    f"{vol_threshold:.4f}"
)

print(
    f"Market stress threshold       : "
    f"{stress_threshold:.4f}"
)

print(
    f"Crisis threshold              : "
    f"{crisis_threshold:.4f}"
)


# ============================================================
# REGIME LABELING
# ============================================================

print("\n[8] Assigning market regimes...")


def assign_regime(score):
    if pd.isna(score):
        return "Insufficient History"

    if score >= crisis_threshold:
        return "Crisis"

    if score >= stress_threshold:
        return "Market Stress"

    if score >= vol_threshold:
        return "Elevated Volatility"

    return "Normal"


df["Rolling_Volatility_Score"] = volatility_score

df["Rolling_Movement_Score"] = movement_score

df["NIFTY_Drawdown_30D"] = rolling_nifty_drawdown

df["VIX_Stress_30D"] = rolling_vix_stress

df["Stress_Score"] = stress_score

df["Regime"] = df["Stress_Score"].apply(
    assign_regime
)


# ============================================================
# REGIME SUMMARY
# ============================================================

print("\n[9] Regime distribution...")

regime_summary = (
    df["Regime"]
    .value_counts()
    .rename_axis("Regime")
    .reset_index(name="Observations")
)

regime_summary["Percentage"] = (
    regime_summary["Observations"]
    / len(df)
    * 100
)

print(
    regime_summary.to_string(
        index=False,
        formatters={
            "Percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# REGIME FEATURE STATISTICS
# ============================================================

print("\n[10] Regime characteristics...")

regime_stats = (
    df.groupby("Regime")[FEATURES]
    .agg(["mean", "std"])
)

print(regime_stats.to_string())


# ============================================================
# REGIME DATE RANGES
# ============================================================

print("\n[11] Regime date ranges...")

date_ranges = (
    df[df["Regime"] != "Insufficient History"]
    .groupby("Regime")[date_column]
    .agg(["min", "max", "count"])
    .sort_values("min")
)

print(
    date_ranges.to_string()
)


# ============================================================
# TOP STRESS PERIODS
# ============================================================

print("\n[12] Top historical stress observations...")

top_stress = (
    df[
        [
            date_column,
            "Stress_Score",
            "Regime",
            "NIFTY50_Return",
            "CRUDE_OIL_Return",
            "USD_INR_Return",
            "INDIA_VIX_Change",
            "INDIA_10Y_YIELD_Change",
            "NIFTY_Drawdown_30D",
        ]
    ]
    .dropna(subset=["Stress_Score"])
    .sort_values(
        "Stress_Score",
        ascending=False
    )
    .head(30)
)

print(
    top_stress.to_string(index=False)
)


# ============================================================
# CONTIGUOUS REGIME EPISODES
# ============================================================

print("\n[13] Detecting contiguous regime episodes...")

regime_series = df["Regime"].tolist()

episodes = []

start_idx = 0
current_regime = regime_series[0]

for i in range(1, len(regime_series)):

    if regime_series[i] != current_regime:

        end_idx = i - 1

        if current_regime != "Insufficient History":

            episodes.append(
                {
                    "Regime": current_regime,
                    "Start_Date": df.loc[
                        start_idx,
                        date_column
                    ],
                    "End_Date": df.loc[
                        end_idx,
                        date_column
                    ],
                    "Observations": (
                        end_idx - start_idx + 1
                    ),
                    "Max_Stress": df.loc[
                        start_idx:end_idx,
                        "Stress_Score"
                    ].max(),
                    "Min_NIFTY_Return": df.loc[
                        start_idx:end_idx,
                        "NIFTY50_Return"
                    ].min(),
                }
            )

        start_idx = i
        current_regime = regime_series[i]

# Final episode
end_idx = len(regime_series) - 1

if current_regime != "Insufficient History":

    episodes.append(
        {
            "Regime": current_regime,
            "Start_Date": df.loc[
                start_idx,
                date_column
            ],
            "End_Date": df.loc[
                end_idx,
                date_column
            ],
            "Observations": (
                end_idx - start_idx + 1
            ),
            "Max_Stress": df.loc[
                start_idx:end_idx,
                "Stress_Score"
            ].max(),
            "Min_NIFTY_Return": df.loc[
                start_idx:end_idx,
                "NIFTY50_Return"
            ].min(),
        }
    )

episodes_df = pd.DataFrame(episodes)

if not episodes_df.empty:

    episodes_df = episodes_df.sort_values(
        "Start_Date"
    ).reset_index(drop=True)

    print(
        episodes_df.to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[14] Saving outputs...")

LABELED_DATA_PATH = (
    OUTPUT_DIR
    / "historical_regime_labels.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "regime_summary.csv"
)

STATS_PATH = (
    OUTPUT_DIR
    / "regime_feature_statistics.csv"
)

EPISODES_PATH = (
    OUTPUT_DIR
    / "regime_episodes.csv"
)

STRESS_PATH = (
    OUTPUT_DIR
    / "top_stress_periods.csv"
)


df.to_csv(
    LABELED_DATA_PATH,
    index=False
)

regime_summary.to_csv(
    SUMMARY_PATH,
    index=False
)

# Flatten multi-index columns
flat_stats = (
    regime_stats.copy()
)

flat_stats.columns = [
    f"{feature}_{stat}"
    for feature, stat in flat_stats.columns
]

flat_stats.reset_index().to_csv(
    STATS_PATH,
    index=False
)

episodes_df.to_csv(
    EPISODES_PATH,
    index=False
)

top_stress.to_csv(
    STRESS_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REGIME ANALYSIS COMPLETED")
print("=" * 70)

print(f"Historical observations : {len(df):,}")
print(f"Regime labels           : {LABELED_DATA_PATH}")
print(f"Regime summary          : {SUMMARY_PATH}")
print(f"Feature statistics      : {STATS_PATH}")
print(f"Regime episodes         : {EPISODES_PATH}")
print(f"Top stress periods      : {STRESS_PATH}")

print("\nNext research step:")
print(
    "Review the regime distribution and historical episodes "
    "before designing regime-conditioned TimeGAN V3.4."
)

print("=" * 70)