
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGIME_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "regime_analysis"
    / "historical_regime_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "regime_analysis"
    / "sequence_context"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30

REGIMES = [
    "Normal",
    "Elevated Volatility",
    "Market Stress",
    "Crisis",
]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("MacroStress-GAN 30-Day Regime Context Alignment Analysis")
print("=" * 75)

df = pd.read_csv(REGIME_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print(f"\nLoaded rows: {len(df)}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# Remove insufficient-history rows
df = df[
    df["Regime"].isin(REGIMES)
].reset_index(drop=True)

print(f"Usable rows: {len(df)}")

# ============================================================
# BUILD 30-DAY WINDOWS
# ============================================================

records = []

for end_idx in range(SEQUENCE_LENGTH - 1, len(df)):

    start_idx = end_idx - SEQUENCE_LENGTH + 1

    window = df.iloc[start_idx:end_idx + 1]

    regime_counts = (
        window["Regime"]
        .value_counts()
        .reindex(REGIMES, fill_value=0)
    )

    dominant_regime = regime_counts.idxmax()

    dominant_count = int(regime_counts.max())

    dominant_fraction = dominant_count / SEQUENCE_LENGTH

    regime_changes = int(
        (window["Regime"].values[1:]
         != window["Regime"].values[:-1]).sum()
    )

    # Final-day regime
    final_regime = window["Regime"].iloc[-1]

    # First-day regime
    initial_regime = window["Regime"].iloc[0]

    # Stress score context
    stress_scores = window["Stress_Score"].astype(float)

    mean_stress = stress_scores.mean()
    median_stress = stress_scores.median()
    final_stress = stress_scores.iloc[-1]
    maximum_stress = stress_scores.max()

    # Continuous context variables
    mean_volatility = window["Rolling_Volatility_Score"].mean()
    mean_movement = window["Rolling_Movement_Score"].mean()
    mean_drawdown = window["NIFTY_Drawdown_30D"].mean()
    mean_vix_stress = window["VIX_Stress_30D"].mean()

    records.append({
        "Sequence_End_Date": window["Date"].iloc[-1],
        "Sequence_Start_Date": window["Date"].iloc[0],

        "Initial_Regime": initial_regime,
        "Final_Regime": final_regime,
        "Dominant_Regime": dominant_regime,

        "Dominant_Count": dominant_count,
        "Dominant_Fraction": dominant_fraction,

        "Regime_Changes": regime_changes,

        "Mean_Stress_Score": mean_stress,
        "Median_Stress_Score": median_stress,
        "Final_Stress_Score": final_stress,
        "Maximum_Stress_Score": maximum_stress,

        "Mean_Volatility_Score": mean_volatility,
        "Mean_Movement_Score": mean_movement,
        "Mean_NIFTY_Drawdown": mean_drawdown,
        "Mean_VIX_Stress": mean_vix_stress,

        "Normal_Count": regime_counts["Normal"],
        "Elevated_Count": regime_counts["Elevated Volatility"],
        "Market_Stress_Count": regime_counts["Market Stress"],
        "Crisis_Count": regime_counts["Crisis"],
    })

seq = pd.DataFrame(records)

print(f"\n30-day sequences analysed: {len(seq)}")

# ============================================================
# SAVE RAW ALIGNMENT DATA
# ============================================================

alignment_path = OUTPUT_DIR / "sequence_context_alignment.csv"

seq.to_csv(
    alignment_path,
    index=False
)

# ============================================================
# 1. REGIME PURITY
# ============================================================

print("\n" + "=" * 75)
print("1. REGIME PURITY BY FINAL REGIME")
print("=" * 75)

purity_summary = (
    seq.groupby("Final_Regime")
    .agg(
        Sequences=("Final_Regime", "size"),
        Mean_Dominant_Fraction=("Dominant_Fraction", "mean"),
        Median_Dominant_Fraction=("Dominant_Fraction", "median"),
        Min_Dominant_Fraction=("Dominant_Fraction", "min"),
        Mean_Regime_Changes=("Regime_Changes", "mean"),
        Median_Regime_Changes=("Regime_Changes", "median"),
    )
    .reindex(REGIMES)
)

print(
    purity_summary.round(4).to_string()
)

purity_summary.to_csv(
    OUTPUT_DIR / "sequence_regime_purity.csv"
)

# ============================================================
# 2. FINAL VS DOMINANT REGIME
# ============================================================

print("\n" + "=" * 75)
print("2. FINAL REGIME vs DOMINANT REGIME")
print("=" * 75)

confusion = pd.crosstab(
    seq["Final_Regime"],
    seq["Dominant_Regime"],
    normalize="index"
)

confusion = confusion.reindex(
    index=REGIMES,
    columns=REGIMES,
    fill_value=0
)

print(
    confusion.round(4).to_string()
)

confusion.to_csv(
    OUTPUT_DIR / "final_vs_dominant_regime.csv"
)

# ============================================================
# 3. AGREEMENT
# ============================================================

agreement = (
    seq["Final_Regime"]
    == seq["Dominant_Regime"]
)

print("\nFinal regime == dominant regime:")
print(
    f"{agreement.sum()} / {len(seq)} "
    f"= {agreement.mean():.2%}"
)

# ============================================================
# 4. LOW-PURITY WINDOWS
# ============================================================

print("\n" + "=" * 75)
print("3. LOW-PURITY WINDOWS")
print("=" * 75)

for threshold in [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]:

    count = (
        seq["Dominant_Fraction"] < threshold
    ).sum()

    percentage = count / len(seq)

    print(
        f"Dominant fraction < {threshold:.2f}: "
        f"{count:4d} ({percentage:.2%})"
    )

# ============================================================
# 5. REGIME-SPECIFIC PURITY
# ============================================================

print("\n" + "=" * 75)
print("4. REGIME-SPECIFIC PURITY DISTRIBUTION")
print("=" * 75)

for regime in REGIMES:

    subset = seq[
        seq["Final_Regime"] == regime
    ]

    if len(subset) == 0:
        continue

    print(f"\n{regime}")
    print("-" * len(regime))

    print(
        f"Sequences: {len(subset)}"
    )

    print(
        f"Dominant fraction mean: "
        f"{subset['Dominant_Fraction'].mean():.4f}"
    )

    print(
        f"Dominant fraction median: "
        f"{subset['Dominant_Fraction'].median():.4f}"
    )

    for threshold in [0.90, 0.80, 0.70]:

        count = (
            subset["Dominant_Fraction"] >= threshold
        ).sum()

        print(
            f"Purity >= {threshold:.0%}: "
            f"{count}/{len(subset)} "
            f"({count/len(subset):.2%})"
        )

# ============================================================
# 6. STRESS CONTEXT BY FINAL REGIME
# ============================================================

print("\n" + "=" * 75)
print("5. CONTINUOUS STRESS CONTEXT")
print("=" * 75)

stress_summary = (
    seq.groupby("Final_Regime")
    [
        [
            "Mean_Stress_Score",
            "Median_Stress_Score",
            "Final_Stress_Score",
            "Maximum_Stress_Score",
            "Mean_Volatility_Score",
            "Mean_Movement_Score",
            "Mean_NIFTY_Drawdown",
            "Mean_VIX_Stress",
        ]
    ]
    .agg(["mean", "std", "median"])
    .reindex(REGIMES)
)

print(
    stress_summary.round(4).to_string()
)

stress_summary.to_csv(
    OUTPUT_DIR / "sequence_stress_context_summary.csv"
)

# ============================================================
# 7. REGIME TRANSITION WINDOWS
# ============================================================

print("\n" + "=" * 75)
print("6. TRANSITION WINDOWS")
print("=" * 75)

transition_windows = seq[
    seq["Regime_Changes"] > 0
].copy()

print(
    f"Sequences containing at least one regime transition: "
    f"{len(transition_windows)} "
    f"({len(transition_windows)/len(seq):.2%})"
)

print(
    f"Sequences with 3+ regime changes: "
    f"{(seq['Regime_Changes'] >= 3).sum()} "
    f"({(seq['Regime_Changes'] >= 3).mean():.2%})"
)

print(
    f"Sequences with 5+ regime changes: "
    f"{(seq['Regime_Changes'] >= 5).sum()} "
    f"({(seq['Regime_Changes'] >= 5).mean():.2%})"
)

# ============================================================
# 8. CRISIS WINDOWS
# ============================================================

print("\n" + "=" * 75)
print("7. CRISIS-CONTEXT WINDOWS")
print("=" * 75)

crisis_windows = seq[
    seq["Final_Regime"] == "Crisis"
].copy()

print(
    f"Crisis final-regime sequences: "
    f"{len(crisis_windows)}"
)

if len(crisis_windows) > 0:

    print(
        "\nCrisis dominant-regime distribution:"
    )

    print(
        crisis_windows["Dominant_Regime"]
        .value_counts()
        .reindex(REGIMES, fill_value=0)
        .to_string()
    )

    print(
        "\nCrisis mean context:"
    )

    print(
        crisis_windows[
            [
                "Dominant_Fraction",
                "Regime_Changes",
                "Mean_Stress_Score",
                "Final_Stress_Score",
                "Maximum_Stress_Score",
            ]
        ]
        .agg(["mean", "std", "median", "min", "max"])
        .round(4)
        .to_string()
    )

# ============================================================
# 9. PROPOSED CONDITIONING STRATEGIES
# ============================================================

print("\n" + "=" * 75)
print("8. CONDITIONING STRATEGY DIAGNOSTIC")
print("=" * 75)

final_dominant_agreement = agreement.mean()

mean_purity = seq["Dominant_Fraction"].mean()

crisis_purity = (
    crisis_windows["Dominant_Fraction"].mean()
    if len(crisis_windows) > 0
    else np.nan
)

print(
    f"Overall final/dominant agreement: "
    f"{final_dominant_agreement:.2%}"
)

print(
    f"Overall mean dominant-regime purity: "
    f"{mean_purity:.2%}"
)

print(
    f"Crisis mean dominant-regime purity: "
    f"{crisis_purity:.2%}"
)

print("\nRecommendation logic:")

if final_dominant_agreement >= 0.90:
    print(
        "FINAL-REGIME CONDITIONING: HIGH AGREEMENT"
    )
else:
    print(
        "FINAL-REGIME CONDITIONING: NEEDS CAUTION"
    )

if mean_purity >= 0.85:
    print(
        "DOMINANT-REGIME CONDITIONING: STRONG"
    )
elif mean_purity >= 0.70:
    print(
        "DOMINANT-REGIME CONDITIONING: MODERATE"
    )
else:
    print(
        "DOMINANT-REGIME CONDITIONING: WEAK"
    )

if crisis_purity >= 0.80:
    print(
        "CRISIS CONDITIONING: SUFFICIENTLY PURE"
    )
else:
    print(
        "CRISIS CONDITIONING: MIXED 30-DAY CONTEXT"
    )

# ============================================================
# 10. REPORT
# ============================================================

report_path = OUTPUT_DIR / "sequence_context_alignment_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write(
        "MacroStress-GAN 30-Day Sequence Context Alignment Report\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Sequences analysed: {len(seq)}\n"
    )

    f.write(
        f"Final/dominant agreement: "
        f"{final_dominant_agreement:.4f}\n"
    )

    f.write(
        f"Overall mean dominant purity: "
        f"{mean_purity:.4f}\n"
    )

    f.write(
        f"Crisis mean dominant purity: "
        f"{crisis_purity:.4f}\n"
    )

    f.write(
        f"Transition windows: "
        f"{len(transition_windows)}\n"
    )

    f.write(
        f"Transition window percentage: "
        f"{len(transition_windows)/len(seq):.4f}\n"
    )

    f.write("\nFinal vs Dominant Regime:\n")
    f.write(
        confusion.round(4).to_string()
    )

    f.write("\n\nPurity Summary:\n")
    f.write(
        purity_summary.round(4).to_string()
    )

print("\n" + "=" * 75)
print("ALIGNMENT ANALYSIS COMPLETE")
print("=" * 75)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {path.name}")

print("\nV3.2 model was NOT modified.")
'| Set-Content .\timegan\sequence_context_alignment.py '