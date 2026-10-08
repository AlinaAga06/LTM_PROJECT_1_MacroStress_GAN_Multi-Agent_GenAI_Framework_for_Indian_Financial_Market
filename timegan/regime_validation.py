from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
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
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30

REGIMES = [
    "Normal",
    "Elevated Volatility",
    "Market Stress",
    "Crisis",
]

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def safe_float(value):
    if pd.isna(value):
        return np.nan
    return float(value)


# ============================================================
# LOAD DATA
# ============================================================

print_section("MACROSTRESS-GAN REGIME VALIDATION")

print(f"Input: {INPUT_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Regime label file not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Dataset shape: {df.shape}")

# Exclude the initial rolling-history period
analysis_df = df[
    df["Regime"].isin(REGIMES)
].copy()

analysis_df = analysis_df.sort_values("Date").reset_index(drop=True)

print(f"Analysis observations: {len(analysis_df)}")

print("\nRegime counts:")
print(
    analysis_df["Regime"]
    .value_counts()
    .reindex(REGIMES)
    .to_string()
)


# ============================================================
# 1. TRANSITION MATRIX
# ============================================================

print_section("1. REGIME TRANSITION MATRIX")

analysis_df["Next_Regime"] = analysis_df["Regime"].shift(-1)

transition_counts = pd.crosstab(
    analysis_df["Regime"],
    analysis_df["Next_Regime"],
).reindex(
    index=REGIMES,
    columns=REGIMES,
    fill_value=0,
)

transition_matrix = (
    transition_counts
    .div(transition_counts.sum(axis=1), axis=0)
    .fillna(0)
)

transition_matrix.to_csv(
    OUTPUT_DIR / "regime_transition_matrix.csv"
)

print("\nTransition probabilities:")
print(
    transition_matrix
    .round(4)
    .to_string()
)

print("\nSelf-persistence:")

for regime in REGIMES:
    persistence = transition_matrix.loc[regime, regime]
    print(f"{regime:<22}: {persistence:.4f}")


# ============================================================
# 2. EPISODE DURATION STATISTICS
# ============================================================

print_section("2. REGIME DURATION ANALYSIS")

episode_path = OUTPUT_DIR / "regime_episodes.csv"

if not episode_path.exists():
    raise FileNotFoundError(
        f"Episode file not found:\n{episode_path}"
    )

episodes = pd.read_csv(episode_path)

duration_stats = (
    episodes
    .groupby("Regime")["Observations"]
    .agg(
        Episodes="count",
        Minimum="min",
        Median="median",
        Mean="mean",
        Maximum="max",
    )
    .reindex(REGIMES)
)

duration_stats["Short_Episodes_<=2"] = (
    episodes.assign(
        Short=episodes["Observations"] <= 2
    )
    .groupby("Regime")["Short"]
    .sum()
    .reindex(REGIMES)
    .fillna(0)
)

duration_stats.to_csv(
    OUTPUT_DIR / "regime_duration_statistics.csv"
)

print(duration_stats.round(2).to_string())


# ============================================================
# 3. OBSERVATION COUNTS
# ============================================================

print_section("3. REGIME OBSERVATION COUNTS")

observation_counts = (
    analysis_df["Regime"]
    .value_counts()
    .reindex(REGIMES)
    .fillna(0)
    .astype(int)
)

observation_percentages = (
    observation_counts / observation_counts.sum() * 100
)

conditioning_summary = pd.DataFrame(
    {
        "Observations": observation_counts,
        "Percentage": observation_percentages,
    }
)

print(conditioning_summary.round(3).to_string())


# ============================================================
# 4. 30-DAY CONDITIONING WINDOWS
# ============================================================

print_section("4. 30-DAY CONDITIONING WINDOW ANALYSIS")

# A sequence is assigned the regime of its final observation.
# This is the simplest regime-conditioning formulation and
# preserves the historical ordering.

window_records = []

values = analysis_df["Regime"].to_numpy()

for end_idx in range(SEQUENCE_LENGTH - 1, len(values)):

    window = values[
        end_idx - SEQUENCE_LENGTH + 1:
        end_idx + 1
    ]

    final_regime = window[-1]

    counts = pd.Series(window).value_counts()

    dominant_regime = counts.idxmax()
    dominant_fraction = counts.max() / SEQUENCE_LENGTH

    window_records.append(
        {
            "End_Date": analysis_df.iloc[end_idx]["Date"],
            "Final_Regime": final_regime,
            "Dominant_Regime": dominant_regime,
            "Dominant_Fraction": dominant_fraction,
            "Window_Regime_Changes": int(
                np.sum(window[1:] != window[:-1])
            ),
        }
    )

windows = pd.DataFrame(window_records)

window_summary = (
    windows
    .groupby("Final_Regime")
    .agg(
        Windows=("Final_Regime", "size"),
        Mean_Dominant_Fraction=("Dominant_Fraction", "mean"),
        Median_Dominant_Fraction=("Dominant_Fraction", "median"),
        Mean_Regime_Changes=("Window_Regime_Changes", "mean"),
    )
    .reindex(REGIMES)
)

window_summary.to_csv(
    OUTPUT_DIR / "regime_conditioning_summary.csv"
)

print(window_summary.round(4).to_string())


# ============================================================
# 5. FEATURE STATISTICS
# ============================================================

print_section("5. REGIME FEATURE SEPARATION")

feature_rows = []

for feature in FEATURES:

    overall_std = analysis_df[feature].std()

    for regime in REGIMES:

        values_regime = (
            analysis_df.loc[
                analysis_df["Regime"] == regime,
                feature,
            ]
            .dropna()
        )

        if len(values_regime) == 0:
            continue

        mean_value = values_regime.mean()
        std_value = values_regime.std()

        standardized_mean = (
            mean_value / overall_std
            if overall_std > 0
            else np.nan
        )

        volatility_ratio = (
            std_value / overall_std
            if overall_std > 0
            else np.nan
        )

        feature_rows.append(
            {
                "Feature": feature,
                "Regime": regime,
                "Observations": len(values_regime),
                "Mean": mean_value,
                "Std": std_value,
                "Standardized_Mean": standardized_mean,
                "Volatility_Ratio": volatility_ratio,
            }
        )

feature_separation = pd.DataFrame(feature_rows)

feature_separation.to_csv(
    OUTPUT_DIR / "regime_feature_separation.csv",
    index=False,
)

print(
    feature_separation
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 6. KRUSKAL-WALLIS REGIME SEPARABILITY
# ============================================================

print_section("6. KRUSKAL-WALLIS REGIME SEPARABILITY")

kw_rows = []

for feature in FEATURES:

    samples = []

    valid_regimes = []

    for regime in REGIMES:

        sample = (
            analysis_df.loc[
                analysis_df["Regime"] == regime,
                feature,
            ]
            .dropna()
            .values
        )

        if len(sample) > 0:
            samples.append(sample)
            valid_regimes.append(regime)

    if len(samples) >= 2:

        statistic, p_value = kruskal(*samples)

        kw_rows.append(
            {
                "Feature": feature,
                "Kruskal_Wallis_H": statistic,
                "P_Value": p_value,
                "Significant_at_0.05": p_value < 0.05,
                "Significant_at_0.01": p_value < 0.01,
            }
        )

kw_results = pd.DataFrame(kw_rows)

kw_results.to_csv(
    OUTPUT_DIR / "regime_kruskal_wallis.csv",
    index=False,
)

print(
    kw_results
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 7. CRISIS VS NORMAL SEPARATION
# ============================================================

print_section("7. CRISIS VS NORMAL COMPARISON")

comparison_rows = []

for feature in FEATURES:

    normal = (
        analysis_df.loc[
            analysis_df["Regime"] == "Normal",
            feature,
        ]
        .dropna()
    )

    crisis = (
        analysis_df.loc[
            analysis_df["Regime"] == "Crisis",
            feature,
        ]
        .dropna()
    )

    normal_mean = normal.mean()
    crisis_mean = crisis.mean()

    normal_std = normal.std()
    crisis_std = crisis.std()

    pooled_std = np.sqrt(
        (
            (len(normal) - 1) * normal_std ** 2
            + (len(crisis) - 1) * crisis_std ** 2
        )
        /
        (
            len(normal)
            + len(crisis)
            - 2
        )
    )

    if pooled_std > 0:
        cohens_d = (
            crisis_mean - normal_mean
        ) / pooled_std
    else:
        cohens_d = np.nan

    comparison_rows.append(
        {
            "Feature": feature,
            "Normal_Mean": normal_mean,
            "Crisis_Mean": crisis_mean,
            "Normal_Std": normal_std,
            "Crisis_Std": crisis_std,
            "Cohens_D_Crisis_vs_Normal": cohens_d,
        }
    )

crisis_normal = pd.DataFrame(comparison_rows)

crisis_normal.to_csv(
    OUTPUT_DIR / "crisis_vs_normal_comparison.csv",
    index=False,
)

print(
    crisis_normal
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 8. CONDITIONING SUFFICIENCY
# ============================================================

print_section("8. CONDITIONING SUFFICIENCY")

sufficiency_rows = []

for regime in REGIMES:

    obs_count = int(
        observation_counts.loc[regime]
    )

    episode_count = int(
        duration_stats.loc[regime, "Episodes"]
    )

    window_count = int(
        window_summary.loc[regime, "Windows"]
    )

    mean_duration = float(
        duration_stats.loc[regime, "Mean"]
    )

    median_duration = float(
        duration_stats.loc[regime, "Median"]
    )

    # This is a descriptive flag, not a statistical guarantee.
    if obs_count >= 500:
        assessment = "Strong"
    elif obs_count >= 200:
        assessment = "Adequate"
    elif obs_count >= 100:
        assessment = "Limited"
    else:
        assessment = "Very Limited"

    sufficiency_rows.append(
        {
            "Regime": regime,
            "Observations": obs_count,
            "Episodes": episode_count,
            "30D_Windows": window_count,
            "Mean_Duration": mean_duration,
            "Median_Duration": median_duration,
            "Data_Sufficiency": assessment,
        }
    )

sufficiency = pd.DataFrame(
    sufficiency_rows
)

print(
    sufficiency
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 9. AUTOMATED RESEARCH ASSESSMENT
# ============================================================

print_section("9. REGIME VALIDATION ASSESSMENT")

min_persistence = (
    transition_matrix
    .loc[REGIMES, REGIMES]
    .values
    .diagonal()
    .min()
)

significant_features = int(
    kw_results["Significant_at_0.01"].sum()
)

crisis_observations = int(
    observation_counts["Crisis"]
)

crisis_episodes = int(
    duration_stats.loc["Crisis", "Episodes"]
)

assessment_lines = []

assessment_lines.append(
    "MacroStress-GAN Regime Validation Assessment"
)

assessment_lines.append(
    "=============================================="
)

assessment_lines.append("")

assessment_lines.append(
    f"Total analyzed observations: {len(analysis_df)}"
)

assessment_lines.append(
    f"Minimum regime self-persistence: "
    f"{min_persistence:.4f}"
)

assessment_lines.append(
    f"Features significant at p < 0.01: "
    f"{significant_features}/{len(FEATURES)}"
)

assessment_lines.append(
    f"Crisis observations: {crisis_observations}"
)

assessment_lines.append(
    f"Crisis episodes: {crisis_episodes}"
)

assessment_lines.append("")

if min_persistence >= 0.90:
    assessment_lines.append(
        "Persistence assessment: PASS - all regimes show "
        "high one-step persistence."
    )
else:
    assessment_lines.append(
        "Persistence assessment: REVIEW - at least one "
        "regime has lower persistence."
    )

if significant_features == len(FEATURES):
    assessment_lines.append(
        "Separability assessment: PASS - all five financial "
        "features show statistically significant differences "
        "across regimes at p < 0.01."
    )
else:
    assessment_lines.append(
        "Separability assessment: REVIEW - not all features "
        "are statistically separated."
    )

if crisis_observations >= 100 and crisis_episodes >= 5:
    assessment_lines.append(
        "Crisis data sufficiency: PASS for exploratory "
        "conditional modeling, subject to sequence-overlap "
        "and event-independence limitations."
    )
else:
    assessment_lines.append(
        "Crisis data sufficiency: LIMITED - conditional "
        "Crisis modeling requires additional caution."
    )

assessment_lines.append("")

assessment_lines.append(
    "Methodological note:"
)

assessment_lines.append(
    "The regime labels are heuristic stress-regime labels "
    "constructed from rolling volatility, movement, drawdown, "
    "and VIX stress measures. They should not be interpreted "
    "as latent economic states estimated by an HMM."
)

assessment_lines.append(
    "The 30-day rolling calculations create overlapping "
    "observations, so observation counts do not represent "
    "independent market events."
)

assessment_lines.append(
    "The current V3.2 TimeGAN remains the baseline model. "
    "This validation does not modify the V3.2 checkpoint."
)

report_path = (
    OUTPUT_DIR
    / "regime_validation_report.txt"
)

report_path.write_text(
    "\n".join(assessment_lines),
    encoding="utf-8",
)

print("\n".join(assessment_lines))


# ============================================================
# FINAL OUTPUT
# ============================================================

print_section("VALIDATION COMPLETE")

print("Generated files:")

output_files = [
    "regime_transition_matrix.csv",
    "regime_duration_statistics.csv",
    "regime_conditioning_summary.csv",
    "regime_feature_separation.csv",
    "regime_kruskal_wallis.csv",
    "crisis_vs_normal_comparison.csv",
    "regime_validation_report.txt",
]

for filename in output_files:
    path = OUTPUT_DIR / filename

    if path.exists():
        print(f"[OK] {filename}")

print("\nV3.2 model was not modified.")
print("No V3.4 training was performed.")