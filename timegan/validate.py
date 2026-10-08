"""
======================================================================
MACROSTRESS-GAN
TIMEGAN REAL vs SYNTHETIC VALIDATION
======================================================================

Purpose:
    Compare real Indian financial market data with TimeGAN-generated
    synthetic financial data.

Validation:
    1. Basic data quality
    2. Mean comparison
    3. Standard deviation comparison
    4. Min / Max comparison
    5. Distribution similarity
    6. Correlation similarity
    7. Volatility comparison
    8. Temporal behavior comparison
    9. Statistical similarity metrics
   10. Validation report and plots

Author: MacroStress-GAN Project
======================================================================
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance, ks_2samp


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_stress_5vars_processed.csv"
)

SYNTHETIC_DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
    / "synthetic_financial_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# FEATURES
# ======================================================================

FEATURES = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_10Y_YIELD"
]


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage_difference(real_value, synthetic_value):
    """
    Calculate percentage difference relative to real value.
    """
    if real_value == 0:
        return np.nan

    return (
        abs(synthetic_value - real_value)
        / abs(real_value)
    ) * 100


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():

    print_section("LOADING REAL AND SYNTHETIC DATA")

    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Real dataset not found:\n{REAL_DATA_PATH}"
        )

    if not SYNTHETIC_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{SYNTHETIC_DATA_PATH}"
        )

    real_df = pd.read_csv(REAL_DATA_PATH)
    synthetic_df = pd.read_csv(SYNTHETIC_DATA_PATH)

    print(f"✓ Real dataset loaded")
    print(f"  Shape: {real_df.shape}")

    print(f"✓ Synthetic dataset loaded")
    print(f"  Shape: {synthetic_df.shape}")

    return real_df, synthetic_df


# ======================================================================
# BASIC DATA QUALITY
# ======================================================================

def validate_data_quality(real_df, synthetic_df):

    print_section("1. BASIC DATA QUALITY")

    print("REAL DATA")
    print(f"Rows       : {len(real_df)}")
    print(f"Columns    : {len(real_df.columns)}")
    print(
        f"Missing    : "
        f"{real_df[FEATURES].isna().sum().sum()}"
    )
    print(
        f"Infinite   : "
        f"{np.isinf(real_df[FEATURES].select_dtypes(include=np.number)).sum().sum()}"
    )

    print()

    print("SYNTHETIC DATA")
    print(f"Rows       : {len(synthetic_df)}")
    print(f"Columns    : {len(synthetic_df.columns)}")
    print(
        f"Missing    : "
        f"{synthetic_df[FEATURES].isna().sum().sum()}"
    )
    print(
        f"Infinite   : "
        f"{np.isinf(synthetic_df[FEATURES].select_dtypes(include=np.number)).sum().sum()}"
    )


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compare_statistics(real_df, synthetic_df):

    print_section("2. DESCRIPTIVE STATISTICS COMPARISON")

    results = []

    for feature in FEATURES:

        real_values = real_df[feature].dropna()
        synthetic_values = synthetic_df[feature].dropna()

        real_mean = real_values.mean()
        synthetic_mean = synthetic_values.mean()

        real_std = real_values.std()
        synthetic_std = synthetic_values.std()

        real_min = real_values.min()
        synthetic_min = synthetic_values.min()

        real_max = real_values.max()
        synthetic_max = synthetic_values.max()

        mean_diff = safe_percentage_difference(
            real_mean,
            synthetic_mean
        )

        std_diff = safe_percentage_difference(
            real_std,
            synthetic_std
        )

        results.append({
            "Feature": feature,

            "Real_Mean": real_mean,
            "Synthetic_Mean": synthetic_mean,
            "Mean_Difference_%": mean_diff,

            "Real_Std": real_std,
            "Synthetic_Std": synthetic_std,
            "Std_Difference_%": std_diff,

            "Real_Min": real_min,
            "Synthetic_Min": synthetic_min,

            "Real_Max": real_max,
            "Synthetic_Max": synthetic_max
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "statistical_comparison.csv",
        index=False
    )

    return results_df


# ======================================================================
# DISTRIBUTION SIMILARITY
# ======================================================================

def compare_distributions(real_df, synthetic_df):

    print_section("3. DISTRIBUTION SIMILARITY")

    results = []

    for feature in FEATURES:

        real_values = real_df[feature].dropna().values
        synthetic_values = synthetic_df[feature].dropna().values

        # --------------------------------------------------------------
        # Wasserstein Distance
        # --------------------------------------------------------------

        wasserstein = wasserstein_distance(
            real_values,
            synthetic_values
        )

        # --------------------------------------------------------------
        # Kolmogorov-Smirnov Test
        # --------------------------------------------------------------

        ks_statistic, ks_pvalue = ks_2samp(
            real_values,
            synthetic_values
        )

        results.append({
            "Feature": feature,
            "Wasserstein_Distance": wasserstein,
            "KS_Statistic": ks_statistic,
            "KS_P_Value": ks_pvalue
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "distribution_similarity.csv",
        index=False
    )

    return results_df


# ======================================================================
# CORRELATION COMPARISON
# ======================================================================

def compare_correlations(real_df, synthetic_df):

    print_section("4. CORRELATION COMPARISON")

    real_corr = real_df[FEATURES].corr()
    synthetic_corr = synthetic_df[FEATURES].corr()

    correlation_difference = (
        real_corr - synthetic_corr
    ).abs()

    print("REAL CORRELATION MATRIX")
    print(real_corr.round(3))

    print()

    print("SYNTHETIC CORRELATION MATRIX")
    print(synthetic_corr.round(3))

    print()

    print("ABSOLUTE CORRELATION DIFFERENCE")
    print(correlation_difference.round(3))

    real_corr.to_csv(
        OUTPUT_DIR / "real_correlation.csv"
    )

    synthetic_corr.to_csv(
        OUTPUT_DIR / "synthetic_correlation.csv"
    )

    correlation_difference.to_csv(
        OUTPUT_DIR / "correlation_difference.csv"
    )

    # --------------------------------------------------------------
    # Average correlation error
    # --------------------------------------------------------------

    upper_triangle = np.triu(
        np.ones(correlation_difference.shape),
        k=1
    ).astype(bool)

    average_error = correlation_difference.where(
        upper_triangle
    ).stack().mean()

    print()
    print(
        f"Average absolute correlation error: "
        f"{average_error:.4f}"
    )

    return (
        real_corr,
        synthetic_corr,
        correlation_difference,
        average_error
    )


# ======================================================================
# VOLATILITY COMPARISON
# ======================================================================

def calculate_volatility(data, feature):

    values = data[feature].dropna()

    returns = values.pct_change().dropna()

    rolling_volatility = (
        returns
        .rolling(30)
        .std()
        .dropna()
    )

    return rolling_volatility


def compare_volatility(real_df, synthetic_df):

    print_section("5. VOLATILITY COMPARISON")

    results = []

    for feature in FEATURES:

        real_vol = calculate_volatility(
            real_df,
            feature
        )

        synthetic_vol = calculate_volatility(
            synthetic_df,
            feature
        )

        real_mean = real_vol.mean()
        synthetic_mean = synthetic_vol.mean()

        difference = safe_percentage_difference(
            real_mean,
            synthetic_mean
        )

        results.append({
            "Feature": feature,
            "Real_30D_Volatility": real_mean,
            "Synthetic_30D_Volatility": synthetic_mean,
            "Difference_%": difference
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "volatility_comparison.csv",
        index=False
    )

    return results_df


# ======================================================================
# TEMPORAL BEHAVIOR
# ======================================================================

def compare_temporal_behavior(real_df, synthetic_df):

    print_section("6. TEMPORAL BEHAVIOR COMPARISON")

    results = []

    for feature in FEATURES:

        real_values = real_df[feature].dropna().values
        synthetic_values = synthetic_df[feature].dropna().values

        # First-order autocorrelation
        real_autocorr = pd.Series(
            real_values
        ).autocorr(lag=1)

        synthetic_autocorr = pd.Series(
            synthetic_values
        ).autocorr(lag=1)

        difference = abs(
            real_autocorr -
            synthetic_autocorr
        )

        results.append({
            "Feature": feature,
            "Real_Autocorrelation_Lag1": real_autocorr,
            "Synthetic_Autocorrelation_Lag1": synthetic_autocorr,
            "Absolute_Difference": difference
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "temporal_comparison.csv",
        index=False
    )

    return results_df


# ======================================================================
# PLOTS - DISTRIBUTIONS
# ======================================================================

def create_distribution_plots(real_df, synthetic_df):

    print_section("7. CREATING DISTRIBUTION PLOTS")

    for feature in FEATURES:

        plt.figure(figsize=(10, 6))

        plt.hist(
            real_df[feature].dropna(),
            bins=50,
            alpha=0.5,
            label="Real"
        )

        plt.hist(
            synthetic_df[feature].dropna(),
            bins=50,
            alpha=0.5,
            label="Synthetic"
        )

        plt.title(
            f"Real vs Synthetic Distribution - {feature}"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.2)

        filename = (
            feature.lower()
            .replace(" ", "_")
            + "_distribution.png"
        )

        plt.savefig(
            PLOTS_DIR / filename,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"✓ {filename}")


# ======================================================================
# PLOTS - CORRELATION
# ======================================================================

def create_correlation_plots(
    real_corr,
    synthetic_corr
):

    print_section("8. CREATING CORRELATION PLOTS")

    # --------------------------------------------------------------
    # Real
    # --------------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.imshow(
        real_corr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(FEATURES)),
        FEATURES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(FEATURES)),
        FEATURES
    )

    plt.title("Real Data Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "real_correlation_matrix.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # Synthetic
    # --------------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.imshow(
        synthetic_corr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(FEATURES)),
        FEATURES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(FEATURES)),
        FEATURES
    )

    plt.title("Synthetic Data Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "synthetic_correlation_matrix.png",
        dpi=150
    )

    plt.close()

    print("✓ Real correlation matrix saved")
    print("✓ Synthetic correlation matrix saved")


# ======================================================================
# PLOTS - TEMPORAL SERIES
# ======================================================================

def create_temporal_plots(real_df, synthetic_df):

    print_section("9. CREATING TEMPORAL PLOTS")

    for feature in FEATURES:

        real_values = real_df[feature].dropna()

        synthetic_values = synthetic_df[feature].dropna()

        # Use first 1000 observations so the plot remains readable
        real_values = real_values.iloc[:1000]
        synthetic_values = synthetic_values.iloc[:1000]

        plt.figure(figsize=(12, 6))

        plt.plot(
            real_values.values,
            label="Real",
            linewidth=1
        )

        plt.plot(
            synthetic_values.values,
            label="Synthetic",
            linewidth=1
        )

        plt.title(
            f"Real vs Synthetic Temporal Pattern - {feature}"
        )

        plt.xlabel("Observation")
        plt.ylabel(feature)

        plt.legend()
        plt.grid(alpha=0.2)

        filename = (
            feature.lower()
            .replace(" ", "_")
            + "_temporal.png"
        )

        plt.savefig(
            PLOTS_DIR / filename,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"✓ {filename}")


# ======================================================================
# OVERALL VALIDATION SCORE
# ======================================================================

def calculate_validation_summary(
    statistics_df,
    distribution_df,
    volatility_df,
    temporal_df,
    correlation_error
):

    print_section("10. OVERALL VALIDATION SUMMARY")

    # --------------------------------------------------------------
    # Mean similarity
    # --------------------------------------------------------------

    mean_difference = (
        statistics_df["Mean_Difference_%"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
    )

    # --------------------------------------------------------------
    # Standard deviation similarity
    # --------------------------------------------------------------

    std_difference = (
        statistics_df["Std_Difference_%"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
    )

    # --------------------------------------------------------------
    # KS statistic
    # --------------------------------------------------------------

    ks_score = distribution_df[
        "KS_Statistic"
    ].mean()

    # --------------------------------------------------------------
    # Volatility difference
    # --------------------------------------------------------------

    volatility_difference = (
        volatility_df["Difference_%"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
    )

    # --------------------------------------------------------------
    # Temporal difference
    # --------------------------------------------------------------

    temporal_difference = (
        temporal_df["Absolute_Difference"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
    )

    summary = pd.DataFrame({
        "Metric": [
            "Average Mean Difference (%)",
            "Average Std Difference (%)",
            "Average KS Statistic",
            "Average Correlation Error",
            "Average Volatility Difference (%)",
            "Average Lag-1 Autocorrelation Difference"
        ],

        "Value": [
            mean_difference,
            std_difference,
            ks_score,
            correlation_error,
            volatility_difference,
            temporal_difference
        ]
    })

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------------

    print()
    print("INTERPRETATION")
    print("-" * 70)

    print(
        "Lower mean/std differences indicate closer statistical "
        "similarity."
    )

    print(
        "Lower KS statistics indicate more similar distributions."
    )

    print(
        "Lower correlation error indicates better preservation "
        "of relationships between market variables."
    )

    print(
        "Lower volatility difference indicates better preservation "
        "of market variability."
    )

    print(
        "Lower autocorrelation difference indicates more similar "
        "temporal behavior."
    )

    return summary


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("MACROSTRESS-GAN")
    print("TIMEGAN REAL vs SYNTHETIC VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    real_df, synthetic_df = load_data()

    # --------------------------------------------------------------
    # Data quality
    # --------------------------------------------------------------

    validate_data_quality(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    statistics_df = compare_statistics(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Distribution
    # --------------------------------------------------------------

    distribution_df = compare_distributions(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Correlation
    # --------------------------------------------------------------

    (
        real_corr,
        synthetic_corr,
        correlation_difference,
        correlation_error
    ) = compare_correlations(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Volatility
    # --------------------------------------------------------------

    volatility_df = compare_volatility(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Temporal
    # --------------------------------------------------------------

    temporal_df = compare_temporal_behavior(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------

    create_distribution_plots(
        real_df,
        synthetic_df
    )

    create_correlation_plots(
        real_corr,
        synthetic_corr
    )

    create_temporal_plots(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary = calculate_validation_summary(
        statistics_df,
        distribution_df,
        volatility_df,
        temporal_df,
        correlation_error
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("TIMEGAN VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print("Validation files saved to:")

    print(
        f"{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    print("1. statistical_comparison.csv")
    print("2. distribution_similarity.csv")
    print("3. real_correlation.csv")
    print("4. synthetic_correlation.csv")
    print("5. correlation_difference.csv")
    print("6. volatility_comparison.csv")
    print("7. temporal_comparison.csv")
    print("8. validation_summary.csv")

    print()
    print(
        f"Plots saved to:\n{PLOTS_DIR}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()