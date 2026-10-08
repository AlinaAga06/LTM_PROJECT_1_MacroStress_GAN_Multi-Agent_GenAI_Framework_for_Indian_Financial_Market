"""
MacroStress-GAN
Stock Market TimeGAN V1
Real vs Synthetic Validation

Compares:
    REAL:
        data/processed/institutions/stock/stock_timegan_features.csv

    SYNTHETIC:
        outputs/synthetic/stock/stock_synthetic_sequences.npy

Validation:
    1. Basic shape and data checks
    2. Mean / standard deviation
    3. Quantiles
    4. Min / max
    5. Correlation matrices
    6. Correlation difference
    7. Lag-1 autocorrelation
    8. Extreme event coverage
    9. Distribution similarity using KS test
   10. Wasserstein distance
   11. Overall validation summary
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "stock_timegan_features.csv"
)

SYNTHETIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
    / "stock"
    / "stock_synthetic_sequences.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "stock"
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

SEQ_LEN = 30
FEATURE_DIM = 5


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ============================================================
# LOAD REAL DATA
# ============================================================

def load_real_data():
    print_header("LOADING REAL STOCK DATA")

    if not REAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Real feature file not found:\n{REAL_FEATURES_PATH}"
        )

    df = pd.read_csv(REAL_FEATURES_PATH)

    print(f"File : {REAL_FEATURES_PATH}")
    print(f"Rows : {len(df):,}")

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in real dataset: {missing_features}"
        )

    data = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        print("\nWARNING: NaN values detected.")
        print(data.isna().sum())
        data = data.dropna()

    real = data.to_numpy(dtype=np.float64)

    print(f"Real feature matrix shape: {real.shape}")

    return real, df


# ============================================================
# LOAD SYNTHETIC DATA
# ============================================================

def load_synthetic_data():
    print_header("LOADING SYNTHETIC STOCK DATA")

    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic file not found:\n{SYNTHETIC_PATH}"
        )

    synthetic = np.load(SYNTHETIC_PATH)

    print(f"File : {SYNTHETIC_PATH}")
    print(f"Shape: {synthetic.shape}")

    if synthetic.ndim != 3:
        raise ValueError(
            f"Expected 3D synthetic array, got {synthetic.ndim}D."
        )

    if synthetic.shape[1] != SEQ_LEN:
        raise ValueError(
            f"Expected sequence length {SEQ_LEN}, "
            f"got {synthetic.shape[1]}."
        )

    if synthetic.shape[2] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features, "
            f"got {synthetic.shape[2]}."
        )

    synthetic_flat = synthetic.reshape(-1, FEATURE_DIM)

    print(f"Flattened synthetic shape: {synthetic_flat.shape}")

    return synthetic, synthetic_flat


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(real, synthetic_flat):
    print_header("1. BASIC STATISTICS")

    rows = []

    for i, feature in enumerate(FEATURES):

        real_values = real[:, i]
        synthetic_values = synthetic_flat[:, i]

        rows.append({
            "Feature": feature,

            "Real Mean": np.mean(real_values),
            "Synthetic Mean": np.mean(synthetic_values),

            "Real Std": np.std(real_values),
            "Synthetic Std": np.std(synthetic_values),

            "Real Min": np.min(real_values),
            "Synthetic Min": np.min(synthetic_values),

            "Real Max": np.max(real_values),
            "Synthetic Max": np.max(synthetic_values),

            "Mean Abs Difference":
                abs(
                    np.mean(real_values)
                    - np.mean(synthetic_values)
                ),

            "Std Abs Difference":
                abs(
                    np.std(real_values)
                    - np.std(synthetic_values)
                ),
        })

    stats = pd.DataFrame(rows)

    print(
        stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return stats


# ============================================================
# QUANTILES
# ============================================================

def calculate_quantiles(real, synthetic_flat):
    print_header("2. QUANTILE COMPARISON")

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    rows = []

    for i, feature in enumerate(FEATURES):

        real_values = real[:, i]
        synthetic_values = synthetic_flat[:, i]

        for q in quantiles:

            real_q = np.quantile(real_values, q)
            synthetic_q = np.quantile(synthetic_values, q)

            rows.append({
                "Feature": feature,
                "Quantile": q,
                "Real": real_q,
                "Synthetic": synthetic_q,
                "Absolute Difference":
                    abs(real_q - synthetic_q),
            })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return result


# ============================================================
# CORRELATION MATRICES
# ============================================================

def calculate_correlations(real, synthetic_flat):
    print_header("3. CORRELATION COMPARISON")

    real_corr = np.corrcoef(real, rowvar=False)
    synthetic_corr = np.corrcoef(
        synthetic_flat,
        rowvar=False
    )

    correlation_difference = np.abs(
        real_corr - synthetic_corr
    )

    print("\nREAL CORRELATION MATRIX")
    print(
        pd.DataFrame(
            real_corr,
            index=FEATURES,
            columns=FEATURES
        ).round(4).to_string()
    )

    print("\nSYNTHETIC CORRELATION MATRIX")
    print(
        pd.DataFrame(
            synthetic_corr,
            index=FEATURES,
            columns=FEATURES
        ).round(4).to_string()
    )

    print("\nABSOLUTE CORRELATION DIFFERENCE")
    print(
        pd.DataFrame(
            correlation_difference,
            index=FEATURES,
            columns=FEATURES
        ).round(4).to_string()
    )

    upper_triangle = np.triu_indices(
        FEATURE_DIM,
        k=1
    )

    mean_corr_error = np.mean(
        correlation_difference[upper_triangle]
    )

    print(
        f"\nMean pairwise correlation error: "
        f"{mean_corr_error:.6f}"
    )

    return (
        real_corr,
        synthetic_corr,
        correlation_difference,
        mean_corr_error,
    )


# ============================================================
# LAG-1 AUTOCORRELATION
# ============================================================

def lag1_autocorrelation(values):
    if len(values) < 2:
        return np.nan

    x = values[:-1]
    y = values[1:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def calculate_lag1(real, synthetic):
    print_header("4. LAG-1 TEMPORAL CORRELATION")

    rows = []

    for i, feature in enumerate(FEATURES):

        real_lag = lag1_autocorrelation(
            real[:, i]
        )

        synthetic_lag_values = []

        for sequence in synthetic:
            synthetic_lag_values.append(
                lag1_autocorrelation(
                    sequence[:, i]
                )
            )

        synthetic_lag = np.nanmean(
            synthetic_lag_values
        )

        rows.append({
            "Feature": feature,
            "Real Lag-1": real_lag,
            "Synthetic Lag-1": synthetic_lag,
            "Absolute Difference":
                abs(real_lag - synthetic_lag),
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return result


# ============================================================
# EXTREME EVENT COVERAGE
# ============================================================

def calculate_extreme_events(real, synthetic_flat):
    print_header("5. EXTREME EVENT COVERAGE")

    rows = []

    for i, feature in enumerate(FEATURES):

        real_values = real[:, i]
        synthetic_values = synthetic_flat[:, i]

        q01 = np.quantile(real_values, 0.01)
        q99 = np.quantile(real_values, 0.99)

        real_lower = np.sum(
            real_values <= q01
        )

        real_upper = np.sum(
            real_values >= q99
        )

        synthetic_lower = np.sum(
            synthetic_values <= q01
        )

        synthetic_upper = np.sum(
            synthetic_values >= q99
        )

        rows.append({
            "Feature": feature,
            "Real <= 1% Threshold": real_lower,
            "Synthetic <= 1% Threshold": synthetic_lower,
            "Real >= 99% Threshold": real_upper,
            "Synthetic >= 99% Threshold": synthetic_upper,
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(index=False)
    )

    return result


# ============================================================
# KS TEST + WASSERSTEIN DISTANCE
# ============================================================

def distribution_tests(real, synthetic_flat):
    print_header("6. DISTRIBUTION SIMILARITY")

    rows = []

    for i, feature in enumerate(FEATURES):

        real_values = real[:, i]
        synthetic_values = synthetic_flat[:, i]

        ks_stat, ks_pvalue = ks_2samp(
            real_values,
            synthetic_values
        )

        wasserstein = wasserstein_distance(
            real_values,
            synthetic_values
        )

        rows.append({
            "Feature": feature,
            "KS Statistic": ks_stat,
            "KS p-value": ks_pvalue,
            "Wasserstein Distance": wasserstein,
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return result


# ============================================================
# RANGE COVERAGE
# ============================================================

def calculate_range_coverage(real, synthetic_flat):
    print_header("7. RANGE COVERAGE")

    rows = []

    for i, feature in enumerate(FEATURES):

        real_values = real[:, i]
        synthetic_values = synthetic_flat[:, i]

        real_min = np.min(real_values)
        real_max = np.max(real_values)

        synthetic_min = np.min(synthetic_values)
        synthetic_max = np.max(synthetic_values)

        real_range = real_max - real_min

        if real_range > 0:
            min_coverage = (
                (synthetic_min - real_min)
                / real_range
                * 100
            )

            max_coverage = (
                (real_max - synthetic_max)
                / real_range
                * 100
            )
        else:
            min_coverage = np.nan
            max_coverage = np.nan

        rows.append({
            "Feature": feature,
            "Real Min": real_min,
            "Synthetic Min": synthetic_min,
            "Real Max": real_max,
            "Synthetic Max": synthetic_max,
            "Min Range Gap %": min_coverage,
            "Max Range Gap %": max_coverage,
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return result


# ============================================================
# OVERALL SUMMARY
# ============================================================

def create_summary(
    stats,
    quantiles,
    correlation_error,
    lag1,
    distribution,
):
    print_header("8. VALIDATION SUMMARY")

    mean_difference = np.mean(
        stats["Mean Abs Difference"]
    )

    std_difference = np.mean(
        stats["Std Abs Difference"]
    )

    mean_ks = np.mean(
        distribution["KS Statistic"]
    )

    mean_wasserstein = np.mean(
        distribution["Wasserstein Distance"]
    )

    mean_lag_difference = np.mean(
        lag1["Absolute Difference"]
    )

    summary = pd.DataFrame([
        {
            "Metric":
                "Mean absolute difference in feature means",
            "Value":
                mean_difference,
        },
        {
            "Metric":
                "Mean absolute difference in feature std",
            "Value":
                std_difference,
        },
        {
            "Metric":
                "Mean pairwise correlation error",
            "Value":
                correlation_error,
        },
        {
            "Metric":
                "Mean lag-1 correlation difference",
            "Value":
                mean_lag_difference,
        },
        {
            "Metric":
                "Mean KS statistic",
            "Value":
                mean_ks,
        },
        {
            "Metric":
                "Mean Wasserstein distance",
            "Value":
                mean_wasserstein,
        },
    ])

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return summary


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    stats,
    quantiles,
    lag1,
    extreme,
    distribution,
    range_coverage,
    summary,
):
    print_header("9. SAVING VALIDATION RESULTS")

    stats.to_csv(
        OUTPUT_DIR / "stock_basic_statistics.csv",
        index=False
    )

    quantiles.to_csv(
        OUTPUT_DIR / "stock_quantile_comparison.csv",
        index=False
    )

    lag1.to_csv(
        OUTPUT_DIR / "stock_lag1_comparison.csv",
        index=False
    )

    extreme.to_csv(
        OUTPUT_DIR / "stock_extreme_events.csv",
        index=False
    )

    distribution.to_csv(
        OUTPUT_DIR / "stock_distribution_tests.csv",
        index=False
    )

    range_coverage.to_csv(
        OUTPUT_DIR / "stock_range_coverage.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "stock_validation_summary.csv",
        index=False
    )

    print(f"Validation directory:")
    print(OUTPUT_DIR)

    print("\nSaved files:")

    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {file.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V1")
    print("REAL vs SYNTHETIC VALIDATION")
    print("=" * 78)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print("\nExpected real dataset:")
    print(REAL_FEATURES_PATH)

    print("\nExpected synthetic dataset:")
    print(SYNTHETIC_PATH)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    real, real_df = load_real_data()

    synthetic, synthetic_flat = load_synthetic_data()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    quantiles = calculate_quantiles(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    (
        real_corr,
        synthetic_corr,
        correlation_difference,
        correlation_error,
    ) = calculate_correlations(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Lag-1
    # --------------------------------------------------------

    lag1 = calculate_lag1(
        real,
        synthetic
    )

    # --------------------------------------------------------
    # Extreme events
    # --------------------------------------------------------

    extreme = calculate_extreme_events(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Distribution tests
    # --------------------------------------------------------

    distribution = distribution_tests(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Range coverage
    # --------------------------------------------------------

    range_coverage = calculate_range_coverage(
        real,
        synthetic_flat
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        stats,
        quantiles,
        correlation_error,
        lag1,
        distribution,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        stats,
        quantiles,
        lag1,
        extreme,
        distribution,
        range_coverage,
        summary,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print_header("STOCK TIMEGAN V1 VALIDATION COMPLETED")

    print("Real data:")
    print(f"  Rows: {len(real):,}")

    print("\nSynthetic data:")
    print(f"  Sequences: {len(synthetic):,}")
    print(f"  Sequence length: {synthetic.shape[1]}")
    print(f"  Features: {synthetic.shape[2]}")
    print(
        f"  Total synthetic observations: "
        f"{len(synthetic_flat):,}"
    )

    print("\nValidation results saved to:")
    print(OUTPUT_DIR)

    print("\nNext step:")
    print(
        "Review the validation metrics before deciding "
        "whether the Stock TimeGAN requires retraining."
    )


if __name__ == "__main__":
    main()