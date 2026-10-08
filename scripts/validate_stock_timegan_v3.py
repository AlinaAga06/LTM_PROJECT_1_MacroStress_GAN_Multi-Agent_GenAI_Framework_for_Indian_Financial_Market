# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V3 VALIDATION
# ============================================================
#
# Purpose:
#   Validate TimeGAN V3 synthetic stock-market sequences
#   against the real stock-market feature dataset.
#
# V3:
#   Tail-aware TimeGAN
#
# Real data:
#   data/processed/institutions/stock/stock_timegan_features.csv
#
# Synthetic data:
#   outputs/synthetic/stock/stock_v3_synthetic_sequences.npy
#
# Output:
#   outputs/validation/stock_v3/
#
# Metrics:
#   1. Basic statistics
#   2. Pairwise correlation
#   3. Lag-1 temporal correlation
#   4. Extreme event coverage
#   5. KS distribution test
#   6. Wasserstein distance
#   7. Quantile comparison
#   8. Range coverage
#   9. Overall validation summary
#
# V1 and V2 files are NOT modified.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

REAL_PATH = (
    "data/processed/institutions/stock/"
    "stock_timegan_features.csv"
)

SYNTHETIC_PATH = (
    "outputs/synthetic/stock/"
    "stock_v3_synthetic_sequences.npy"
)

OUTPUT_DIR = "outputs/validation/stock_v3"


FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(value):
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD REAL DATA
# ============================================================

def load_real_data():

    print_section("LOADING REAL STOCK DATA")

    if not os.path.exists(REAL_PATH):
        raise FileNotFoundError(
            f"Real dataset not found:\n{REAL_PATH}"
        )

    df = pd.read_csv(REAL_PATH)

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required real-data features:\n"
            + "\n".join(missing_features)
        )

    real = df[FEATURE_NAMES].copy()

    real = real.replace([np.inf, -np.inf], np.nan)

    if real.isna().any().any():
        print("Warning: NaN values found in real data.")
        print("Removing invalid rows.")

        real = real.dropna()

    real_array = real.to_numpy(dtype=np.float64)

    print(f"Real file      : {REAL_PATH}")
    print(f"Rows           : {len(real_array)}")
    print(f"Features       : {real_array.shape[1]}")
    print(f"Shape          : {real_array.shape}")

    print(
        f"NaN values     : "
        f"{np.isnan(real_array).sum()}"
    )

    print(
        f"Inf values     : "
        f"{np.isinf(real_array).sum()}"
    )

    return real_array


# ============================================================
# LOAD SYNTHETIC DATA
# ============================================================

def load_synthetic_data():

    print_section("LOADING V3 SYNTHETIC DATA")

    if not os.path.exists(SYNTHETIC_PATH):
        raise FileNotFoundError(
            f"Synthetic V3 dataset not found:\n"
            f"{SYNTHETIC_PATH}"
        )

    synthetic = np.load(SYNTHETIC_PATH)

    print(f"Synthetic file : {SYNTHETIC_PATH}")
    print(f"Original shape : {synthetic.shape}")

    if synthetic.ndim != 3:
        raise ValueError(
            "Expected synthetic data with shape "
            "(num_sequences, sequence_length, features)."
        )

    if synthetic.shape[2] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, "
            f"but received {synthetic.shape[2]}."
        )

    synthetic = synthetic.astype(np.float64)

    synthetic = np.nan_to_num(
        synthetic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print(f"Sequences      : {synthetic.shape[0]}")
    print(f"Sequence length : {synthetic.shape[1]}")
    print(f"Features       : {synthetic.shape[2]}")

    print(
        f"Flattened shape : "
        f"({synthetic.shape[0] * synthetic.shape[1]}, "
        f"{synthetic.shape[2]})"
    )

    print(
        f"Min            : {synthetic.min():.8f}"
    )

    print(
        f"Max            : {synthetic.max():.8f}"
    )

    print(
        f"NaN            : {np.isnan(synthetic).sum()}"
    )

    print(
        f"Inf            : {np.isinf(synthetic).sum()}"
    )

    return synthetic


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_basic_statistics(real, synthetic_flat):

    print_section("1. BASIC STATISTICS")

    rows = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]
        synthetic_feature = synthetic_flat[:, i]

        real_mean = np.mean(real_feature)
        synthetic_mean = np.mean(synthetic_feature)

        real_std = np.std(real_feature)
        synthetic_std = np.std(synthetic_feature)

        real_min = np.min(real_feature)
        synthetic_min = np.min(synthetic_feature)

        real_max = np.max(real_feature)
        synthetic_max = np.max(synthetic_feature)

        mean_difference = abs(
            real_mean - synthetic_mean
        )

        std_difference = abs(
            real_std - synthetic_std
        )

        rows.append({
            "Feature": feature,

            "Real_Mean": real_mean,
            "Synthetic_Mean": synthetic_mean,
            "Mean_Absolute_Difference": mean_difference,

            "Real_Std": real_std,
            "Synthetic_Std": synthetic_std,
            "Std_Absolute_Difference": std_difference,

            "Real_Min": real_min,
            "Synthetic_Min": synthetic_min,

            "Real_Max": real_max,
            "Synthetic_Max": synthetic_max,
        })

        print()
        print(feature)

        print(
            f"  Real Mean     : {real_mean:.8f}"
        )
        print(
            f"  Synth Mean    : {synthetic_mean:.8f}"
        )
        print(
            f"  Mean Diff     : {mean_difference:.8f}"
        )

        print(
            f"  Real Std      : {real_std:.8f}"
        )
        print(
            f"  Synth Std     : {synthetic_std:.8f}"
        )
        print(
            f"  Std Diff      : {std_difference:.8f}"
        )

        print(
            f"  Real Min      : {real_min:.8f}"
        )
        print(
            f"  Synth Min     : {synthetic_min:.8f}"
        )

        print(
            f"  Real Max      : {real_max:.8f}"
        )
        print(
            f"  Synth Max     : {synthetic_max:.8f}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_basic_statistics.csv",
        ),
        index=False,
    )

    return df


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlation(real, synthetic_flat):

    print_section("2. PAIRWISE CORRELATION")

    real_df = pd.DataFrame(
        real,
        columns=FEATURE_NAMES,
    )

    synthetic_df = pd.DataFrame(
        synthetic_flat,
        columns=FEATURE_NAMES,
    )

    real_corr = real_df.corr()
    synthetic_corr = synthetic_df.corr()

    error_matrix = (
        real_corr - synthetic_corr
    ).abs()

    print()
    print("REAL CORRELATION")
    print(real_corr.round(4))

    print()
    print("SYNTHETIC CORRELATION")
    print(synthetic_corr.round(4))

    print()
    print("ABSOLUTE CORRELATION ERROR")
    print(error_matrix.round(4))

    upper_triangle = np.triu(
        np.ones(error_matrix.shape),
        k=1
    ).astype(bool)

    pairwise_errors = error_matrix.values[
        upper_triangle
    ]

    mean_correlation_error = np.mean(
        pairwise_errors
    )

    print()
    print(
        f"Mean pairwise correlation error : "
        f"{mean_correlation_error:.6f}"
    )

    real_corr.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_real_correlation.csv",
        )
    )

    synthetic_corr.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_synthetic_correlation.csv",
        )
    )

    error_matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_correlation_error.csv",
        )
    )

    return mean_correlation_error


# ============================================================
# LAG-1 TEMPORAL CORRELATION
# ============================================================

def calculate_lag1(real, synthetic):

    print_section("3. LAG-1 TEMPORAL CORRELATION")

    rows = []

    real_lag_values = []
    synthetic_lag_values = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]

        real_lag = np.corrcoef(
            real_feature[:-1],
            real_feature[1:],
        )[0, 1]

        # ----------------------------------------------------
        # Synthetic lag correlation
        # Calculate separately for each sequence.
        # ----------------------------------------------------

        synthetic_sequence_lags = []

        for sequence in synthetic:

            values = sequence[:, i]

            if np.std(values) == 0:
                continue

            lag = np.corrcoef(
                values[:-1],
                values[1:],
            )[0, 1]

            if np.isfinite(lag):
                synthetic_sequence_lags.append(
                    lag
                )

        if len(synthetic_sequence_lags) == 0:
            synthetic_lag = 0.0
        else:
            synthetic_lag = np.mean(
                synthetic_sequence_lags
            )

        difference = abs(
            real_lag - synthetic_lag
        )

        real_lag_values.append(real_lag)
        synthetic_lag_values.append(
            synthetic_lag
        )

        rows.append({
            "Feature": feature,
            "Real_Lag1": real_lag,
            "Synthetic_Lag1": synthetic_lag,
            "Absolute_Difference": difference,
        })

        print()
        print(feature)

        print(
            f"  Real Lag-1    : {real_lag:.6f}"
        )

        print(
            f"  Synth Lag-1   : {synthetic_lag:.6f}"
        )

        print(
            f"  Difference    : {difference:.6f}"
        )

    df = pd.DataFrame(rows)

    mean_lag_difference = df[
        "Absolute_Difference"
    ].mean()

    print()
    print(
        f"Mean lag-1 difference : "
        f"{mean_lag_difference:.6f}"
    )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_lag1_comparison.csv",
        ),
        index=False,
    )

    return mean_lag_difference


# ============================================================
# EXTREME EVENT COVERAGE
# ============================================================

def calculate_extreme_events(real, synthetic_flat):

    print_section("4. EXTREME EVENT COVERAGE")

    rows = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]
        synthetic_feature = synthetic_flat[:, i]

        low_threshold = np.percentile(
            real_feature,
            1,
        )

        high_threshold = np.percentile(
            real_feature,
            99,
        )

        real_low_count = np.sum(
            real_feature <= low_threshold
        )

        real_high_count = np.sum(
            real_feature >= high_threshold
        )

        synthetic_low_count = np.sum(
            synthetic_feature <= low_threshold
        )

        synthetic_high_count = np.sum(
            synthetic_feature >= high_threshold
        )

        low_ratio = (
            synthetic_low_count /
            real_low_count
            if real_low_count > 0
            else 0.0
        )

        high_ratio = (
            synthetic_high_count /
            real_high_count
            if real_high_count > 0
            else 0.0
        )

        rows.append({
            "Feature": feature,

            "Real_1pct_Threshold":
                low_threshold,

            "Real_99pct_Threshold":
                high_threshold,

            "Real_Low_Count":
                real_low_count,

            "Synthetic_Low_Count":
                synthetic_low_count,

            "Low_Tail_Coverage_Ratio":
                low_ratio,

            "Real_High_Count":
                real_high_count,

            "Synthetic_High_Count":
                synthetic_high_count,

            "High_Tail_Coverage_Ratio":
                high_ratio,
        })

        print()
        print(feature)

        print(
            f"  1% threshold  : "
            f"{low_threshold:.8f}"
        )

        print(
            f"  Real <= 1%    : "
            f"{real_low_count}"
        )

        print(
            f"  Synth <= 1%   : "
            f"{synthetic_low_count}"
        )

        print(
            f"  99% threshold : "
            f"{high_threshold:.8f}"
        )

        print(
            f"  Real >= 99%   : "
            f"{real_high_count}"
        )

        print(
            f"  Synth >= 99%  : "
            f"{synthetic_high_count}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_extreme_events.csv",
        ),
        index=False,
    )

    return df


# ============================================================
# DISTRIBUTION TESTS
# ============================================================

def calculate_distribution_tests(
    real,
    synthetic_flat,
):

    print_section("5. DISTRIBUTION SIMILARITY")

    rows = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]
        synthetic_feature = synthetic_flat[:, i]

        ks_result = ks_2samp(
            real_feature,
            synthetic_feature,
        )

        ks_statistic = ks_result.statistic
        ks_pvalue = ks_result.pvalue

        wasserstein = wasserstein_distance(
            real_feature,
            synthetic_feature,
        )

        rows.append({
            "Feature": feature,

            "KS_Statistic":
                ks_statistic,

            "KS_P_Value":
                ks_pvalue,

            "Wasserstein_Distance":
                wasserstein,
        })

        print()
        print(feature)

        print(
            f"  KS statistic  : "
            f"{ks_statistic:.6f}"
        )

        print(
            f"  KS p-value    : "
            f"{ks_pvalue:.6e}"
        )

        print(
            f"  Wasserstein   : "
            f"{wasserstein:.6f}"
        )

    df = pd.DataFrame(rows)

    mean_ks = df[
        "KS_Statistic"
    ].mean()

    mean_wasserstein = df[
        "Wasserstein_Distance"
    ].mean()

    print()
    print(
        f"Mean KS statistic        : "
        f"{mean_ks:.6f}"
    )

    print(
        f"Mean Wasserstein distance : "
        f"{mean_wasserstein:.6f}"
    )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_distribution_tests.csv",
        ),
        index=False,
    )

    return mean_ks, mean_wasserstein


# ============================================================
# QUANTILE COMPARISON
# ============================================================

def calculate_quantiles(
    real,
    synthetic_flat,
):

    print_section("6. QUANTILE COMPARISON")

    quantiles = [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]

    rows = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]
        synthetic_feature = synthetic_flat[:, i]

        real_quantiles = np.quantile(
            real_feature,
            quantiles,
        )

        synthetic_quantiles = np.quantile(
            synthetic_feature,
            quantiles,
        )

        print()
        print(feature)

        for q, real_value, synthetic_value in zip(
            quantiles,
            real_quantiles,
            synthetic_quantiles,
        ):

            difference = abs(
                real_value -
                synthetic_value
            )

            print(
                f"  {q * 100:5.1f}% | "
                f"Real: {real_value: .8f} | "
                f"Synth: {synthetic_value: .8f} | "
                f"Diff: {difference:.8f}"
            )

            rows.append({
                "Feature": feature,
                "Quantile": q,
                "Real_Value": real_value,
                "Synthetic_Value":
                    synthetic_value,
                "Absolute_Difference":
                    difference,
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_quantile_comparison.csv",
        ),
        index=False,
    )

    return df


# ============================================================
# RANGE COVERAGE
# ============================================================

def calculate_range_coverage(
    real,
    synthetic_flat,
):

    print_section("7. RANGE COVERAGE")

    rows = []

    for i, feature in enumerate(FEATURE_NAMES):

        real_feature = real[:, i]
        synthetic_feature = synthetic_flat[:, i]

        real_min = np.min(real_feature)
        real_max = np.max(real_feature)

        synthetic_min = np.min(
            synthetic_feature
        )

        synthetic_max = np.max(
            synthetic_feature
        )

        real_range = (
            real_max -
            real_min
        )

        synthetic_overlap_min = max(
            real_min,
            synthetic_min,
        )

        synthetic_overlap_max = min(
            real_max,
            synthetic_max,
        )

        overlap = max(
            0.0,
            synthetic_overlap_max -
            synthetic_overlap_min,
        )

        if real_range > 0:
            coverage = (
                overlap /
                real_range
            )
        else:
            coverage = 0.0

        rows.append({
            "Feature": feature,

            "Real_Min": real_min,
            "Real_Max": real_max,

            "Synthetic_Min":
                synthetic_min,
            "Synthetic_Max":
                synthetic_max,

            "Real_Range":
                real_range,

            "Overlapping_Range":
                overlap,

            "Range_Coverage":
                coverage,
        })

        print()
        print(feature)

        print(
            f"  Real range     : "
            f"{real_min:.8f} → "
            f"{real_max:.8f}"
        )

        print(
            f"  Synthetic range: "
            f"{synthetic_min:.8f} → "
            f"{synthetic_max:.8f}"
        )

        print(
            f"  Range coverage : "
            f"{coverage:.6f}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_range_coverage.csv",
        ),
        index=False,
    )

    return df


# ============================================================
# FINAL SUMMARY
# ============================================================

def create_summary(
    basic_stats,
    mean_correlation_error,
    mean_lag_difference,
    distribution_df,
    range_df,
):

    print_section("8. V3 VALIDATION SUMMARY")

    mean_error = (
        basic_stats[
            "Mean_Absolute_Difference"
        ].mean()
    )

    mean_std_error = (
        basic_stats[
            "Std_Absolute_Difference"
        ].mean()
    )

    mean_ks = (
        distribution_df[
            "KS_Statistic"
        ].mean()
    )

    mean_wasserstein = (
        distribution_df[
            "Wasserstein_Distance"
        ].mean()
    )

    mean_range_coverage = (
        range_df[
            "Range_Coverage"
        ].mean()
    )

    summary = pd.DataFrame([
        {
            "Metric":
                "Mean absolute feature mean difference",

            "Value":
                mean_error,
        },

        {
            "Metric":
                "Mean absolute feature std difference",

            "Value":
                mean_std_error,
        },

        {
            "Metric":
                "Mean pairwise correlation error",

            "Value":
                mean_correlation_error,
        },

        {
            "Metric":
                "Mean lag-1 difference",

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

        {
            "Metric":
                "Mean range coverage",

            "Value":
                mean_range_coverage,
        },
    ])

    print()

    for _, row in summary.iterrows():

        print(
            f"{row['Metric']:<48} "
            f"{row['Value']:.6f}"
        )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "stock_v3_validation_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V3 VALIDATION")
    print("=" * 78)

    print()
    print("V3 validation only.")
    print("V1 and V2 outputs will NOT be modified.")

    ensure_output_directory()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    real = load_real_data()

    synthetic = load_synthetic_data()

    # --------------------------------------------------------
    # Flatten synthetic sequences for distribution metrics
    # --------------------------------------------------------

    synthetic_flat = synthetic.reshape(
        -1,
        synthetic.shape[-1],
    )

    print_section("DATA SHAPES")

    print(
        f"Real feature matrix     : "
        f"{real.shape}"
    )

    print(
        f"Synthetic sequence data : "
        f"{synthetic.shape}"
    )

    print(
        f"Synthetic flattened data: "
        f"{synthetic_flat.shape}"
    )

    # --------------------------------------------------------
    # Validation 1
    # --------------------------------------------------------

    basic_stats = calculate_basic_statistics(
        real,
        synthetic_flat,
    )

    # --------------------------------------------------------
    # Validation 2
    # --------------------------------------------------------

    mean_correlation_error = (
        calculate_correlation(
            real,
            synthetic_flat,
        )
    )

    # --------------------------------------------------------
    # Validation 3
    # --------------------------------------------------------

    mean_lag_difference = calculate_lag1(
        real,
        synthetic,
    )

    # --------------------------------------------------------
    # Validation 4
    # --------------------------------------------------------

    calculate_extreme_events(
        real,
        synthetic_flat,
    )

    # --------------------------------------------------------
    # Validation 5
    # --------------------------------------------------------

    mean_ks, mean_wasserstein = (
        calculate_distribution_tests(
            real,
            synthetic_flat,
        )
    )

    # --------------------------------------------------------
    # Validation 6
    # --------------------------------------------------------

    calculate_quantiles(
        real,
        synthetic_flat,
    )

    # --------------------------------------------------------
    # Validation 7
    # --------------------------------------------------------

    range_df = calculate_range_coverage(
        real,
        synthetic_flat,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    distribution_df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "stock_v3_distribution_tests.csv",
        )
    )

    summary = create_summary(
        basic_stats=basic_stats,
        mean_correlation_error=
            mean_correlation_error,
        mean_lag_difference=
            mean_lag_difference,
        distribution_df=
            distribution_df,
        range_df=
            range_df,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_section("VALIDATION COMPLETED")

    print(
        f"Validation directory:\n"
        f"{os.path.abspath(OUTPUT_DIR)}"
    )

    print()
    print("Generated files:")

    output_files = sorted(
        os.listdir(OUTPUT_DIR)
    )

    for index, filename in enumerate(
        output_files,
        start=1,
    ):

        print(
            f"{index}. "
            f"{os.path.join(OUTPUT_DIR, filename)}"
        )

    print()
    print("=" * 78)
    print("V3 VALIDATION FINISHED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()