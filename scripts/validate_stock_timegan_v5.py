# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V5 VALIDATION
# ============================================================
#
# Validates V5 synthetic stock-market sequences against
# REAL stock-market feature data.
#
# Metrics:
#   1. Basic statistics
#   2. Correlation structure
#   3. Lag-1 temporal behavior
#   4. Extreme-event coverage
#   5. KS statistic
#   6. Wasserstein distance
#   7. Quantile comparison
#   8. Range coverage
#   9. Summary metrics
#  10. V1/V2/V3/V4/V5 comparison
#
# IMPORTANT:
#   This script does NOT modify V1-V4.
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    wasserstein_distance
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# INPUT FILES
# ============================================================

REAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_features.csv"
)

SYNTHETIC_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "stock",
    "stock_v5_synthetic_sequences.npy"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "validation",
    "stock_v5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

BASIC_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_basic_statistics.csv"
)

CORRELATION_ERROR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_correlation_error.csv"
)

DISTRIBUTION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_distribution_tests.csv"
)

EXTREME_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_extreme_events.csv"
)

LAG_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_lag1_comparison.csv"
)

QUANTILE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_quantile_comparison.csv"
)

RANGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_range_coverage.csv"
)

REAL_CORRELATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_real_correlation.csv"
)

SYNTHETIC_CORRELATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_synthetic_correlation.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_validation_summary.csv"
)

VERSION_COMPARISON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_vs_v1_v2_v3_v4_comparison.csv"
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V5 VALIDATION")
print("=" * 78)


# ============================================================
# LOAD REAL DATA
# ============================================================

print("\n")
print("=" * 78)
print("LOADING REAL DATA")
print("=" * 78)

if not os.path.exists(REAL_FILE):

    raise FileNotFoundError(
        f"\nReal feature file not found:\n{REAL_FILE}"
    )

real_df = pd.read_csv(
    REAL_FILE
)

# ------------------------------------------------------------
# Verify feature columns
# ------------------------------------------------------------

missing_features = [
    feature
    for feature in FEATURE_NAMES
    if feature not in real_df.columns
]

if missing_features:

    raise ValueError(
        "\nMissing feature columns:\n"
        + "\n".join(missing_features)
    )


real_data = real_df[
    FEATURE_NAMES
].values.astype(
    np.float64
)


print(
    f"\nReal file:\n{REAL_FILE}"
)

print(
    f"\nReal shape: {real_data.shape}"
)

print(
    f"NaN count: {np.isnan(real_data).sum()}"
)

print(
    f"Inf count: {np.isinf(real_data).sum()}"
)

if np.isnan(real_data).any():

    raise ValueError(
        "Real data contains NaN values."
    )

if np.isinf(real_data).any():

    raise ValueError(
        "Real data contains Inf values."
    )


# ============================================================
# LOAD SYNTHETIC V5 DATA
# ============================================================

print("\n")
print("=" * 78)
print("LOADING V5 SYNTHETIC DATA")
print("=" * 78)

if not os.path.exists(SYNTHETIC_FILE):

    raise FileNotFoundError(
        f"\nV5 synthetic file not found:\n"
        f"{SYNTHETIC_FILE}"
    )

synthetic_sequences = np.load(
    SYNTHETIC_FILE
).astype(
    np.float64
)

print(
    f"\nSynthetic file:\n"
    f"{SYNTHETIC_FILE}"
)

print(
    f"\nSynthetic shape: "
    f"{synthetic_sequences.shape}"
)

print(
    f"NaN count: "
    f"{np.isnan(synthetic_sequences).sum()}"
)

print(
    f"Inf count: "
    f"{np.isinf(synthetic_sequences).sum()}"
)


# ============================================================
# VALIDATE SYNTHETIC SHAPE
# ============================================================

if synthetic_sequences.ndim != 3:

    raise ValueError(
        "Synthetic data must be 3-dimensional "
        "(sequences, timesteps, features)."
    )


num_sequences = synthetic_sequences.shape[0]
seq_len = synthetic_sequences.shape[1]
num_features = synthetic_sequences.shape[2]


if num_features != len(FEATURE_NAMES):

    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, "
        f"received {num_features}."
    )


if np.isnan(
    synthetic_sequences
).any():

    raise ValueError(
        "Synthetic data contains NaN values."
    )


if np.isinf(
    synthetic_sequences
).any():

    raise ValueError(
        "Synthetic data contains Inf values."
    )


print(
    f"\nNumber of sequences: "
    f"{num_sequences}"
)

print(
    f"Sequence length: "
    f"{seq_len}"
)

print(
    f"Number of features: "
    f"{num_features}"
)

print(
    f"Minimum: "
    f"{synthetic_sequences.min():.10f}"
)

print(
    f"Maximum: "
    f"{synthetic_sequences.max():.10f}"
)


# ============================================================
# FLATTEN SYNTHETIC DATA
# ============================================================

synthetic_data = synthetic_sequences.reshape(
    -1,
    num_features
)


print(
    f"\nFlattened synthetic shape: "
    f"{synthetic_data.shape}"
)


# ============================================================
# 1. BASIC STATISTICS
# ============================================================

print("\n")
print("=" * 78)
print("1. BASIC STATISTICS")
print("=" * 78)

basic_rows = []

mean_differences = []
std_differences = []

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = real_data[
        :, index
    ]

    synthetic_values = synthetic_data[
        :, index
    ]

    real_mean = np.mean(
        real_values
    )

    synthetic_mean = np.mean(
        synthetic_values
    )

    real_std = np.std(
        real_values
    )

    synthetic_std = np.std(
        synthetic_values
    )

    mean_difference = abs(
        real_mean -
        synthetic_mean
    )

    std_difference = abs(
        real_std -
        synthetic_std
    )

    mean_differences.append(
        mean_difference
    )

    std_differences.append(
        std_difference
    )

    row = {

        "Feature":
            feature,

        "Real_Mean":
            real_mean,

        "Synthetic_Mean":
            synthetic_mean,

        "Mean_Absolute_Difference":
            mean_difference,

        "Real_Std":
            real_std,

        "Synthetic_Std":
            synthetic_std,

        "Std_Absolute_Difference":
            std_difference,

        "Real_Min":
            np.min(real_values),

        "Synthetic_Min":
            np.min(synthetic_values),

        "Real_Max":
            np.max(real_values),

        "Synthetic_Max":
            np.max(synthetic_values),
    }

    basic_rows.append(
        row
    )

    print(
        f"\n{feature}"
    )

    print(
        f"  Real mean      : "
        f"{real_mean:.8f}"
    )

    print(
        f"  Synthetic mean : "
        f"{synthetic_mean:.8f}"
    )

    print(
        f"  Mean difference: "
        f"{mean_difference:.8f}"
    )

    print(
        f"  Real std       : "
        f"{real_std:.8f}"
    )

    print(
        f"  Synthetic std  : "
        f"{synthetic_std:.8f}"
    )

    print(
        f"  Std difference : "
        f"{std_difference:.8f}"
    )

    print(
        f"  Real range     : "
        f"{np.min(real_values):.8f} "
        f"to "
        f"{np.max(real_values):.8f}"
    )

    print(
        f"  Synthetic range: "
        f"{np.min(synthetic_values):.8f} "
        f"to "
        f"{np.max(synthetic_values):.8f}"
    )


basic_df = pd.DataFrame(
    basic_rows
)

basic_df.to_csv(
    BASIC_OUTPUT,
    index=False
)


mean_difference_summary = np.mean(
    mean_differences
)

std_difference_summary = np.mean(
    std_differences
)


print("\n")
print(
    f"Mean absolute feature mean difference: "
    f"{mean_difference_summary:.6f}"
)

print(
    f"Mean absolute feature std difference : "
    f"{std_difference_summary:.6f}"
)


# ============================================================
# 2. CORRELATION
# ============================================================

print("\n")
print("=" * 78)
print("2. CORRELATION STRUCTURE")
print("=" * 78)

real_corr = np.corrcoef(
    real_data,
    rowvar=False
)

synthetic_corr = np.corrcoef(
    synthetic_data,
    rowvar=False
)


real_corr_df = pd.DataFrame(
    real_corr,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

synthetic_corr_df = pd.DataFrame(
    synthetic_corr,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)


real_corr_df.to_csv(
    REAL_CORRELATION_OUTPUT
)

synthetic_corr_df.to_csv(
    SYNTHETIC_CORRELATION_OUTPUT
)


correlation_error_matrix = np.abs(
    real_corr -
    synthetic_corr
)


# ------------------------------------------------------------
# Diagonal is always approximately zero, so use pairwise
# off-diagonal values for the main correlation metric.
# ------------------------------------------------------------

upper_triangle = np.triu_indices(
    len(FEATURE_NAMES),
    k=1
)

pairwise_errors = (
    correlation_error_matrix[
        upper_triangle
    ]
)

mean_pairwise_correlation_error = np.mean(
    pairwise_errors
)


correlation_rows = []

for i in range(
    len(FEATURE_NAMES)
):

    for j in range(
        i + 1,
        len(FEATURE_NAMES)
    ):

        correlation_rows.append({

            "Feature_1":
                FEATURE_NAMES[i],

            "Feature_2":
                FEATURE_NAMES[j],

            "Real_Correlation":
                real_corr[i, j],

            "Synthetic_Correlation":
                synthetic_corr[i, j],

            "Absolute_Error":
                abs(
                    real_corr[i, j]
                    -
                    synthetic_corr[i, j]
                )
        })


correlation_error_df = pd.DataFrame(
    correlation_rows
)

correlation_error_df.to_csv(
    CORRELATION_ERROR_OUTPUT,
    index=False
)


print("\nReal correlation matrix:")

print(
    real_corr_df.round(4).to_string()
)

print("\nSynthetic V5 correlation matrix:")

print(
    synthetic_corr_df.round(4).to_string()
)

print(
    "\nMean pairwise correlation error: "
    f"{mean_pairwise_correlation_error:.6f}"
)


# ============================================================
# 3. LAG-1 TEMPORAL BEHAVIOR
# ============================================================

print("\n")
print("=" * 78)
print("3. LAG-1 TEMPORAL BEHAVIOR")
print("=" * 78)


def calculate_lag1(values):

    if len(values) < 2:

        return np.nan

    x = values[:-1]

    y = values[1:]

    if (
        np.std(x) < 1e-12
        or
        np.std(y) < 1e-12
    ):

        return 0.0

    return np.corrcoef(
        x,
        y
    )[0, 1]


# ------------------------------------------------------------
# Real lag-1
# ------------------------------------------------------------

real_lag1_values = []

for index in range(
    num_features
):

    real_lag1 = calculate_lag1(
        real_data[:, index]
    )

    real_lag1_values.append(
        real_lag1
    )


# ------------------------------------------------------------
# Synthetic lag-1
# Calculate separately inside each sequence.
# ------------------------------------------------------------

synthetic_lag1_values = []

for feature_index in range(
    num_features
):

    feature_lags = []

    for sequence_index in range(
        num_sequences
    ):

        sequence = synthetic_sequences[
            sequence_index,
            :,
            feature_index
        ]

        lag = calculate_lag1(
            sequence
        )

        if not np.isnan(lag):

            feature_lags.append(
                lag
            )

    synthetic_lag1_values.append(
        np.mean(feature_lags)
    )


lag_rows = []

lag_differences = []

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_lag = real_lag1_values[
        index
    ]

    synthetic_lag = synthetic_lag1_values[
        index
    ]

    difference = abs(
        real_lag -
        synthetic_lag
    )

    lag_differences.append(
        difference
    )

    lag_rows.append({

        "Feature":
            feature,

        "Real_Lag1":
            real_lag,

        "Synthetic_Lag1":
            synthetic_lag,

        "Absolute_Difference":
            difference
    })

    print(
        f"\n{feature}"
    )

    print(
        f"  Real lag-1      : "
        f"{real_lag:.8f}"
    )

    print(
        f"  Synthetic lag-1 : "
        f"{synthetic_lag:.8f}"
    )

    print(
        f"  Absolute diff    : "
        f"{difference:.8f}"
    )


lag_df = pd.DataFrame(
    lag_rows
)

lag_df.to_csv(
    LAG_OUTPUT,
    index=False
)


mean_lag1_difference = np.mean(
    lag_differences
)

print(
    "\nMean lag-1 difference: "
    f"{mean_lag1_difference:.6f}"
)


# ============================================================
# 4. EXTREME EVENT COVERAGE
# ============================================================

print("\n")
print("=" * 78)
print("4. EXTREME EVENT COVERAGE")
print("=" * 78)


extreme_rows = []


for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = real_data[
        :, index
    ]

    synthetic_values = synthetic_data[
        :, index
    ]

    # --------------------------------------------------------
    # Real 1st and 99th percentiles
    # --------------------------------------------------------

    lower_threshold = np.percentile(
        real_values,
        1
    )

    upper_threshold = np.percentile(
        real_values,
        99
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    real_lower_count = np.sum(
        real_values <=
        lower_threshold
    )

    synthetic_lower_count = np.sum(
        synthetic_values <=
        lower_threshold
    )

    real_upper_count = np.sum(
        real_values >=
        upper_threshold
    )

    synthetic_upper_count = np.sum(
        synthetic_values >=
        upper_threshold
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    lower_coverage = (
        synthetic_lower_count /
        real_lower_count
        if real_lower_count > 0
        else 0
    )

    upper_coverage = (
        synthetic_upper_count /
        real_upper_count
        if real_upper_count > 0
        else 0
    )

    extreme_rows.append({

        "Feature":
            feature,

        "Real_1pct_Threshold":
            lower_threshold,

        "Real_99pct_Threshold":
            upper_threshold,

        "Real_Lower_Count":
            real_lower_count,

        "Synthetic_Lower_Count":
            synthetic_lower_count,

        "Lower_Coverage":
            lower_coverage,

        "Real_Upper_Count":
            real_upper_count,

        "Synthetic_Upper_Count":
            synthetic_upper_count,

        "Upper_Coverage":
            upper_coverage,
    })

    print(
        f"\n{feature}"
    )

    print(
        f"  Lower threshold: "
        f"{lower_threshold:.8f}"
    )

    print(
        f"  Upper threshold: "
        f"{upper_threshold:.8f}"
    )

    print(
        f"  Real lower     : "
        f"{real_lower_count}"
    )

    print(
        f"  Synthetic lower: "
        f"{synthetic_lower_count}"
    )

    print(
        f"  Lower coverage : "
        f"{lower_coverage:.6f}"
    )

    print(
        f"  Real upper     : "
        f"{real_upper_count}"
    )

    print(
        f"  Synthetic upper: "
        f"{synthetic_upper_count}"
    )

    print(
        f"  Upper coverage : "
        f"{upper_coverage:.6f}"
    )


extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_df.to_csv(
    EXTREME_OUTPUT,
    index=False
)


# ============================================================
# 5. KS + WASSERSTEIN
# ============================================================

print("\n")
print("=" * 78)
print("5. DISTRIBUTION TESTS")
print("=" * 78)

distribution_rows = []

ks_values = []
wasserstein_values = []


for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = real_data[
        :, index
    ]

    synthetic_values = synthetic_data[
        :, index
    ]

    # --------------------------------------------------------
    # KS
    # --------------------------------------------------------

    ks_result = ks_2samp(
        real_values,
        synthetic_values
    )

    ks_statistic = ks_result.statistic

    # --------------------------------------------------------
    # Wasserstein
    # --------------------------------------------------------

    wasserstein = wasserstein_distance(
        real_values,
        synthetic_values
    )

    ks_values.append(
        ks_statistic
    )

    wasserstein_values.append(
        wasserstein
    )

    distribution_rows.append({

        "Feature":
            feature,

        "KS_Statistic":
            ks_statistic,

        "KS_p_value":
            ks_result.pvalue,

        "Wasserstein_Distance":
            wasserstein
    })

    print(
        f"\n{feature}"
    )

    print(
        f"  KS statistic       : "
        f"{ks_statistic:.6f}"
    )

    print(
        f"  KS p-value         : "
        f"{ks_result.pvalue:.6e}"
    )

    print(
        f"  Wasserstein distance: "
        f"{wasserstein:.6f}"
    )


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    DISTRIBUTION_OUTPUT,
    index=False
)


mean_ks = np.mean(
    ks_values
)

mean_wasserstein = np.mean(
    wasserstein_values
)


print(
    f"\nMean KS statistic: "
    f"{mean_ks:.6f}"
)

print(
    f"Mean Wasserstein distance: "
    f"{mean_wasserstein:.6f}"
)


# ============================================================
# 6. QUANTILE COMPARISON
# ============================================================

print("\n")
print("=" * 78)
print("6. QUANTILE COMPARISON")
print("=" * 78)

quantiles = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99
]

quantile_rows = []


for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = real_data[
        :, index
    ]

    synthetic_values = synthetic_data[
        :, index
    ]

    print(
        f"\n{feature}"
    )

    for q in quantiles:

        real_q = np.quantile(
            real_values,
            q
        )

        synthetic_q = np.quantile(
            synthetic_values,
            q
        )

        difference = abs(
            real_q -
            synthetic_q
        )

        quantile_rows.append({

            "Feature":
                feature,

            "Quantile":
                q,

            "Real_Value":
                real_q,

            "Synthetic_Value":
                synthetic_q,

            "Absolute_Difference":
                difference
        })

        print(
            f"  Q{q:.2f}: "
            f"Real={real_q:.8f} | "
            f"Synthetic={synthetic_q:.8f} | "
            f"Diff={difference:.8f}"
        )


quantile_df = pd.DataFrame(
    quantile_rows
)

quantile_df.to_csv(
    QUANTILE_OUTPUT,
    index=False
)


# ============================================================
# 7. RANGE COVERAGE
# ============================================================

print("\n")
print("=" * 78)
print("7. RANGE COVERAGE")
print("=" * 78)

range_rows = []

range_coverages = []


for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = real_data[
        :, index
    ]

    synthetic_values = synthetic_data[
        :, index
    ]

    real_min = np.min(
        real_values
    )

    real_max = np.max(
        real_values
    )

    synthetic_min = np.min(
        synthetic_values
    )

    synthetic_max = np.max(
        synthetic_values
    )

    real_range = (
        real_max -
        real_min
    )

    synthetic_range = (
        synthetic_max -
        synthetic_min
    )

    if real_range <= 0:

        coverage = 0.0

    else:

        overlap_min = max(
            real_min,
            synthetic_min
        )

        overlap_max = min(
            real_max,
            synthetic_max
        )

        overlap = max(
            0.0,
            overlap_max -
            overlap_min
        )

        coverage = (
            overlap /
            real_range
        )

    range_coverages.append(
        coverage
    )

    range_rows.append({

        "Feature":
            feature,

        "Real_Min":
            real_min,

        "Real_Max":
            real_max,

        "Synthetic_Min":
            synthetic_min,

        "Synthetic_Max":
            synthetic_max,

        "Real_Range":
            real_range,

        "Synthetic_Range":
            synthetic_range,

        "Range_Coverage":
            coverage
    })

    print(
        f"\n{feature}"
    )

    print(
        f"  Real range     : "
        f"{real_min:.8f} "
        f"to "
        f"{real_max:.8f}"
    )

    print(
        f"  Synthetic range: "
        f"{synthetic_min:.8f} "
        f"to "
        f"{synthetic_max:.8f}"
    )

    print(
        f"  Range coverage : "
        f"{coverage:.6f}"
    )


range_df = pd.DataFrame(
    range_rows
)

range_df.to_csv(
    RANGE_OUTPUT,
    index=False
)


mean_range_coverage = np.mean(
    range_coverages
)

print(
    f"\nMean range coverage: "
    f"{mean_range_coverage:.6f}"
)


# ============================================================
# 8. OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("8. V5 VALIDATION SUMMARY")
print("=" * 78)


summary_data = {

    "Model":
        "TimeGAN_Stock_V5",

    "Mean_Absolute_Feature_Mean_Difference":
        mean_difference_summary,

    "Mean_Absolute_Feature_Std_Difference":
        std_difference_summary,

    "Mean_Pairwise_Correlation_Error":
        mean_pairwise_correlation_error,

    "Mean_Lag1_Difference":
        mean_lag1_difference,

    "Mean_KS_Statistic":
        mean_ks,

    "Mean_Wasserstein_Distance":
        mean_wasserstein,

    "Mean_Range_Coverage":
        mean_range_coverage,

    "Number_of_Sequences":
        num_sequences,

    "Sequence_Length":
        seq_len,

    "Number_of_Features":
        num_features,
}


summary_df = pd.DataFrame(
    [summary_data]
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


print(
    "\nModel: TimeGAN_Stock_V5"
)

print(
    f"\nMean absolute feature mean difference: "
    f"{mean_difference_summary:.6f}"
)

print(
    f"Mean absolute feature std difference: "
    f"{std_difference_summary:.6f}"
)

print(
    f"Mean pairwise correlation error: "
    f"{mean_pairwise_correlation_error:.6f}"
)

print(
    f"Mean lag-1 difference: "
    f"{mean_lag1_difference:.6f}"
)

print(
    f"Mean KS statistic: "
    f"{mean_ks:.6f}"
)

print(
    f"Mean Wasserstein distance: "
    f"{mean_wasserstein:.6f}"
)

print(
    f"Mean range coverage: "
    f"{mean_range_coverage:.6f}"
)


# ============================================================
# 9. V1/V2/V3/V4/V5 COMPARISON
# ============================================================

print("\n")
print("=" * 78)
print("9. V1 / V2 / V3 / V4 / V5 COMPARISON")
print("=" * 78)


# ------------------------------------------------------------
# Known validation results from previous experiments.
#
# These values are historical experiment results and are
# included only for direct comparison.
# ------------------------------------------------------------

comparison_rows = [

    {
        "Model":
            "V1",

        "Mean_Difference":
            0.047181,

        "Std_Difference":
            0.046435,

        "Correlation_Error":
            0.518209,

        "Lag1_Difference":
            0.440136,

        "KS_Statistic":
            0.360809,

        "Wasserstein":
            0.131823,

        "Range_Coverage":
            np.nan,
    },

    {
        "Model":
            "V2",

        "Mean_Difference":
            0.007817,

        "Std_Difference":
            0.143029,

        "Correlation_Error":
            0.080896,

        "Lag1_Difference":
            0.084444,

        "KS_Statistic":
            0.132178,

        "Wasserstein":
            0.058600,

        "Range_Coverage":
            np.nan,
    },

    {
        "Model":
            "V3",

        "Mean_Difference":
            0.009438,

        "Std_Difference":
            0.338303,

        "Correlation_Error":
            0.495528,

        "Lag1_Difference":
            0.533190,

        "KS_Statistic":
            0.504256,

        "Wasserstein":
            0.179924,

        "Range_Coverage":
            0.001023,
    },

    {
        "Model":
            "V4",

        "Mean_Difference":
            0.032927,

        "Std_Difference":
            0.323094,

        "Correlation_Error":
            0.296655,

        "Lag1_Difference":
            0.908865,

        "KS_Statistic":
            0.433255,

        "Wasserstein":
            0.174476,

        "Range_Coverage":
            0.013839,
    },

    {
        "Model":
            "V5",

        "Mean_Difference":
            mean_difference_summary,

        "Std_Difference":
            std_difference_summary,

        "Correlation_Error":
            mean_pairwise_correlation_error,

        "Lag1_Difference":
            mean_lag1_difference,

        "KS_Statistic":
            mean_ks,

        "Wasserstein":
            mean_wasserstein,

        "Range_Coverage":
            mean_range_coverage,
    }
]


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    VERSION_COMPARISON_OUTPUT,
    index=False
)


print(
    "\n"
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 10. SIMPLE NUMERICAL DIAGNOSIS
# ============================================================

print("\n")
print("=" * 78)
print("10. V5 DIAGNOSTIC CHECK")
print("=" * 78)


# ------------------------------------------------------------
# Detect severe range compression.
# ------------------------------------------------------------

compression_flags = []

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_std = np.std(
        real_data[:, index]
    )

    synthetic_std = np.std(
        synthetic_data[:, index]
    )

    if real_std > 1e-12:

        std_ratio = (
            synthetic_std /
            real_std
        )

    else:

        std_ratio = 0.0

    compression_flags.append(
        std_ratio
    )

    print(
        f"\n{feature}"
    )

    print(
        f"  Synthetic/Real std ratio: "
        f"{std_ratio:.6f}"
    )


mean_std_ratio = np.mean(
    compression_flags
)


print(
    f"\nMean synthetic/real std ratio: "
    f"{mean_std_ratio:.6f}"
)


# ------------------------------------------------------------
# Diagnostic classification
# ------------------------------------------------------------

if mean_std_ratio < 0.10:

    print(
        "\nWARNING: V5 shows severe variance "
        "compression / mode collapse."
    )

elif mean_std_ratio < 0.50:

    print(
        "\nWARNING: V5 shows substantial "
        "variance compression."
    )

else:

    print(
        "\nV5 variance is not severely compressed."
    )


if mean_pairwise_correlation_error < 0.10:

    print(
        "Correlation structure is close to "
        "the real data."
    )

elif mean_pairwise_correlation_error < 0.20:

    print(
        "Correlation structure shows moderate "
        "error."
    )

else:

    print(
        "Correlation structure shows substantial "
        "error."
    )


if mean_lag1_difference < 0.15:

    print(
        "Temporal lag-1 behavior is relatively close."
    )

else:

    print(
        "Temporal lag-1 behavior shows substantial "
        "difference."
    )


if mean_range_coverage < 0.10:

    print(
        "WARNING: V5 has poor real-data range coverage."
    )

elif mean_range_coverage < 0.50:

    print(
        "V5 has moderate range coverage."
    )

else:

    print(
        "V5 has substantial range coverage."
    )


# ============================================================
# OUTPUT FILE LIST
# ============================================================

print("\n")
print("=" * 78)
print("VALIDATION FILES CREATED")
print("=" * 78)

output_files = [

    BASIC_OUTPUT,
    CORRELATION_ERROR_OUTPUT,
    DISTRIBUTION_OUTPUT,
    EXTREME_OUTPUT,
    LAG_OUTPUT,
    QUANTILE_OUTPUT,
    RANGE_OUTPUT,
    REAL_CORRELATION_OUTPUT,
    SYNTHETIC_CORRELATION_OUTPUT,
    SUMMARY_OUTPUT,
    VERSION_COMPARISON_OUTPUT,
]

for number, path in enumerate(
    output_files,
    start=1
):

    print(
        f"{number}. {path}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 78)
print("TIMEGAN STOCK V5 VALIDATION COMPLETED")
print("=" * 78)

print(
    "\nV5 has been validated against the REAL "
    "stock-market dataset."
)

print(
    "\nThe V1/V2/V3/V4/V5 comparison has also "
    "been saved."
)

print(
    "\nImportant:"
)

print(
    "Do not select V5 for the final MacroStress-GAN "
    "pipeline until the validation metrics have been "
    "reviewed."
)

print("\n")