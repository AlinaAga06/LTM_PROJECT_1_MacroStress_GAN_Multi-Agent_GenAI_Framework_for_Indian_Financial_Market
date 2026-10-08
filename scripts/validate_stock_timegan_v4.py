# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V4 VALIDATION
# ============================================================
#
# Purpose:
#   Validate Stock Market TimeGAN V4 synthetic data against
#   the REAL stock-market dataset.
#
# V4 model:
#   TimeGAN_Stock_V4
#
# Validation:
#   1. Basic statistics
#   2. Correlation matrices
#   3. Correlation error
#   4. Lag-1 temporal correlation
#   5. Extreme event coverage
#   6. KS test
#   7. Wasserstein distance
#   8. Quantile comparison
#   9. Range coverage
#  10. Overall validation summary
#
# IMPORTANT:
#   V1, V2 and V3 files are NOT modified.
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ------------------------------------------------------------
# REAL DATA
# ------------------------------------------------------------

REAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_features.csv",
)

# ------------------------------------------------------------
# V4 SYNTHETIC DATA
# ------------------------------------------------------------

SYNTHETIC_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "stock",
    "stock_v4_synthetic_sequences.npy",
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "validation",
    "stock_v4",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# QUANTILES
# ============================================================

QUANTILES = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_float(value):
    """
    Convert NumPy scalar to normal Python float.
    """
    try:
        return float(value)
    except Exception:
        return np.nan


def lag1_correlation(values):
    """
    Calculate lag-1 autocorrelation.

    Example:
        x[t] compared with x[t-1]
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    x1 = values[:-1]
    x2 = values[1:]

    if np.std(x1) == 0 or np.std(x2) == 0:
        return np.nan

    return float(np.corrcoef(x1, x2)[0, 1])


def flatten_sequences(sequences):
    """
    Convert:

        (num_sequences, sequence_length, features)

    into:

        (num_sequences * sequence_length, features)
    """

    if sequences.ndim != 3:
        raise ValueError(
            f"Expected 3D synthetic array, got shape {sequences.shape}"
        )

    return sequences.reshape(
        -1,
        sequences.shape[-1]
    )


# ============================================================
# LOAD REAL DATA
# ============================================================

print_section("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V4 VALIDATION")

print("\nProject root:")
print(PROJECT_ROOT)

print("\nReal dataset:")
print(REAL_FILE)

print("\nSynthetic V4 dataset:")
print(SYNTHETIC_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# REAL DATA VALIDATION
# ============================================================

print_section("LOADING REAL DATA")

if not os.path.exists(REAL_FILE):
    raise FileNotFoundError(
        f"Real dataset not found:\n{REAL_FILE}"
    )

real_df = pd.read_csv(REAL_FILE)

missing_features = [
    feature
    for feature in FEATURE_NAMES
    if feature not in real_df.columns
]

if missing_features:
    raise ValueError(
        "Missing required features in real dataset:\n"
        + "\n".join(missing_features)
    )

real_data = real_df[FEATURE_NAMES].copy()

real_data = real_data.replace(
    [np.inf, -np.inf],
    np.nan
)

if real_data.isna().any().any():
    print("\nWARNING: NaN values detected in real data.")
    print(real_data.isna().sum())

    real_data = real_data.dropna()

real_values = real_data.values.astype(np.float64)

print("\nReal data shape:")
print(real_values.shape)

print("\nReal NaN count:")
print(np.isnan(real_values).sum())

print("\nReal Inf count:")
print(np.isinf(real_values).sum())


# ============================================================
# LOAD SYNTHETIC V4 DATA
# ============================================================

print_section("LOADING V4 SYNTHETIC DATA")

if not os.path.exists(SYNTHETIC_FILE):
    raise FileNotFoundError(
        f"V4 synthetic dataset not found:\n{SYNTHETIC_FILE}"
    )

synthetic_sequences = np.load(SYNTHETIC_FILE)

print("\nSynthetic sequence shape:")
print(synthetic_sequences.shape)

if synthetic_sequences.ndim != 3:
    raise ValueError(
        "V4 synthetic data must be 3-dimensional."
    )

num_sequences = synthetic_sequences.shape[0]
sequence_length = synthetic_sequences.shape[1]
feature_dimension = synthetic_sequences.shape[2]

print("\nNumber of sequences :", num_sequences)
print("Sequence length     :", sequence_length)
print("Feature dimension   :", feature_dimension)

if feature_dimension != len(FEATURE_NAMES):
    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, "
        f"but received {feature_dimension}."
    )

print("\nSynthetic NaN count:")
print(np.isnan(synthetic_sequences).sum())

print("\nSynthetic Inf count:")
print(np.isinf(synthetic_sequences).sum())

if np.isnan(synthetic_sequences).any():
    raise ValueError("Synthetic V4 data contains NaN values.")

if np.isinf(synthetic_sequences).any():
    raise ValueError("Synthetic V4 data contains Inf values.")


# ============================================================
# FLATTEN SYNTHETIC DATA
# ============================================================

synthetic_values = flatten_sequences(
    synthetic_sequences
).astype(np.float64)

print("\nFlattened synthetic shape:")
print(synthetic_values.shape)

print("\nSynthetic minimum:")
print(np.min(synthetic_values))

print("\nSynthetic maximum:")
print(np.max(synthetic_values))


# ============================================================
# 1. BASIC STATISTICS
# ============================================================

print_section("1. BASIC STATISTICS")

basic_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    real_feature = real_values[:, index]
    synthetic_feature = synthetic_values[:, index]

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

    min_difference = abs(
        real_min - synthetic_min
    )

    max_difference = abs(
        real_max - synthetic_max
    )

    basic_rows.append({
        "Feature": feature,

        "Real_Mean": real_mean,
        "Synthetic_Mean": synthetic_mean,
        "Mean_Difference": mean_difference,

        "Real_Std": real_std,
        "Synthetic_Std": synthetic_std,
        "Std_Difference": std_difference,

        "Real_Min": real_min,
        "Synthetic_Min": synthetic_min,
        "Min_Difference": min_difference,

        "Real_Max": real_max,
        "Synthetic_Max": synthetic_max,
        "Max_Difference": max_difference,
    })

    print(f"\n{feature}")

    print(
        f"  Real Mean      : {real_mean:.8f}"
    )

    print(
        f"  Synthetic Mean : {synthetic_mean:.8f}"
    )

    print(
        f"  Mean Difference: {mean_difference:.8f}"
    )

    print(
        f"  Real Std       : {real_std:.8f}"
    )

    print(
        f"  Synthetic Std  : {synthetic_std:.8f}"
    )

    print(
        f"  Std Difference : {std_difference:.8f}"
    )

    print(
        f"  Real Min       : {real_min:.8f}"
    )

    print(
        f"  Synthetic Min  : {synthetic_min:.8f}"
    )

    print(
        f"  Real Max       : {real_max:.8f}"
    )

    print(
        f"  Synthetic Max  : {synthetic_max:.8f}"
    )


basic_df = pd.DataFrame(basic_rows)

basic_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_basic_statistics.csv"
)

basic_df.to_csv(
    basic_output,
    index=False
)

print("\nSaved:")
print(basic_output)


# ============================================================
# 2. CORRELATION MATRICES
# ============================================================

print_section("2. CORRELATION ANALYSIS")

real_corr = np.corrcoef(
    real_values,
    rowvar=False
)

synthetic_corr = np.corrcoef(
    synthetic_values,
    rowvar=False
)

correlation_error = np.abs(
    real_corr - synthetic_corr
)

mean_correlation_error = np.mean(
    correlation_error
)

# Exclude diagonal because diagonal is always 1.
upper_triangle = np.triu(
    np.ones_like(correlation_error, dtype=bool),
    k=1
)

pairwise_correlation_error = (
    correlation_error[upper_triangle]
)

mean_pairwise_correlation_error = np.mean(
    pairwise_correlation_error
)


print("\nReal correlation matrix:")
print(
    pd.DataFrame(
        real_corr,
        index=FEATURE_NAMES,
        columns=FEATURE_NAMES
    ).round(4)
)

print("\nSynthetic correlation matrix:")
print(
    pd.DataFrame(
        synthetic_corr,
        index=FEATURE_NAMES,
        columns=FEATURE_NAMES
    ).round(4)
)

print(
    "\nMean correlation error INCLUDING diagonal:"
)
print(
    f"{mean_correlation_error:.6f}"
)

print(
    "\nMean pairwise correlation error:"
)
print(
    f"{mean_pairwise_correlation_error:.6f}"
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

correlation_error_df = pd.DataFrame(
    correlation_error,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

real_corr_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_real_correlation.csv"
)

synthetic_corr_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_synthetic_correlation.csv"
)

correlation_error_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_correlation_error.csv"
)

real_corr_df.to_csv(real_corr_output)
synthetic_corr_df.to_csv(synthetic_corr_output)
correlation_error_df.to_csv(correlation_error_output)

print("\nSaved:")
print(real_corr_output)
print(synthetic_corr_output)
print(correlation_error_output)


# ============================================================
# 3. LAG-1 TEMPORAL CORRELATION
# ============================================================

print_section("3. LAG-1 TEMPORAL CORRELATION")

lag_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    real_feature = real_values[:, index]

    real_lag1 = lag1_correlation(
        real_feature
    )

    # --------------------------------------------------------
    # SYNTHETIC
    #
    # Calculate lag-1 inside each generated sequence instead
    # of across the artificial boundary between sequences.
    # --------------------------------------------------------

    synthetic_sequence_values = (
        synthetic_sequences[:, :, index]
    )

    sequence_lag_values = []

    for sequence in synthetic_sequence_values:

        value = lag1_correlation(sequence)

        if not np.isnan(value):
            sequence_lag_values.append(value)

    if len(sequence_lag_values) > 0:
        synthetic_lag1 = np.mean(
            sequence_lag_values
        )
    else:
        synthetic_lag1 = np.nan

    lag_difference = abs(
        real_lag1 - synthetic_lag1
    )

    lag_rows.append({
        "Feature": feature,
        "Real_Lag1": real_lag1,
        "Synthetic_Lag1": synthetic_lag1,
        "Lag1_Difference": lag_difference,
    })

    print(f"\n{feature}")

    print(
        f"  Real Lag-1      : {real_lag1:.6f}"
    )

    print(
        f"  Synthetic Lag-1 : {synthetic_lag1:.6f}"
    )

    print(
        f"  Difference      : {lag_difference:.6f}"
    )


lag_df = pd.DataFrame(lag_rows)

lag_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_lag1_comparison.csv"
)

lag_df.to_csv(
    lag_output,
    index=False
)

mean_lag1_difference = lag_df[
    "Lag1_Difference"
].mean()

print(
    "\nMean Lag-1 Difference:"
)

print(
    f"{mean_lag1_difference:.6f}"
)

print("\nSaved:")
print(lag_output)


# ============================================================
# 4. EXTREME EVENT COVERAGE
# ============================================================

print_section("4. EXTREME EVENT COVERAGE")

extreme_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    real_feature = real_values[:, index]
    synthetic_feature = synthetic_values[:, index]

    # --------------------------------------------------------
    # Thresholds are calculated ONLY from REAL data.
    # --------------------------------------------------------

    lower_threshold = np.quantile(
        real_feature,
        0.01
    )

    upper_threshold = np.quantile(
        real_feature,
        0.99
    )

    real_lower_count = np.sum(
        real_feature <= lower_threshold
    )

    real_upper_count = np.sum(
        real_feature >= upper_threshold
    )

    synthetic_lower_count = np.sum(
        synthetic_feature <= lower_threshold
    )

    synthetic_upper_count = np.sum(
        synthetic_feature >= upper_threshold
    )

    real_lower_rate = (
        real_lower_count /
        len(real_feature)
    )

    real_upper_rate = (
        real_upper_count /
        len(real_feature)
    )

    synthetic_lower_rate = (
        synthetic_lower_count /
        len(synthetic_feature)
    )

    synthetic_upper_rate = (
        synthetic_upper_count /
        len(synthetic_feature)
    )

    lower_coverage = (
        synthetic_lower_rate /
        real_lower_rate
        if real_lower_rate > 0
        else np.nan
    )

    upper_coverage = (
        synthetic_upper_rate /
        real_upper_rate
        if real_upper_rate > 0
        else np.nan
    )

    extreme_rows.append({
        "Feature": feature,

        "Real_1pct_Threshold": lower_threshold,
        "Real_99pct_Threshold": upper_threshold,

        "Real_Lower_Count": real_lower_count,
        "Synthetic_Lower_Count": synthetic_lower_count,

        "Real_Upper_Count": real_upper_count,
        "Synthetic_Upper_Count": synthetic_upper_count,

        "Real_Lower_Rate": real_lower_rate,
        "Synthetic_Lower_Rate": synthetic_lower_rate,

        "Real_Upper_Rate": real_upper_rate,
        "Synthetic_Upper_Rate": synthetic_upper_rate,

        "Lower_Tail_Coverage": lower_coverage,
        "Upper_Tail_Coverage": upper_coverage,
    })

    print(f"\n{feature}")

    print(
        f"  Real 1% threshold  : {lower_threshold:.8f}"
    )

    print(
        f"  Real 99% threshold : {upper_threshold:.8f}"
    )

    print(
        f"  Real lower count   : {real_lower_count}"
    )

    print(
        f"  Synthetic lower    : {synthetic_lower_count}"
    )

    print(
        f"  Real upper count   : {real_upper_count}"
    )

    print(
        f"  Synthetic upper    : {synthetic_upper_count}"
    )

    print(
        f"  Lower coverage     : {lower_coverage:.6f}"
    )

    print(
        f"  Upper coverage     : {upper_coverage:.6f}"
    )


extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_extreme_events.csv"
)

extreme_df.to_csv(
    extreme_output,
    index=False
)

print("\nSaved:")
print(extreme_output)


# ============================================================
# 5. DISTRIBUTION TESTS
# ============================================================

print_section("5. DISTRIBUTION TESTS")

distribution_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    real_feature = real_values[:, index]
    synthetic_feature = synthetic_values[:, index]

    # --------------------------------------------------------
    # Kolmogorov-Smirnov
    # --------------------------------------------------------

    ks_result = ks_2samp(
        real_feature,
        synthetic_feature
    )

    ks_statistic = ks_result.statistic
    ks_pvalue = ks_result.pvalue

    # --------------------------------------------------------
    # Wasserstein
    # --------------------------------------------------------

    wasserstein = wasserstein_distance(
        real_feature,
        synthetic_feature
    )

    distribution_rows.append({
        "Feature": feature,
        "KS_Statistic": ks_statistic,
        "KS_PValue": ks_pvalue,
        "Wasserstein_Distance": wasserstein,
    })

    print(f"\n{feature}")

    print(
        f"  KS statistic       : {ks_statistic:.6f}"
    )

    print(
        f"  KS p-value         : {ks_pvalue:.6e}"
    )

    print(
        f"  Wasserstein        : {wasserstein:.6f}"
    )


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_distribution_tests.csv"
)

distribution_df.to_csv(
    distribution_output,
    index=False
)

mean_ks = distribution_df[
    "KS_Statistic"
].mean()

mean_wasserstein = distribution_df[
    "Wasserstein_Distance"
].mean()

print(
    "\nMean KS statistic:"
)

print(
    f"{mean_ks:.6f}"
)

print(
    "\nMean Wasserstein distance:"
)

print(
    f"{mean_wasserstein:.6f}"
)

print("\nSaved:")
print(distribution_output)


# ============================================================
# 6. QUANTILE COMPARISON
# ============================================================

print_section("6. QUANTILE COMPARISON")

quantile_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    real_feature = real_values[:, index]
    synthetic_feature = synthetic_values[:, index]

    real_quantiles = np.quantile(
        real_feature,
        QUANTILES
    )

    synthetic_quantiles = np.quantile(
        synthetic_feature,
        QUANTILES
    )

    for q, real_q, synthetic_q in zip(
        QUANTILES,
        real_quantiles,
        synthetic_quantiles
    ):

        difference = abs(
            real_q - synthetic_q
        )

        quantile_rows.append({
            "Feature": feature,
            "Quantile": q,
            "Real_Value": real_q,
            "Synthetic_Value": synthetic_q,
            "Absolute_Difference": difference,
        })

        print(
            f"{feature:25s} "
            f"Q={q:.2f} "
            f"Real={real_q:.8f} "
            f"Synthetic={synthetic_q:.8f} "
            f"Diff={difference:.8f}"
        )


quantile_df = pd.DataFrame(
    quantile_rows
)

quantile_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_quantile_comparison.csv"
)

quantile_df.to_csv(
    quantile_output,
    index=False
)

print("\nSaved:")
print(quantile_output)


# ============================================================
# 7. RANGE COVERAGE
# ============================================================

print_section("7. RANGE COVERAGE")

range_rows = []

for index, feature in enumerate(FEATURE_NAMES):

    real_feature = real_values[:, index]
    synthetic_feature = synthetic_values[:, index]

    real_min = np.min(real_feature)
    real_max = np.max(real_feature)

    synthetic_min = np.min(synthetic_feature)
    synthetic_max = np.max(synthetic_feature)

    real_range = (
        real_max - real_min
    )

    synthetic_range = (
        synthetic_max - synthetic_min
    )

    if real_range > 0:

        range_coverage = (
            synthetic_range /
            real_range
        )

    else:

        range_coverage = np.nan

    range_rows.append({
        "Feature": feature,

        "Real_Min": real_min,
        "Real_Max": real_max,
        "Real_Range": real_range,

        "Synthetic_Min": synthetic_min,
        "Synthetic_Max": synthetic_max,
        "Synthetic_Range": synthetic_range,

        "Range_Coverage": range_coverage,
    })

    print(f"\n{feature}")

    print(
        f"  Real range      : "
        f"{real_min:.8f} → {real_max:.8f}"
    )

    print(
        f"  Synthetic range : "
        f"{synthetic_min:.8f} → {synthetic_max:.8f}"
    )

    print(
        f"  Range coverage  : "
        f"{range_coverage:.6f}"
    )


range_df = pd.DataFrame(
    range_rows
)

range_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_range_coverage.csv"
)

range_df.to_csv(
    range_output,
    index=False
)

mean_range_coverage = range_df[
    "Range_Coverage"
].mean()

print(
    "\nMean range coverage:"
)

print(
    f"{mean_range_coverage:.6f}"
)

print("\nSaved:")
print(range_output)


# ============================================================
# 8. OVERALL SUMMARY
# ============================================================

print_section("8. V4 VALIDATION SUMMARY")

mean_difference = basic_df[
    "Mean_Difference"
].mean()

std_difference = basic_df[
    "Std_Difference"
].mean()

summary_data = {
    "Model": "TimeGAN_Stock_V4",

    "Real_Rows": len(real_values),

    "Synthetic_Sequences": num_sequences,

    "Sequence_Length": sequence_length,

    "Feature_Dimension": feature_dimension,

    "Mean_Absolute_Feature_Mean_Difference":
        mean_difference,

    "Mean_Absolute_Feature_Std_Difference":
        std_difference,

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
}

summary_df = pd.DataFrame(
    [summary_data]
)

summary_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_validation_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("TIMEGAN STOCK V4 VALIDATION RESULTS")
print("=" * 78)

print(
    f"\nMean absolute feature mean difference : "
    f"{mean_difference:.6f}"
)

print(
    f"Mean absolute feature std difference  : "
    f"{std_difference:.6f}"
)

print(
    f"Mean pairwise correlation error       : "
    f"{mean_pairwise_correlation_error:.6f}"
)

print(
    f"Mean Lag-1 difference                 : "
    f"{mean_lag1_difference:.6f}"
)

print(
    f"Mean KS statistic                     : "
    f"{mean_ks:.6f}"
)

print(
    f"Mean Wasserstein distance             : "
    f"{mean_wasserstein:.6f}"
)

print(
    f"Mean range coverage                   : "
    f"{mean_range_coverage:.6f}"
)


# ============================================================
# V1 / V2 / V3 / V4 COMPARISON
# ============================================================

print_section(
    "V1 / V2 / V3 / V4 COMPARISON"
)

comparison_data = pd.DataFrame({
    "Metric": [
        "Mean difference",
        "Std difference",
        "Correlation error",
        "Lag-1 difference",
        "KS statistic",
        "Wasserstein",
        "Range coverage",
    ],

    "V1": [
        0.047181,
        0.046435,
        0.518209,
        0.440136,
        0.360809,
        0.131823,
        np.nan,
    ],

    "V2": [
        0.007817,
        0.143029,
        0.080896,
        0.084444,
        0.132178,
        0.058600,
        np.nan,
    ],

    "V3": [
        0.009438,
        0.338303,
        0.495528,
        0.533190,
        0.504256,
        0.179924,
        0.001023,
    ],

    "V4": [
        mean_difference,
        std_difference,
        mean_pairwise_correlation_error,
        mean_lag1_difference,
        mean_ks,
        mean_wasserstein,
        mean_range_coverage,
    ],
})

print(
    "\n"
)

print(
    comparison_data.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


comparison_output = os.path.join(
    OUTPUT_DIR,
    "stock_v4_vs_v1_v2_v3_comparison.csv"
)

comparison_data.to_csv(
    comparison_output,
    index=False
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print_section(
    "VALIDATION COMPLETED SUCCESSFULLY"
)

print("\nGenerated validation files:")

output_files = [
    "stock_v4_basic_statistics.csv",
    "stock_v4_correlation_error.csv",
    "stock_v4_distribution_tests.csv",
    "stock_v4_extreme_events.csv",
    "stock_v4_lag1_comparison.csv",
    "stock_v4_quantile_comparison.csv",
    "stock_v4_range_coverage.csv",
    "stock_v4_real_correlation.csv",
    "stock_v4_synthetic_correlation.csv",
    "stock_v4_validation_summary.csv",
    "stock_v4_vs_v1_v2_v3_comparison.csv",
]

for number, filename in enumerate(
    output_files,
    start=1
):

    full_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print(
        f"{number}. {full_path}"
    )


print("\n")
print("=" * 78)
print("V4 VALIDATION FINISHED")
print("=" * 78)