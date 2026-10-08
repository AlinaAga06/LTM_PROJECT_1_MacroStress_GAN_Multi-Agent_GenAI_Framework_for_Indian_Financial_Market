
"""
MACROSTRESS-GAN
Stock Market TimeGAN V6 Validation

Purpose
-------
Compare REAL stock-market features against V6 synthetic features.

Real features:
    NIFTY50_Return
    CRUDE_OIL_Return
    USD_INR_Return
    INDIA_VIX_Change
    INDIA_10Y_YIELD_Change

Validation metrics:
    1. Feature mean error
    2. Feature standard-deviation error
    3. Standard-deviation ratio
    4. Pairwise correlation error
    5. Lag-1 temporal correlation error
    6. Kolmogorov-Smirnov statistic
    7. Wasserstein distance
    8. Tail/range coverage
    9. Extreme-value comparison

The script compares V6 against the REAL data.
It also prints the previously obtained V2 benchmark
for direct comparison.

No model files are modified.
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


# ============================================================
# CONFIGURATION
# ============================================================

REAL_FEATURE_FILE = (
    "data/processed/institutions/stock/v6/"
    "stock_v6_features.csv"
)

REAL_RAW_FILE = (
    "data/processed/institutions/stock/"
    "stock_5vars.csv"
)

V6_SYNTHETIC_FILE = (
    "outputs/synthetic/stock/"
    "stock_v6_synthetic_sequences.npy"
)

SEQ_LEN = 30

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# V2 BENCHMARK
# ============================================================

V2_BENCHMARK = {
    "mean_error": 0.007817,
    "std_error": 0.143029,
    "correlation_error": 0.080896,
    "lag1_error": 0.084444,
    "ks": 0.132178,
    "wasserstein": 0.058600,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def calculate_lag1(data):
    """
    Calculate lag-1 correlation for each feature.

    Parameters
    ----------
    data : numpy.ndarray
        Shape:
            (rows, features)

    Returns
    -------
    numpy.ndarray
        Lag-1 correlation for each feature.
    """

    lag_values = []

    for feature_index in range(data.shape[1]):

        current = data[:-1, feature_index]
        previous = data[1:, feature_index]

        if (
            np.std(current) == 0
            or np.std(previous) == 0
        ):
            lag_values.append(0.0)
        else:
            correlation = np.corrcoef(
                current,
                previous
            )[0, 1]

            lag_values.append(correlation)

    return np.array(lag_values)


def safe_correlation_matrix(data):
    """
    Calculate correlation matrix safely.
    """

    correlation = np.corrcoef(data.T)

    correlation = np.nan_to_num(
        correlation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return correlation


# ============================================================
# START
# ============================================================

print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V6 VALIDATION")
print("=" * 78)

print()
print("Purpose:")
print("Compare V6 synthetic data against REAL stock-market data.")
print()
print("No model or dataset will be modified.")


# ============================================================
# 1. CHECK FILES
# ============================================================

print_section("1. CHECKING REQUIRED FILES")

files_to_check = {
    "Real feature file": REAL_FEATURE_FILE,
    "Real raw dataset": REAL_RAW_FILE,
    "V6 synthetic file": V6_SYNTHETIC_FILE,
}

for name, path in files_to_check.items():

    exists = os.path.exists(path)

    print(
        f"{name:<25}: "
        f"{'FOUND' if exists else 'NOT FOUND'}"
    )

    if not exists:
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# 2. LOAD REAL FEATURES
# ============================================================

print_section("2. LOADING REAL FEATURE DATA")

real_df = pd.read_csv(
    REAL_FEATURE_FILE
)

print(
    f"Real feature file:\n"
    f"{REAL_FEATURE_FILE}"
)

print(
    f"Rows: {len(real_df):,}"
)

print(
    f"Columns: {len(real_df.columns)}"
)

missing_features = [
    feature
    for feature in FEATURE_NAMES
    if feature not in real_df.columns
]

if missing_features:

    raise ValueError(
        "The following required features are missing:\n"
        + "\n".join(missing_features)
    )


real_features = (
    real_df[FEATURE_NAMES]
    .values
    .astype(np.float64)
)

print(
    f"Real feature matrix shape: "
    f"{real_features.shape}"
)

print(
    f"NaN count: "
    f"{np.isnan(real_features).sum():,}"
)

print(
    f"Inf count: "
    f"{np.isinf(real_features).sum():,}"
)


if np.isnan(real_features).any():
    raise ValueError(
        "Real feature data contains NaN values."
    )

if np.isinf(real_features).any():
    raise ValueError(
        "Real feature data contains infinite values."
    )


# ============================================================
# 3. REAL FEATURE STATISTICS
# ============================================================

print_section("3. REAL FEATURE STATISTICS")

print(
    f"{'Feature':<27}"
    f"{'Mean':>14}"
    f"{'Std':>14}"
    f"{'Min':>14}"
    f"{'Max':>14}"
)

print("-" * 83)

for index, feature in enumerate(FEATURE_NAMES):

    values = real_features[:, index]

    print(
        f"{feature:<27}"
        f"{np.mean(values):>14.8f}"
        f"{np.std(values):>14.8f}"
        f"{np.min(values):>14.8f}"
        f"{np.max(values):>14.8f}"
    )


# ============================================================
# 4. LOAD V6 SYNTHETIC DATA
# ============================================================

print_section("4. LOADING V6 SYNTHETIC DATA")

synthetic = np.load(
    V6_SYNTHETIC_FILE
)

print(
    f"V6 synthetic file:\n"
    f"{V6_SYNTHETIC_FILE}"
)

print(
    f"Shape: {synthetic.shape}"
)

print(
    f"Data type: {synthetic.dtype}"
)

print(
    f"NaN count: "
    f"{np.isnan(synthetic).sum():,}"
)

print(
    f"Inf count: "
    f"{np.isinf(synthetic).sum():,}"
)


if synthetic.ndim != 3:

    raise ValueError(
        "Expected V6 synthetic data to have 3 dimensions:\n"
        "(number_of_sequences, sequence_length, features)"
    )


if synthetic.shape[1] != SEQ_LEN:

    raise ValueError(
        f"Expected sequence length {SEQ_LEN}, "
        f"but received {synthetic.shape[1]}"
    )


if synthetic.shape[2] != len(FEATURE_NAMES):

    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, "
        f"but received {synthetic.shape[2]}"
    )


if np.isnan(synthetic).any():

    raise ValueError(
        "V6 synthetic data contains NaN values."
    )


if np.isinf(synthetic).any():

    raise ValueError(
        "V6 synthetic data contains infinite values."
    )


# ============================================================
# 5. FLATTEN SYNTHETIC DATA
# ============================================================

print_section("5. PREPARING SYNTHETIC DATA")

number_sequences = synthetic.shape[0]

synthetic_flat = synthetic.reshape(
    -1,
    len(FEATURE_NAMES)
)

print(
    f"Number of sequences: "
    f"{number_sequences:,}"
)

print(
    f"Sequence length: "
    f"{synthetic.shape[1]}"
)

print(
    f"Feature dimension: "
    f"{synthetic.shape[2]}"
)

print(
    f"Flattened synthetic shape: "
    f"{synthetic_flat.shape}"
)


# ============================================================
# 6. SYNTHETIC FEATURE STATISTICS
# ============================================================

print_section("6. V6 SYNTHETIC FEATURE STATISTICS")

print(
    f"{'Feature':<27}"
    f"{'Mean':>14}"
    f"{'Std':>14}"
    f"{'Min':>14}"
    f"{'Max':>14}"
)

print("-" * 83)

for index, feature in enumerate(FEATURE_NAMES):

    values = synthetic_flat[:, index]

    print(
        f"{feature:<27}"
        f"{np.mean(values):>14.8f}"
        f"{np.std(values):>14.8f}"
        f"{np.min(values):>14.8f}"
        f"{np.max(values):>14.8f}"
    )


# ============================================================
# 7. MEAN VALIDATION
# ============================================================

print_section("7. FEATURE MEAN VALIDATION")

mean_errors = []

print(
    f"{'Feature':<27}"
    f"{'Real Mean':>14}"
    f"{'V6 Mean':>14}"
    f"{'Abs Error':>14}"
)

print("-" * 71)

for index, feature in enumerate(FEATURE_NAMES):

    real_mean = np.mean(
        real_features[:, index]
    )

    synthetic_mean = np.mean(
        synthetic_flat[:, index]
    )

    error = abs(
        real_mean - synthetic_mean
    )

    mean_errors.append(error)

    print(
        f"{feature:<27}"
        f"{real_mean:>14.8f}"
        f"{synthetic_mean:>14.8f}"
        f"{error:>14.8f}"
    )


mean_mean_error = np.mean(
    mean_errors
)

print()
print(
    f"Mean absolute feature mean error: "
    f"{mean_mean_error:.6f}"
)


# ============================================================
# 8. STANDARD DEVIATION VALIDATION
# ============================================================

print_section("8. STANDARD DEVIATION VALIDATION")

std_errors = []
std_ratios = []

print(
    f"{'Feature':<27}"
    f"{'Real Std':>14}"
    f"{'V6 Std':>14}"
    f"{'Abs Error':>14}"
    f"{'Ratio':>12}"
)

print("-" * 83)

for index, feature in enumerate(FEATURE_NAMES):

    real_std = np.std(
        real_features[:, index]
    )

    synthetic_std = np.std(
        synthetic_flat[:, index]
    )

    error = abs(
        real_std - synthetic_std
    )

    if real_std != 0:

        ratio = (
            synthetic_std /
            real_std
        )

    else:

        ratio = 0.0

    std_errors.append(error)
    std_ratios.append(ratio)

    print(
        f"{feature:<27}"
        f"{real_std:>14.8f}"
        f"{synthetic_std:>14.8f}"
        f"{error:>14.8f}"
        f"{ratio:>12.4f}"
    )


mean_std_error = np.mean(
    std_errors
)

mean_std_ratio = np.mean(
    std_ratios
)

print()
print(
    f"Mean absolute feature std error: "
    f"{mean_std_error:.6f}"
)

print(
    f"Mean synthetic/real std ratio: "
    f"{mean_std_ratio:.6f}"
)


# ============================================================
# 9. CORRELATION VALIDATION
# ============================================================

print_section("9. CORRELATION VALIDATION")

real_corr = safe_correlation_matrix(
    real_features
)

synthetic_corr = safe_correlation_matrix(
    synthetic_flat
)

print("REAL CORRELATION MATRIX")
print()

print(
    pd.DataFrame(
        real_corr,
        index=FEATURE_NAMES,
        columns=FEATURE_NAMES
    ).round(4)
)

print()
print("V6 SYNTHETIC CORRELATION MATRIX")
print()

print(
    pd.DataFrame(
        synthetic_corr,
        index=FEATURE_NAMES,
        columns=FEATURE_NAMES
    ).round(4)
)


upper_indices = np.triu_indices(
    len(FEATURE_NAMES),
    k=1
)

correlation_errors = np.abs(
    real_corr[upper_indices]
    -
    synthetic_corr[upper_indices]
)

mean_correlation_error = np.mean(
    correlation_errors
)

print()
print(
    "Pairwise correlation errors:"
)

for pair_index, error in enumerate(
    correlation_errors
):

    i = upper_indices[0][pair_index]
    j = upper_indices[1][pair_index]

    print(
        f"  {FEATURE_NAMES[i]} vs "
        f"{FEATURE_NAMES[j]}: "
        f"{error:.6f}"
    )

print()
print(
    f"Mean pairwise correlation error: "
    f"{mean_correlation_error:.6f}"
)


# ============================================================
# 10. TEMPORAL VALIDATION
# ============================================================

print_section("10. TEMPORAL / LAG-1 VALIDATION")

real_lag1 = calculate_lag1(
    real_features
)

synthetic_lag1_values = []

for sequence_index in range(
    synthetic.shape[0]
):

    sequence = synthetic[
        sequence_index
    ]

    lag1 = calculate_lag1(
        sequence
    )

    synthetic_lag1_values.append(
        lag1
    )


synthetic_lag1_values = np.array(
    synthetic_lag1_values
)

synthetic_lag1 = np.mean(
    synthetic_lag1_values,
    axis=0
)

lag1_errors = np.abs(
    real_lag1 -
    synthetic_lag1
)

mean_lag1_error = np.mean(
    lag1_errors
)

print(
    f"{'Feature':<27}"
    f"{'Real Lag1':>14}"
    f"{'V6 Lag1':>14}"
    f"{'Abs Error':>14}"
)

print("-" * 71)

for index, feature in enumerate(
    FEATURE_NAMES
):

    print(
        f"{feature:<27}"
        f"{real_lag1[index]:>14.8f}"
        f"{synthetic_lag1[index]:>14.8f}"
        f"{lag1_errors[index]:>14.8f}"
    )

print()
print(
    f"Mean lag-1 correlation error: "
    f"{mean_lag1_error:.6f}"
)


# ============================================================
# 11. KS TEST
# ============================================================

print_section("11. KOLMOGOROV-SMIRNOV VALIDATION")

ks_scores = []

print(
    f"{'Feature':<27}"
    f"{'KS Statistic':>18}"
)

print("-" * 48)

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = (
        real_features[:, index]
    )

    synthetic_values = (
        synthetic_flat[:, index]
    )

    ks_result = ks_2samp(
        real_values,
        synthetic_values
    )

    ks_statistic = ks_result.statistic

    ks_scores.append(
        ks_statistic
    )

    print(
        f"{feature:<27}"
        f"{ks_statistic:>18.8f}"
    )


mean_ks = np.mean(
    ks_scores
)

print()
print(
    f"Mean KS statistic: "
    f"{mean_ks:.6f}"
)


# ============================================================
# 12. WASSERSTEIN DISTANCE
# ============================================================

print_section("12. WASSERSTEIN DISTANCE VALIDATION")

wasserstein_scores = []

print(
    f"{'Feature':<27}"
    f"{'Wasserstein':>18}"
)

print("-" * 48)

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = (
        real_features[:, index]
    )

    synthetic_values = (
        synthetic_flat[:, index]
    )

    distance = wasserstein_distance(
        real_values,
        synthetic_values
    )

    wasserstein_scores.append(
        distance
    )

    print(
        f"{feature:<27}"
        f"{distance:>18.8f}"
    )


mean_wasserstein = np.mean(
    wasserstein_scores
)

print()
print(
    f"Mean Wasserstein distance: "
    f"{mean_wasserstein:.6f}"
)


# ============================================================
# 13. TAIL COVERAGE
# ============================================================

print_section("13. TAIL / RANGE COVERAGE")

tail_coverages = []

print(
    f"{'Feature':<27}"
    f"{'Real P01':>13}"
    f"{'Real P99':>13}"
    f"{'V6 P01':>13}"
    f"{'V6 P99':>13}"
)

print("-" * 79)

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = (
        real_features[:, index]
    )

    synthetic_values = (
        synthetic_flat[:, index]
    )

    real_p01 = np.percentile(
        real_values,
        1
    )

    real_p99 = np.percentile(
        real_values,
        99
    )

    synthetic_p01 = np.percentile(
        synthetic_values,
        1
    )

    synthetic_p99 = np.percentile(
        synthetic_values,
        99
    )

    below_real_p01 = np.mean(
        synthetic_values <
        real_p01
    )

    above_real_p99 = np.mean(
        synthetic_values >
        real_p99
    )

    coverage = (
        below_real_p01 +
        above_real_p99
    ) / 2

    tail_coverages.append(
        coverage
    )

    print(
        f"{feature:<27}"
        f"{real_p01:>13.6f}"
        f"{real_p99:>13.6f}"
        f"{synthetic_p01:>13.6f}"
        f"{synthetic_p99:>13.6f}"
    )


mean_tail_coverage = np.mean(
    tail_coverages
)

print()
print(
    f"Mean tail coverage: "
    f"{mean_tail_coverage:.6f}"
)


# ============================================================
# 14. EXTREME VALUE COVERAGE
# ============================================================

print_section("14. EXTREME VALUE COMPARISON")

print(
    f"{'Feature':<27}"
    f"{'Real Min':>14}"
    f"{'V6 Min':>14}"
    f"{'Real Max':>14}"
    f"{'V6 Max':>14}"
)

print("-" * 83)

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = (
        real_features[:, index]
    )

    synthetic_values = (
        synthetic_flat[:, index]
    )

    print(
        f"{feature:<27}"
        f"{np.min(real_values):>14.8f}"
        f"{np.min(synthetic_values):>14.8f}"
        f"{np.max(real_values):>14.8f}"
        f"{np.max(synthetic_values):>14.8f}"
    )


# ============================================================
# 15. PERCENTILE COMPARISON
# ============================================================

print_section("15. DETAILED PERCENTILE COMPARISON")

percentiles = [
    1,
    5,
    25,
    50,
    75,
    95,
    99,
]

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_values = (
        real_features[:, index]
    )

    synthetic_values = (
        synthetic_flat[:, index]
    )

    print()
    print(feature)

    print(
        f"{'Percentile':<15}"
        f"{'Real':>18}"
        f"{'V6 Synthetic':>18}"
        f"{'Absolute Diff':>18}"
    )

    print("-" * 69)

    for percentile in percentiles:

        real_value = np.percentile(
            real_values,
            percentile
        )

        synthetic_value = np.percentile(
            synthetic_values,
            percentile
        )

        difference = abs(
            real_value -
            synthetic_value
        )

        print(
            f"P{percentile:<14}"
            f"{real_value:>18.8f}"
            f"{synthetic_value:>18.8f}"
            f"{difference:>18.8f}"
        )


# ============================================================
# 16. STANDARD DEVIATION RATIO
# ============================================================

print_section("16. STANDARD DEVIATION RATIO")

print(
    f"{'Feature':<27}"
    f"{'Real Std':>15}"
    f"{'V6 Std':>15}"
    f"{'V6 / Real':>15}"
)

print("-" * 72)

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_std = np.std(
        real_features[:, index]
    )

    synthetic_std = np.std(
        synthetic_flat[:, index]
    )

    ratio = (
        synthetic_std /
        real_std
        if real_std != 0
        else 0
    )

    print(
        f"{feature:<27}"
        f"{real_std:>15.8f}"
        f"{synthetic_std:>15.8f}"
        f"{ratio:>15.6f}"
    )


# ============================================================
# 17. V2 VS V6 COMPARISON
# ============================================================

print_section("17. V2 VS V6 COMPARISON")

v6_metrics = {
    "Mean absolute feature mean error":
        mean_mean_error,

    "Mean absolute feature std error":
        mean_std_error,

    "Mean pairwise correlation error":
        mean_correlation_error,

    "Mean lag-1 correlation error":
        mean_lag1_error,

    "Mean KS statistic":
        mean_ks,

    "Mean Wasserstein distance":
        mean_wasserstein,
}


v2_values = {
    "Mean absolute feature mean error":
        V2_BENCHMARK["mean_error"],

    "Mean absolute feature std error":
        V2_BENCHMARK["std_error"],

    "Mean pairwise correlation error":
        V2_BENCHMARK["correlation_error"],

    "Mean lag-1 correlation error":
        V2_BENCHMARK["lag1_error"],

    "Mean KS statistic":
        V2_BENCHMARK["ks"],

    "Mean Wasserstein distance":
        V2_BENCHMARK["wasserstein"],
}


print(
    f"{'Metric':<42}"
    f"{'V2':>14}"
    f"{'V6':>14}"
    f"{'Change':>14}"
)

print("-" * 84)

for metric in v6_metrics:

    v2_value = v2_values[metric]

    v6_value = v6_metrics[metric]

    difference = (
        v6_value -
        v2_value
    )

    print(
        f"{metric:<42}"
        f"{v2_value:>14.6f}"
        f"{v6_value:>14.6f}"
        f"{difference:>14.6f}"
    )


# ============================================================
# 18. V2 VS V6 INTERPRETATION
# ============================================================

print_section("18. V2 VS V6 METRIC INTERPRETATION")

print()
print("For the following metrics, LOWER is better:")
print()
print("  • Mean feature mean error")
print("  • Mean feature std error")
print("  • Correlation error")
print("  • Lag-1 error")
print("  • KS statistic")
print("  • Wasserstein distance")

print()

for metric in v6_metrics:

    v2_value = v2_values[metric]
    v6_value = v6_metrics[metric]

    if v6_value < v2_value:

        result = "V6 improved"

    elif v6_value > v2_value:

        result = "V2 better"

    else:

        result = "Equal"

    print(
        f"{metric:<42}: {result}"
    )


# ============================================================
# 19. VARIANCE QUALITY CHECK
# ============================================================

print_section("19. V6 VARIANCE QUALITY CHECK")

print(
    "Ideal synthetic/real standard-deviation ratio "
    "is approximately 1.0."
)

print()

variance_status = []

for index, feature in enumerate(
    FEATURE_NAMES
):

    real_std = np.std(
        real_features[:, index]
    )

    synthetic_std = np.std(
        synthetic_flat[:, index]
    )

    ratio = (
        synthetic_std /
        real_std
        if real_std != 0
        else 0
    )

    if 0.80 <= ratio <= 1.20:

        status = "GOOD"

    elif 0.50 <= ratio < 0.80:

        status = "MODERATE COMPRESSION"

    elif ratio < 0.50:

        status = "STRONG COMPRESSION"

    else:

        status = "HIGH VARIANCE"

    variance_status.append(
        status
    )

    print(
        f"{feature:<27}"
        f"Ratio={ratio:.4f}    "
        f"{status}"
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print_section("20. FINAL V6 VALIDATION SUMMARY")

print()
print("V6 VALIDATION METRICS")
print()

print(
    f"Mean feature mean error       : "
    f"{mean_mean_error:.6f}"
)

print(
    f"Mean feature std error        : "
    f"{mean_std_error:.6f}"
)

print(
    f"Mean correlation error        : "
    f"{mean_correlation_error:.6f}"
)

print(
    f"Mean lag-1 error              : "
    f"{mean_lag1_error:.6f}"
)

print(
    f"Mean KS statistic             : "
    f"{mean_ks:.6f}"
)

print(
    f"Mean Wasserstein distance     : "
    f"{mean_wasserstein:.6f}"
)

print(
    f"Mean tail coverage            : "
    f"{mean_tail_coverage:.6f}"
)

print(
    f"Mean V6/Real std ratio        : "
    f"{mean_std_ratio:.6f}"
)


# ============================================================
# 21. V2 BENCHMARK
# ============================================================

print()
print("V2 BENCHMARK")
print()

print(
    f"Mean feature mean error       : "
    f"{V2_BENCHMARK['mean_error']:.6f}"
)

print(
    f"Mean feature std error        : "
    f"{V2_BENCHMARK['std_error']:.6f}"
)

print(
    f"Mean correlation error        : "
    f"{V2_BENCHMARK['correlation_error']:.6f}"
)

print(
    f"Mean lag-1 error              : "
    f"{V2_BENCHMARK['lag1_error']:.6f}"
)

print(
    f"Mean KS statistic             : "
    f"{V2_BENCHMARK['ks']:.6f}"
)

print(
    f"Mean Wasserstein distance     : "
    f"{V2_BENCHMARK['wasserstein']:.6f}"
)


# ============================================================
# 22. IMPORTANT CONCLUSION
# ============================================================

print_section("21. CONCLUSION")

print()
print(
    "V6 generation completed successfully."
)

print()
print(
    "The validation above determines whether V6 "
    "is statistically better than the V2 benchmark."
)

print()
print(
    "Do NOT replace V2 with V6 until the validation "
    "metrics have been reviewed."
)

print()
print(
    "If V6 has lower distribution/correlation/temporal "
    "errors but poor tail coverage, it may still need "
    "additional improvement before production use."
)

print()
print(
    "Validation completed successfully."
)

print()
print("=" * 78)
print("END OF V6 VALIDATION")
print("=" * 78)

