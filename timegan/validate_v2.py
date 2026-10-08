# ============================================================
# TimeGAN V2 - Sequence-Aware Validation
# ============================================================
#
# Purpose:
# Validate TimeGAN V2 synthetic financial return/change
# sequences against real financial return/change data.
#
# Important:
# - Uses joblib because the V2 scaler was saved using joblib.dump()
# - Does NOT retrain the model
# - Does NOT regenerate synthetic data
# - Avoids mixing Day 30 of one sequence with Day 1 of another
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\LTM_Project\LTM_Project1-Macroeconomics_crisis-"


# ============================================================
# INPUT FILES
# ============================================================

REAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "macro_stress_5vars_processed.csv"
)

SYNTHETIC_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "synthetic_v2_financial_data.csv"
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v2_scaler.pkl"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TIMEGAN FEATURES
# ============================================================

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]


# ============================================================
# SEQUENCE LENGTH
# ============================================================

SEQUENCE_LENGTH = 30


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_lag1_autocorrelation(sequence):
    """
    Calculate lag-1 autocorrelation for one sequence.

    The calculation is performed inside one sequence only.
    This prevents sequence boundaries from affecting the result.
    """

    values = np.asarray(
        sequence,
        dtype=float
    )

    # Need enough observations
    if len(values) < 3:
        return np.nan

    # Constant sequence has undefined autocorrelation
    if np.std(values) == 0:
        return np.nan

    autocorrelation = pd.Series(
        values
    ).autocorr(
        lag=1
    )

    return autocorrelation


# ============================================================
# SEQUENCE LAG-1 AUTOCORRELATION
# ============================================================

def calculate_sequence_lag1(sequences):
    """
    Calculate lag-1 autocorrelation independently for
    every sequence and every feature.

    Returns:
        mean autocorrelation
        standard deviation of autocorrelation
    """

    results = []

    for seq in sequences:

        feature_results = []

        for feature_index in range(
            seq.shape[1]
        ):

            autocorrelation = (
                calculate_lag1_autocorrelation(
                    seq[:, feature_index]
                )
            )

            feature_results.append(
                autocorrelation
            )

        results.append(
            feature_results
        )

    results = np.asarray(
        results,
        dtype=float
    )

    mean_ac = np.nanmean(
        results,
        axis=0
    )

    std_ac = np.nanstd(
        results,
        axis=0
    )

    return mean_ac, std_ac


# ============================================================
# SEQUENCE VOLATILITY
# ============================================================

def calculate_sequence_volatility(sequences):
    """
    Calculate volatility separately inside every 30-day
    sequence.

    Because each sequence contains exactly 30 observations,
    this produces one volatility value per sequence
    for each feature.
    """

    results = []

    for seq in sequences:

        feature_volatility = []

        for feature_index in range(
            seq.shape[1]
        ):

            values = seq[:, feature_index]

            volatility = np.std(
                values,
                ddof=1
            )

            feature_volatility.append(
                volatility
            )

        results.append(
            feature_volatility
        )

    results = np.asarray(
        results,
        dtype=float
    )

    mean_volatility = np.nanmean(
        results,
        axis=0
    )

    std_volatility = np.nanstd(
        results,
        axis=0
    )

    return (
        mean_volatility,
        std_volatility
    )


# ============================================================
# CREATE REAL 30-DAY WINDOWS
# ============================================================

def create_real_windows(
    real_values,
    sequence_length=30
):
    """
    Convert the continuous real dataset into 30-day windows.

    This creates a comparable structure:

        Real:
        30-day window
        30-day window
        30-day window
        ...

        Synthetic:
        Sequence 1 = 30 days
        Sequence 2 = 30 days
        Sequence 3 = 30 days
        ...

    Any remaining observations that do not form a complete
    30-day window are excluded.
    """

    usable_length = (
        len(real_values)
        // sequence_length
    ) * sequence_length

    trimmed = real_values[
        :usable_length
    ]

    windows = trimmed.reshape(
        -1,
        sequence_length,
        real_values.shape[1]
    )

    return windows


# ============================================================
# CALCULATE WINDOW CORRELATION
# ============================================================

def calculate_sequence_correlations(
    sequences
):
    """
    Calculate a correlation matrix for every sequence/window
    and then average the matrices.

    This is sequence-aware and therefore avoids mixing
    different synthetic sequences.
    """

    correlation_matrices = []

    for seq in sequences:

        df = pd.DataFrame(
            seq,
            columns=FEATURES
        )

        corr = df.corr().values

        correlation_matrices.append(
            corr
        )

    correlation_matrices = np.asarray(
        correlation_matrices,
        dtype=float
    )

    average_correlation = np.nanmean(
        correlation_matrices,
        axis=0
    )

    return average_correlation


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TIMEGAN V2 - SEQUENCE-AWARE VALIDATION")
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\nChecking input files...")

required_files = [
    REAL_FILE,
    SYNTHETIC_FILE,
    SCALER_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print(
        f"Found: {file_path}"
    )


# ============================================================
# LOAD REAL DATA
# ============================================================

print("\nLoading real data...")

real_df = pd.read_csv(
    REAL_FILE
)

print(
    f"Real dataset shape: "
    f"{real_df.shape}"
)


# ============================================================
# LOAD SYNTHETIC DATA
# ============================================================

print("\nLoading synthetic data...")

synthetic_df = pd.read_csv(
    SYNTHETIC_FILE
)

print(
    f"Synthetic dataset shape: "
    f"{synthetic_df.shape}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nChecking required features...")

for feature in FEATURES:

    if feature not in real_df.columns:

        raise ValueError(
            f"Missing feature in real dataset: {feature}"
        )

    if feature not in synthetic_df.columns:

        raise ValueError(
            f"Missing feature in synthetic dataset: {feature}"
        )

print(
    "All five TimeGAN V2 features found."
)


# ============================================================
# LOAD V2 SCALER
# ============================================================

print("\nLoading V2 scaler...")

# IMPORTANT:
# The scaler was created using joblib.dump()
# Therefore it must be loaded using joblib.load().

scaler = joblib.load(
    SCALER_FILE
)

print(
    "V2 scaler loaded successfully."
)


# ============================================================
# PREPARE REAL VALUES
# ============================================================

print("\nPreparing real return/change data...")

real_values = (
    real_df[FEATURES]
    .astype(float)
    .values
)


# Replace invalid values if any
real_values = np.nan_to_num(
    real_values,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print(
    f"Real feature matrix: "
    f"{real_values.shape}"
)


# ============================================================
# PREPARE SYNTHETIC VALUES
# ============================================================

print(
    "\nLoading and inverse-transforming "
    "synthetic data..."
)

synthetic_scaled = (
    synthetic_df[FEATURES]
    .astype(float)
    .values
)


# ============================================================
# CHECK SYNTHETIC SCALED DATA
# ============================================================

print(
    f"Synthetic scaled matrix: "
    f"{synthetic_scaled.shape}"
)

print(
    f"Synthetic scaled minimum: "
    f"{synthetic_scaled.min():.6f}"
)

print(
    f"Synthetic scaled maximum: "
    f"{synthetic_scaled.max():.6f}"
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

synthetic_values = scaler.inverse_transform(
    synthetic_scaled
)


# Replace invalid values
synthetic_values = np.nan_to_num(
    synthetic_values,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


print(
    "Synthetic data successfully "
    "converted back to original return/change units."
)

print(
    f"Inverse-transformed minimum: "
    f"{synthetic_values.min():.8f}"
)

print(
    f"Inverse-transformed maximum: "
    f"{synthetic_values.max():.8f}"
)


# ============================================================
# CHECK SYNTHETIC SEQUENCE COUNT
# ============================================================

total_synthetic_rows = len(
    synthetic_values
)

if (
    total_synthetic_rows
    % SEQUENCE_LENGTH
    != 0
):

    print(
        "\nWARNING:"
        "\nSynthetic row count is not exactly divisible "
        "by sequence length."
    )

    usable_synthetic_rows = (
        total_synthetic_rows
        // SEQUENCE_LENGTH
    ) * SEQUENCE_LENGTH

else:

    usable_synthetic_rows = (
        total_synthetic_rows
    )


# ============================================================
# RESHAPE SYNTHETIC DATA
# ============================================================

synthetic_values = synthetic_values[
    :usable_synthetic_rows
]

synthetic_sequences = synthetic_values.reshape(
    -1,
    SEQUENCE_LENGTH,
    len(FEATURES)
)


print(
    f"\nSynthetic sequences: "
    f"{synthetic_sequences.shape}"
)


print(
    f"Number of synthetic sequences: "
    f"{synthetic_sequences.shape[0]}"
)


print(
    f"Days per sequence: "
    f"{synthetic_sequences.shape[1]}"
)


print(
    f"Features per sequence: "
    f"{synthetic_sequences.shape[2]}"
)


# ============================================================
# CREATE REAL 30-DAY WINDOWS
# ============================================================

print(
    "\nCreating real 30-day windows..."
)

real_windows = create_real_windows(
    real_values,
    SEQUENCE_LENGTH
)


print(
    f"Real 30-day windows: "
    f"{real_windows.shape}"
)


print(
    f"Number of real windows: "
    f"{real_windows.shape[0]}"
)


# ============================================================
# MARGINAL DISTRIBUTION VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MARGINAL DISTRIBUTION VALIDATION"
)

print(
    "=" * 70
)


summary_rows = []


for feature_index, feature in enumerate(
    FEATURES
):

    real_feature = real_values[
        :, feature_index
    ]

    synthetic_feature = synthetic_values[
        :, feature_index
    ]


    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    real_mean = np.mean(
        real_feature
    )

    synthetic_mean = np.mean(
        synthetic_feature
    )


    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    real_std = np.std(
        real_feature,
        ddof=1
    )

    synthetic_std = np.std(
        synthetic_feature,
        ddof=1
    )


    # --------------------------------------------------------
    # MINIMUM
    # --------------------------------------------------------

    real_min = np.min(
        real_feature
    )

    synthetic_min = np.min(
        synthetic_feature
    )


    # --------------------------------------------------------
    # MAXIMUM
    # --------------------------------------------------------

    real_max = np.max(
        real_feature
    )

    synthetic_max = np.max(
        synthetic_feature
    )


    # --------------------------------------------------------
    # KS TEST
    # --------------------------------------------------------

    ks_statistic, ks_pvalue = ks_2samp(
        real_feature,
        synthetic_feature
    )


    # --------------------------------------------------------
    # ABSOLUTE MEAN DIFFERENCE
    # --------------------------------------------------------

    absolute_mean_difference = abs(
        synthetic_mean
        - real_mean
    )


    # --------------------------------------------------------
    # STANDARD DEVIATION RATIO
    # --------------------------------------------------------

    if real_std != 0:

        std_ratio = (
            synthetic_std
            / real_std
        )

    else:

        std_ratio = np.nan


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    summary_rows.append({

        "Feature": feature,

        "Real_Mean": real_mean,

        "Synthetic_Mean": synthetic_mean,

        "Absolute_Mean_Difference":
            absolute_mean_difference,

        "Real_Std": real_std,

        "Synthetic_Std": synthetic_std,

        "Std_Ratio": std_ratio,

        "Real_Min": real_min,

        "Synthetic_Min": synthetic_min,

        "Real_Max": real_max,

        "Synthetic_Max": synthetic_max,

        "KS_Statistic": ks_statistic,

        "KS_pvalue": ks_pvalue
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SEQUENCE-AWARE LAG-1 AUTOCORRELATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SEQUENCE-AWARE TEMPORAL VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nCalculating real sequence/window "
    "lag-1 autocorrelation..."
)

real_lag1, real_lag1_std = (
    calculate_sequence_lag1(
        real_windows
    )
)


print(
    "Calculating synthetic sequence "
    "lag-1 autocorrelation..."
)

synthetic_lag1, synthetic_lag1_std = (
    calculate_sequence_lag1(
        synthetic_sequences
    )
)


# ============================================================
# SEQUENCE-AWARE VOLATILITY
# ============================================================

print(
    "\n"
    "Calculating real 30-day "
    "sequence volatility..."
)

real_volatility, real_volatility_std = (
    calculate_sequence_volatility(
        real_windows
    )
)


print(
    "Calculating synthetic 30-day "
    "sequence volatility..."
)

synthetic_volatility, synthetic_volatility_std = (
    calculate_sequence_volatility(
        synthetic_sequences
    )
)


# ============================================================
# ADD TEMPORAL METRICS TO SUMMARY
# ============================================================

summary_df[
    "Real_Lag1_Autocorrelation"
] = real_lag1


summary_df[
    "Synthetic_Lag1_Autocorrelation"
] = synthetic_lag1


summary_df[
    "Lag1_Absolute_Difference"
] = np.abs(
    synthetic_lag1
    - real_lag1
)


summary_df[
    "Real_Lag1_AC_Std"
] = real_lag1_std


summary_df[
    "Synthetic_Lag1_AC_Std"
] = synthetic_lag1_std


# ============================================================
# ADD VOLATILITY METRICS
# ============================================================

summary_df[
    "Real_30D_Volatility"
] = real_volatility


summary_df[
    "Synthetic_30D_Volatility"
] = synthetic_volatility


summary_df[
    "Volatility_Absolute_Difference"
] = np.abs(
    synthetic_volatility
    - real_volatility
)


summary_df[
    "Volatility_Ratio"
] = (
    synthetic_volatility
    / real_volatility
)


# ============================================================
# CORRELATION VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "30-DAY WINDOW CORRELATION VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nCalculating real 30-day "
    "window correlation..."
)

real_corr = calculate_sequence_correlations(
    real_windows
)


print(
    "Calculating synthetic 30-day "
    "sequence correlation..."
)

synthetic_corr = calculate_sequence_correlations(
    synthetic_sequences
)


# ============================================================
# CORRELATION DIFFERENCE
# ============================================================

correlation_difference = (
    synthetic_corr
    - real_corr
)


# ============================================================
# UPPER TRIANGLE
# ============================================================

upper_triangle = np.triu_indices(
    len(FEATURES),
    k=1
)


absolute_correlation_errors = np.abs(
    correlation_difference[
        upper_triangle
    ]
)


average_correlation_error = np.mean(
    absolute_correlation_errors
)


maximum_correlation_error = np.max(
    absolute_correlation_errors
)


# ============================================================
# SAVE CORRELATION MATRICES
# ============================================================

real_corr_df = pd.DataFrame(
    real_corr,
    index=FEATURES,
    columns=FEATURES
)


synthetic_corr_df = pd.DataFrame(
    synthetic_corr,
    index=FEATURES,
    columns=FEATURES
)


correlation_difference_df = pd.DataFrame(
    correlation_difference,
    index=FEATURES,
    columns=FEATURES
)


real_corr_file = os.path.join(
    OUTPUT_DIR,
    "timegan_v2_correlation_window_real.csv"
)


synthetic_corr_file = os.path.join(
    OUTPUT_DIR,
    "timegan_v2_correlation_window_synthetic.csv"
)


difference_corr_file = os.path.join(
    OUTPUT_DIR,
    "timegan_v2_correlation_window_difference.csv"
)


real_corr_df.to_csv(
    real_corr_file
)


synthetic_corr_df.to_csv(
    synthetic_corr_file
)


correlation_difference_df.to_csv(
    difference_corr_file
)


# ============================================================
# OVERALL METRICS
# ============================================================

average_std_ratio = np.mean(
    summary_df[
        "Std_Ratio"
    ]
)


average_ks = np.mean(
    summary_df[
        "KS_Statistic"
    ]
)


average_lag1_error = np.mean(
    summary_df[
        "Lag1_Absolute_Difference"
    ]
)


average_volatility_ratio = np.mean(
    summary_df[
        "Volatility_Ratio"
    ]
)


average_volatility_error = np.mean(
    summary_df[
        "Volatility_Absolute_Difference"
    ]
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "timegan_v2_validation_sequence_aware.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

overall_file = os.path.join(
    OUTPUT_DIR,
    "timegan_v2_validation_overall_sequence_aware.csv"
)


overall_df = pd.DataFrame({

    "Metric": [

        "Average_Std_Ratio",

        "Average_KS_Statistic",

        "Average_Sequence_Aware_Lag1_Error",

        "Average_30D_Volatility_Ratio",

        "Average_30D_Volatility_Absolute_Error",

        "Average_30D_Correlation_Error",

        "Maximum_30D_Correlation_Error"
    ],

    "Value": [

        average_std_ratio,

        average_ks,

        average_lag1_error,

        average_volatility_ratio,

        average_volatility_error,

        average_correlation_error,

        maximum_correlation_error
    ]
})


overall_df.to_csv(
    overall_file,
    index=False
)


# ============================================================
# PRINT FEATURE RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SEQUENCE-AWARE VALIDATION RESULTS"
)

print(
    "=" * 70
)


display_columns = [

    "Feature",

    "Real_Mean",

    "Synthetic_Mean",

    "Absolute_Mean_Difference",

    "Real_Std",

    "Synthetic_Std",

    "Std_Ratio",

    "KS_Statistic",

    "Real_Lag1_Autocorrelation",

    "Synthetic_Lag1_Autocorrelation",

    "Lag1_Absolute_Difference",

    "Real_30D_Volatility",

    "Synthetic_30D_Volatility",

    "Volatility_Ratio"
]


print(
    summary_df[
        display_columns
    ].round(6).to_string(
        index=False
    )
)


# ============================================================
# PRINT REAL CORRELATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "REAL 30-DAY WINDOW CORRELATION"
)

print(
    "=" * 70
)


print(
    real_corr_df.round(4)
)


# ============================================================
# PRINT SYNTHETIC CORRELATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SYNTHETIC 30-DAY SEQUENCE CORRELATION"
)

print(
    "=" * 70
)


print(
    synthetic_corr_df.round(4)
)


# ============================================================
# PRINT CORRELATION DIFFERENCE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CORRELATION DIFFERENCE"
)

print(
    "=" * 70
)


print(
    correlation_difference_df.round(4)
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "OVERALL VALIDATION METRICS"
)

print(
    "=" * 70
)


print(
    f"\nAverage Std Ratio: "
    f"{average_std_ratio:.6f}"
)


print(
    f"Average KS Statistic: "
    f"{average_ks:.6f}"
)


print(
    f"Average Sequence-Aware Lag-1 Error: "
    f"{average_lag1_error:.6f}"
)


print(
    f"Average 30D Volatility Ratio: "
    f"{average_volatility_ratio:.6f}"
)


print(
    f"Average 30D Volatility Absolute Error: "
    f"{average_volatility_error:.6f}"
)


print(
    f"Average 30D Correlation Error: "
    f"{average_correlation_error:.6f}"
)


print(
    f"Maximum 30D Correlation Error: "
    f"{maximum_correlation_error:.6f}"
)


# ============================================================
# PRINT OUTPUT FILES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FILES SAVED"
)

print(
    "=" * 70
)


print(
    f"\n1. {summary_file}"
)


print(
    f"2. {real_corr_file}"
)


print(
    f"3. {synthetic_corr_file}"
)


print(
    f"4. {difference_corr_file}"
)


print(
    f"5. {overall_file}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TIMEGAN V2 SEQUENCE-AWARE "
    "VALIDATION COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nNo model training was performed."
)

print(
    "No synthetic data was regenerated."
)

print(
    "Existing V2 model and generated data were preserved."
)