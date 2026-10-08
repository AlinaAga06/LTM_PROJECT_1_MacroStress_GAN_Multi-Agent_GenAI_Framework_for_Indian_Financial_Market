# ============================================================
# MACROSTRESS-GAN V3 - VALIDATION SCRIPT
#
# Purpose:
#   Research-quality validation of TimeGAN V3 synthetic
#   financial return/change sequences.
#
# V3 training data:
#   data/processed/timegan_v2_sequences.npy
#
# V3 scaler:
#   data/processed/timegan_v2_scaler.pkl
#
# V3 generated data:
#   outputs/synthetic/synthetic_v3_sequences.npy
#
# Features:
#   1. NIFTY50_Return
#   2. CRUDE_OIL_Return
#   3. USD_INR_Return
#   4. INDIA_VIX_Change
#   5. INDIA_10Y_YIELD_Change
#
# Validation:
#   - Data integrity
#   - Marginal statistics
#   - Distribution similarity
#   - Quantile comparison
#   - Correlation preservation
#   - Lag-1 temporal correlation
#   - Volatility comparison
#   - Tail behavior
#   - VaR
#   - Expected Shortfall
#   - Maximum Drawdown
#
# ============================================================

from pathlib import Path

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
    skew,
    kurtosis,
)


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


REAL_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_sequences.npy"
)


SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_scaler.pkl"
)


SYNTHETIC_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
    / "synthetic_v3_sequences.npy"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "v3"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


SEQ_LENGTH = 30


CONFIDENCE_LEVEL = 0.95


RANDOM_SEED = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(value):

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def flatten_sequences(data):

    return data.reshape(
        -1,
        data.shape[-1]
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print_header("LOADING V3 VALIDATION DATA")

    if not REAL_SEQUENCE_PATH.exists():

        raise FileNotFoundError(
            f"Real sequence file not found:\n"
            f"{REAL_SEQUENCE_PATH}"
        )


    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Scaler file not found:\n"
            f"{SCALER_PATH}"
        )


    if not SYNTHETIC_SEQUENCE_PATH.exists():

        raise FileNotFoundError(
            f"Synthetic V3 sequence file not found:\n"
            f"{SYNTHETIC_SEQUENCE_PATH}"
        )


    print(
        f"Real sequences     : {REAL_SEQUENCE_PATH}"
    )

    print(
        f"Scaler             : {SCALER_PATH}"
    )

    print(
        f"Synthetic sequences: {SYNTHETIC_SEQUENCE_PATH}"
    )


    real_scaled = np.load(
        REAL_SEQUENCE_PATH
    )


    synthetic_scaled = np.load(
        SYNTHETIC_SEQUENCE_PATH
    )


    scaler = joblib.load(
        SCALER_PATH
    )


    print()
    print(
        f"Real shape         : {real_scaled.shape}"
    )

    print(
        f"Synthetic shape    : {synthetic_scaled.shape}"
    )


    return (
        real_scaled,
        synthetic_scaled,
        scaler
    )


# ============================================================
# DATA INTEGRITY
# ============================================================

def validate_integrity(
    real_scaled,
    synthetic_scaled
):

    print_header("DATA INTEGRITY VALIDATION")


    checks = {}


    checks["real_nan"] = int(
        np.isnan(real_scaled).sum()
    )


    checks["real_inf"] = int(
        np.isinf(real_scaled).sum()
    )


    checks["synthetic_nan"] = int(
        np.isnan(synthetic_scaled).sum()
    )


    checks["synthetic_inf"] = int(
        np.isinf(synthetic_scaled).sum()
    )


    checks["real_is_3d"] = (
        real_scaled.ndim == 3
    )


    checks["synthetic_is_3d"] = (
        synthetic_scaled.ndim == 3
    )


    checks["real_sequence_length"] = (
        real_scaled.shape[1]
        if real_scaled.ndim == 3
        else None
    )


    checks["synthetic_sequence_length"] = (
        synthetic_scaled.shape[1]
        if synthetic_scaled.ndim == 3
        else None
    )


    checks["real_feature_count"] = (
        real_scaled.shape[2]
        if real_scaled.ndim == 3
        else None
    )


    checks["synthetic_feature_count"] = (
        synthetic_scaled.shape[2]
        if synthetic_scaled.ndim == 3
        else None
    )


    for key, value in checks.items():

        print(
            f"{key:30s}: {value}"
        )


    if (
        checks["real_nan"] > 0
        or checks["real_inf"] > 0
    ):

        raise ValueError(
            "Real dataset contains NaN/Inf values."
        )


    if (
        checks["synthetic_nan"] > 0
        or checks["synthetic_inf"] > 0
    ):

        raise ValueError(
            "Synthetic dataset contains NaN/Inf values."
        )


    if real_scaled.ndim != 3:

        raise ValueError(
            "Real data must be 3-dimensional."
        )


    if synthetic_scaled.ndim != 3:

        raise ValueError(
            "Synthetic data must be 3-dimensional."
        )


    if real_scaled.shape[1] != SEQ_LENGTH:

        raise ValueError(
            "Unexpected real sequence length."
        )


    if synthetic_scaled.shape[1] != SEQ_LENGTH:

        raise ValueError(
            "Unexpected synthetic sequence length."
        )


    if real_scaled.shape[2] != len(FEATURES):

        raise ValueError(
            "Unexpected real feature count."
        )


    if synthetic_scaled.shape[2] != len(FEATURES):

        raise ValueError(
            "Unexpected synthetic feature count."
        )


# ============================================================
# INVERSE TRANSFORMATION
# ============================================================

def inverse_transform_data(
    real_scaled,
    synthetic_scaled,
    scaler
):

    print_header(
        "INVERSE TRANSFORMING FINANCIAL FEATURES"
    )


    real_flat = flatten_sequences(
        real_scaled
    )


    synthetic_flat = flatten_sequences(
        synthetic_scaled
    )


    print(
        f"Real flattened shape      : "
        f"{real_flat.shape}"
    )

    print(
        f"Synthetic flattened shape : "
        f"{synthetic_flat.shape}"
    )


    real_flat_original = scaler.inverse_transform(
        real_flat
    )


    synthetic_flat_original = scaler.inverse_transform(
        synthetic_flat
    )


    real_data = real_flat_original.reshape(
        real_scaled.shape
    )


    synthetic_data = synthetic_flat_original.reshape(
        synthetic_scaled.shape
    )


    print()
    print(
        f"Real inverse shape        : "
        f"{real_data.shape}"
    )

    print(
        f"Synthetic inverse shape   : "
        f"{synthetic_data.shape}"
    )


    print()
    print("Inverse transformation completed.")


    return (
        real_data,
        synthetic_data
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary_statistics(
    real_data,
    synthetic_data
):

    print_header(
        "MARGINAL STATISTICS"
    )


    real_flat = flatten_sequences(
        real_data
    )


    synthetic_flat = flatten_sequences(
        synthetic_data
    )


    rows = []


    for i, feature in enumerate(FEATURES):

        real_values = real_flat[:, i]
        synthetic_values = synthetic_flat[:, i]


        row = {

            "Feature": feature,

            "Real_Mean":
                np.mean(real_values),

            "Synthetic_Mean":
                np.mean(synthetic_values),

            "Real_Std":
                np.std(real_values),

            "Synthetic_Std":
                np.std(synthetic_values),

            "Real_Min":
                np.min(real_values),

            "Synthetic_Min":
                np.min(synthetic_values),

            "Real_Max":
                np.max(real_values),

            "Synthetic_Max":
                np.max(synthetic_values),

            "Real_Skewness":
                skew(real_values),

            "Synthetic_Skewness":
                skew(synthetic_values),

            "Real_Kurtosis":
                kurtosis(
                    real_values,
                    fisher=True
                ),

            "Synthetic_Kurtosis":
                kurtosis(
                    synthetic_values,
                    fisher=True
                ),

        }


        rows.append(row)


    result = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "v3_marginal_statistics.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# QUANTILE COMPARISON
# ============================================================

def calculate_quantiles(
    real_data,
    synthetic_data
):

    print_header(
        "QUANTILE COMPARISON"
    )


    real_flat = flatten_sequences(
        real_data
    )


    synthetic_flat = flatten_sequences(
        synthetic_data
    )


    quantiles = [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]


    rows = []


    for i, feature in enumerate(FEATURES):

        real_values = real_flat[:, i]

        synthetic_values = synthetic_flat[:, i]


        real_q = np.quantile(
            real_values,
            quantiles
        )


        synthetic_q = np.quantile(
            synthetic_values,
            quantiles
        )


        for q, rv, sv in zip(
            quantiles,
            real_q,
            synthetic_q
        ):

            rows.append({

                "Feature": feature,

                "Quantile": q,

                "Real":
                    rv,

                "Synthetic":
                    sv,

                "Absolute_Difference":
                    abs(rv - sv)

            })


    result = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "v3_quantiles.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# DISTRIBUTION SIMILARITY
# ============================================================

def calculate_distribution_metrics(
    real_data,
    synthetic_data
):

    print_header(
        "DISTRIBUTION SIMILARITY"
    )


    real_flat = flatten_sequences(
        real_data
    )


    synthetic_flat = flatten_sequences(
        synthetic_data
    )


    rows = []


    for i, feature in enumerate(FEATURES):

        real_values = real_flat[:, i]

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

            "KS_Statistic":
                ks_stat,

            "KS_p_value":
                ks_pvalue,

            "Wasserstein_Distance":
                wasserstein

        })


    result = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "v3_distribution_metrics.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# CORRELATION MATRIX
# ============================================================

def calculate_correlation_validation(
    real_data,
    synthetic_data
):

    print_header(
        "CROSS-VARIABLE CORRELATION"
    )


    real_flat = flatten_sequences(
        real_data
    )


    synthetic_flat = flatten_sequences(
        synthetic_data
    )


    real_df = pd.DataFrame(
        real_flat,
        columns=FEATURES
    )


    synthetic_df = pd.DataFrame(
        synthetic_flat,
        columns=FEATURES
    )


    real_corr = real_df.corr()

    synthetic_corr = synthetic_df.corr()


    difference = (
        synthetic_corr
        - real_corr
    )


    mae = np.mean(
        np.abs(
            difference.values
        )
    )


    print("Real correlation matrix:")
    print(real_corr.round(4))


    print()
    print("Synthetic correlation matrix:")
    print(synthetic_corr.round(4))


    print()
    print(
        "Mean absolute correlation error: "
        f"{mae:.6f}"
    )


    real_corr.to_csv(
        OUTPUT_DIR
        / "v3_real_correlation.csv"
    )


    synthetic_corr.to_csv(
        OUTPUT_DIR
        / "v3_synthetic_correlation.csv"
    )


    difference.to_csv(
        OUTPUT_DIR
        / "v3_correlation_difference.csv"
    )


    return {
        "real": real_corr,
        "synthetic": synthetic_corr,
        "difference": difference,
        "mae": mae
    }


# ============================================================
# LAG-1 TEMPORAL CORRELATION
# ============================================================

def calculate_temporal_validation(
    real_data,
    synthetic_data
):

    print_header(
        "TEMPORAL LAG-1 VALIDATION"
    )


    rows = []


    for i, feature in enumerate(FEATURES):

        real_lags = []

        synthetic_lags = []


        for sequence in real_data:

            current = sequence[:-1, i]

            next_value = sequence[1:, i]


            if (
                np.std(current) > 0
                and np.std(next_value) > 0
            ):

                corr = np.corrcoef(
                    current,
                    next_value
                )[0, 1]

                if np.isfinite(corr):

                    real_lags.append(
                        corr
                    )


        for sequence in synthetic_data:

            current = sequence[:-1, i]

            next_value = sequence[1:, i]


            if (
                np.std(current) > 0
                and np.std(next_value) > 0
            ):

                corr = np.corrcoef(
                    current,
                    next_value
                )[0, 1]

                if np.isfinite(corr):

                    synthetic_lags.append(
                        corr
                    )


        real_mean = (
            np.mean(real_lags)
            if real_lags
            else np.nan
        )


        synthetic_mean = (
            np.mean(synthetic_lags)
            if synthetic_lags
            else np.nan
        )


        rows.append({

            "Feature": feature,

            "Real_Lag1":
                real_mean,

            "Synthetic_Lag1":
                synthetic_mean,

            "Absolute_Difference":
                abs(
                    real_mean
                    - synthetic_mean
                )

        })


    result = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "v3_temporal_lag1.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# VOLATILITY VALIDATION
# ============================================================

def calculate_volatility_validation(
    real_data,
    synthetic_data
):

    print_header(
        "VOLATILITY VALIDATION"
    )


    rows = []


    for i, feature in enumerate(FEATURES):

        real_vol = []

        synthetic_vol = []


        for sequence in real_data:

            if len(sequence) >= 2:

                real_vol.append(
                    np.std(
                        sequence[:, i]
                    )
                )


        for sequence in synthetic_data:

            if len(sequence) >= 2:

                synthetic_vol.append(
                    np.std(
                        sequence[:, i]
                    )
                )


        rows.append({

            "Feature": feature,

            "Real_Mean_Sequence_Volatility":
                np.mean(real_vol),

            "Synthetic_Mean_Sequence_Volatility":
                np.mean(synthetic_vol),

            "Real_Volatility_Median":
                np.median(real_vol),

            "Synthetic_Volatility_Median":
                np.median(synthetic_vol),

            "Absolute_Mean_Difference":
                abs(
                    np.mean(real_vol)
                    - np.mean(synthetic_vol)
                )

        })


    result = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "v3_volatility_validation.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# RISK METRICS
# ============================================================

def calculate_var(
    returns,
    confidence=0.95
):

    losses = -np.asarray(
        returns
    )

    return np.quantile(
        losses,
        confidence
    )


def calculate_expected_shortfall(
    returns,
    confidence=0.95
):

    losses = -np.asarray(
        returns
    )

    var = np.quantile(
        losses,
        confidence
    )


    tail = losses[
        losses >= var
    ]


    if len(tail) == 0:

        return var


    return np.mean(tail)


def calculate_max_drawdown(
    returns
):

    returns = np.asarray(
        returns,
        dtype=float
    )


    wealth = np.cumprod(
        1.0 + returns
    )


    running_max = np.maximum.accumulate(
        wealth
    )


    drawdown = (
        wealth / running_max
    ) - 1.0


    return np.min(
        drawdown
    )


def calculate_risk_validation(
    real_data,
    synthetic_data
):

    print_header(
        "FINANCIAL RISK VALIDATION"
    )


    # --------------------------------------------------------
    # NIFTY50 returns
    # --------------------------------------------------------

    feature_index = FEATURES.index(
        "NIFTY50_Return"
    )


    real_returns = flatten_sequences(
        real_data
    )[:, feature_index]


    synthetic_returns = flatten_sequences(
        synthetic_data
    )[:, feature_index]


    # --------------------------------------------------------
    # Risk metrics
    # --------------------------------------------------------

    real_var = calculate_var(
        real_returns,
        CONFIDENCE_LEVEL
    )


    synthetic_var = calculate_var(
        synthetic_returns,
        CONFIDENCE_LEVEL
    )


    real_es = calculate_expected_shortfall(
        real_returns,
        CONFIDENCE_LEVEL
    )


    synthetic_es = calculate_expected_shortfall(
        synthetic_returns,
        CONFIDENCE_LEVEL
    )


    # --------------------------------------------------------
    # Sequence-level MDD
    #
    # We calculate MDD independently within each sequence
    # rather than treating the synthetic windows as one
    # continuous 30,000-day market history.
    # --------------------------------------------------------

    real_mdd = []

    synthetic_mdd = []


    for sequence in real_data:

        real_mdd.append(
            calculate_max_drawdown(
                sequence[:, feature_index]
            )
        )


    for sequence in synthetic_data:

        synthetic_mdd.append(
            calculate_max_drawdown(
                sequence[:, feature_index]
            )
        )


    result = pd.DataFrame({

        "Metric": [

            "VaR_95",

            "Expected_Shortfall_95",

            "Mean_30D_Maximum_Drawdown",

            "Median_30D_Maximum_Drawdown",

            "Worst_30D_Maximum_Drawdown"

        ],

        "Real": [

            real_var,

            real_es,

            np.mean(real_mdd),

            np.median(real_mdd),

            np.min(real_mdd)

        ],

        "Synthetic": [

            synthetic_var,

            synthetic_es,

            np.mean(synthetic_mdd),

            np.median(synthetic_mdd),

            np.min(synthetic_mdd)

        ]

    })


    result["Absolute_Difference"] = (
        np.abs(
            result["Real"]
            - result["Synthetic"]
        )
    )


    output_path = (
        OUTPUT_DIR
        / "v3_risk_metrics.csv"
    )


    result.to_csv(
        output_path,
        index=False
    )


    print(
        result.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved: {output_path}"
    )


    return result


# ============================================================
# OVERALL VALIDATION SUMMARY
# ============================================================

def create_validation_summary(
    marginal,
    distribution,
    correlation,
    temporal,
    volatility,
    risk
):

    print_header(
        "CREATING V3 VALIDATION SUMMARY"
    )


    summary = {

        "project":
            "MacroStress-GAN",

        "model":
            "TimeGAN V3",

        "sequence_length":
            SEQ_LENGTH,

        "features":
            FEATURES,

        "real_sequence_count":
            None,

        "synthetic_sequence_count":
            None,

        "correlation_mae":
            safe_float(
                correlation["mae"]
            ),

        "mean_ks_statistic":
            safe_float(
                distribution[
                    "KS_Statistic"
                ].mean()
            ),

        "mean_wasserstein_distance":
            safe_float(
                distribution[
                    "Wasserstein_Distance"
                ].mean()
            ),

        "mean_temporal_lag1_difference":
            safe_float(
                temporal[
                    "Absolute_Difference"
                ].mean()
            ),

        "mean_volatility_difference":
            safe_float(
                volatility[
                    "Absolute_Mean_Difference"
                ].mean()
            ),

        "risk_metrics_file":
            "v3_risk_metrics.csv"

    }


    output_path = (
        OUTPUT_DIR
        / "v3_validation_summary.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    print(
        f"Saved: {output_path}"
    )


    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN V3 - VALIDATION")
    print("=" * 78)

    print()
    print(
        "Purpose:"
    )

    print(
        "Validate synthetic financial return/change "
        "sequences against the real V2 training sequences."
    )

    print()
    print(
        f"Confidence level: {CONFIDENCE_LEVEL:.0%}"
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        real_scaled,
        synthetic_scaled,
        scaler
    ) = load_data()


    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    validate_integrity(
        real_scaled,
        synthetic_scaled
    )


    # --------------------------------------------------------
    # Inverse transform
    # --------------------------------------------------------

    (
        real_data,
        synthetic_data
    ) = inverse_transform_data(
        real_scaled,
        synthetic_scaled,
        scaler
    )


    # --------------------------------------------------------
    # Verify inverse-transformed values
    # --------------------------------------------------------

    print_header(
        "INVERSE-TRANSFORMED RANGE CHECK"
    )


    print(
        f"Real minimum      : "
        f"{real_data.min():.8f}"
    )

    print(
        f"Real maximum      : "
        f"{real_data.max():.8f}"
    )

    print(
        f"Synthetic minimum : "
        f"{synthetic_data.min():.8f}"
    )

    print(
        f"Synthetic maximum : "
        f"{synthetic_data.max():.8f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    marginal = calculate_summary_statistics(
        real_data,
        synthetic_data
    )


    quantiles = calculate_quantiles(
        real_data,
        synthetic_data
    )


    distribution = calculate_distribution_metrics(
        real_data,
        synthetic_data
    )


    correlation = calculate_correlation_validation(
        real_data,
        synthetic_data
    )


    temporal = calculate_temporal_validation(
        real_data,
        synthetic_data
    )


    volatility = calculate_volatility_validation(
        real_data,
        synthetic_data
    )


    risk = calculate_risk_validation(
        real_data,
        synthetic_data
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_validation_summary(
        marginal,
        distribution,
        correlation,
        temporal,
        volatility,
        risk
    )


    # --------------------------------------------------------
    # Save inverse-transformed datasets
    # --------------------------------------------------------

    real_flat = flatten_sequences(
        real_data
    )


    synthetic_flat = flatten_sequences(
        synthetic_data
    )


    real_df = pd.DataFrame(
        real_flat,
        columns=FEATURES
    )


    synthetic_df = pd.DataFrame(
        synthetic_flat,
        columns=FEATURES
    )


    real_df.to_csv(
        OUTPUT_DIR
        / "real_v3_validation_data.csv",
        index=False
    )


    synthetic_df.to_csv(
        OUTPUT_DIR
        / "synthetic_v3_inverse_transformed.csv",
        index=False
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("V3 VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 78)

    print()
    print(
        f"Real sequences      : "
        f"{real_scaled.shape[0]}"
    )

    print(
        f"Synthetic sequences : "
        f"{synthetic_scaled.shape[0]}"
    )

    print(
        f"Sequence length     : "
        f"{SEQ_LENGTH}"
    )

    print(
        f"Features            : "
        f"{len(FEATURES)}"
    )

    print()
    print(
        f"Validation output   : "
        f"{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    for file in sorted(
        OUTPUT_DIR.glob("*")
    ):

        print(
            f"  - {file.name}"
        )


    print()
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()