import os
import json
import numpy as np
import pandas as pd
import joblib

from scipy.stats import ks_2samp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REAL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v34",
    "timegan_v34_sequences.npy"
)

SYNTHETIC_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "synthetic_v34_sequences.npy"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v2_scaler.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "validation",
    "v34",
    "complete"
)

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

N_FEATURES = len(FEATURES)

NIFTY = 0


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
    0.99
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def inverse_transform_sequences(
    sequences,
    scaler
):

    original_shape = sequences.shape

    flat = sequences.reshape(
        -1,
        original_shape[-1]
    )

    inverse = scaler.inverse_transform(
        flat
    )

    return inverse.reshape(
        original_shape
    )


def correlation_matrix(data):

    flat = data.reshape(
        -1,
        data.shape[-1]
    )

    return np.corrcoef(
        flat,
        rowvar=False
    )


def mean_absolute_correlation_error(
    real,
    synthetic
):

    real_corr = correlation_matrix(
        real
    )

    synthetic_corr = correlation_matrix(
        synthetic
    )

    mask = ~np.eye(
        real_corr.shape[0],
        dtype=bool
    )

    error = np.abs(
        real_corr - synthetic_corr
    )

    return float(
        error[mask].mean()
    )


def lag_one_correlation(
    data
):

    values = []

    for feature_index in range(
        data.shape[-1]
    ):

        x = data[:, :-1, feature_index].reshape(-1)

        y = data[:, 1:, feature_index].reshape(-1)

        if (
            np.std(x) == 0
            or np.std(y) == 0
        ):

            values.append(
                np.nan
            )

        else:

            values.append(
                np.corrcoef(
                    x,
                    y
                )[0, 1]
            )

    return np.array(
        values,
        dtype=float
    )


def lag_one_error(
    real,
    synthetic
):

    real_lag = lag_one_correlation(
        real
    )

    synthetic_lag = lag_one_correlation(
        synthetic
    )

    errors = np.abs(
        real_lag - synthetic_lag
    )

    return (
        real_lag,
        synthetic_lag,
        errors
    )


def rolling_volatility(
    data,
    window=30
):

    values = []

    for sequence in data:

        if sequence.shape[0] < window:

            continue

        for feature_index in range(
            data.shape[-1]
        ):

            feature_values = sequence[
                :,
                feature_index
            ]

            returns = pd.Series(
                feature_values
            )

            volatility = (
                returns
                .rolling(window)
                .std()
                .dropna()
                .values
            )

            if len(volatility) > 0:

                values.extend(
                    volatility.tolist()
                )

    return np.array(
        values,
        dtype=float
    )


def feature_rolling_volatility(
    data,
    feature_index,
    window=30
):

    values = []

    for sequence in data:

        series = pd.Series(
            sequence[:, feature_index]
        )

        vol = (
            series
            .rolling(window)
            .std()
            .dropna()
            .values
        )

        if len(vol) > 0:

            values.extend(
                vol.tolist()
            )

    return np.array(
        values,
        dtype=float
    )


def calculate_var(
    values,
    confidence=0.95
):

    return float(
        -np.quantile(
            values,
            1.0 - confidence
        )
    )


def calculate_expected_shortfall(
    values,
    confidence=0.95
):

    threshold = np.quantile(
        values,
        1.0 - confidence
    )

    tail = values[
        values <= threshold
    ]

    if len(tail) == 0:

        return float("nan")

    return float(
        -tail.mean()
    )


def calculate_drawdowns(
    data,
    feature_index=NIFTY
):

    all_drawdowns = []

    for sequence in data:

        values = sequence[
            :,
            feature_index
        ]

        cumulative = np.cumprod(
            1.0 + values
        )

        running_max = np.maximum.accumulate(
            cumulative
        )

        drawdown = (
            cumulative / running_max
        ) - 1.0

        all_drawdowns.append(
            drawdown
        )

    return np.array(
        all_drawdowns
    )


def summarize_drawdown(
    data
):

    drawdowns = calculate_drawdowns(
        data
    )

    sequence_mdds = drawdowns.min(
        axis=1
    )

    return {
        "mean_mdd":
            float(
                sequence_mdds.mean()
            ),

        "median_mdd":
            float(
                np.median(
                    sequence_mdds
                )
            ),

        "worst_mdd":
            float(
                sequence_mdds.min()
            ),

        "best_mdd":
            float(
                sequence_mdds.max()
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MacroStress-GAN V3.4")
    print("COMPLETE STATISTICAL VALIDATION")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. CHECK FILES
    # --------------------------------------------------------

    print("\n1. CHECKING INPUT FILES")
    print("-" * 80)

    for path in [
        REAL_PATH,
        SYNTHETIC_PATH,
        SCALER_PATH
    ]:

        check_file(path)

        print(
            f"PASS: {path}"
        )

    # --------------------------------------------------------
    # 2. LOAD
    # --------------------------------------------------------

    print("\n2. LOADING DATA")
    print("-" * 80)

    real_scaled = np.load(
        REAL_PATH
    )

    synthetic_scaled = np.load(
        SYNTHETIC_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"Real scaled shape: "
        f"{real_scaled.shape}"
    )

    print(
        f"Synthetic scaled shape: "
        f"{synthetic_scaled.shape}"
    )

    # --------------------------------------------------------
    # 3. INVERSE TRANSFORM
    # --------------------------------------------------------

    print("\n3. INVERSE TRANSFORM")
    print("-" * 80)

    real = inverse_transform_sequences(
        real_scaled,
        scaler
    )

    synthetic = inverse_transform_sequences(
        synthetic_scaled,
        scaler
    )

    print(
        f"Real shape: "
        f"{real.shape}"
    )

    print(
        f"Synthetic shape: "
        f"{synthetic.shape}"
    )

    if not np.isfinite(real).all():

        raise ValueError(
            "Real data contains NaN/Inf."
        )

    if not np.isfinite(synthetic).all():

        raise ValueError(
            "Synthetic data contains NaN/Inf."
        )

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    real_flat = real.reshape(
        -1,
        N_FEATURES
    )

    synthetic_flat = synthetic.reshape(
        -1,
        N_FEATURES
    )

    # ========================================================
    # 4. MEAN / STANDARD DEVIATION
    # ========================================================

    print("\n4. MEAN / STANDARD DEVIATION")
    print("-" * 80)

    real_mean = real_flat.mean(
        axis=0
    )

    synthetic_mean = synthetic_flat.mean(
        axis=0
    )

    real_std = real_flat.std(
        axis=0
    )

    synthetic_std = synthetic_flat.std(
        axis=0
    )

    mean_std_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        mean_error = abs(
            real_mean[i]
            - synthetic_mean[i]
        )

        std_error = abs(
            real_std[i]
            - synthetic_std[i]
        )

        mean_std_rows.append({

            "Feature":
                feature,

            "Real_Mean":
                real_mean[i],

            "Synthetic_Mean":
                synthetic_mean[i],

            "Absolute_Mean_Error":
                mean_error,

            "Real_Std":
                real_std[i],

            "Synthetic_Std":
                synthetic_std[i],

            "Absolute_Std_Error":
                std_error
        })

        print(
            f"{feature:<25} "
            f"Mean: "
            f"{real_mean[i]: .8f} / "
            f"{synthetic_mean[i]: .8f} | "
            f"Std: "
            f"{real_std[i]: .8f} / "
            f"{synthetic_std[i]: .8f}"
        )

    mean_std_df = pd.DataFrame(
        mean_std_rows
    )

    mean_std_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mean_std_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # 5. CORRELATION
    # ========================================================

    print("\n5. CORRELATION ANALYSIS")
    print("-" * 80)

    real_corr = correlation_matrix(
        real
    )

    synthetic_corr = correlation_matrix(
        synthetic
    )

    corr_error = np.abs(
        real_corr - synthetic_corr
    )

    correlation_df = pd.DataFrame(
        corr_error,
        index=FEATURES,
        columns=FEATURES
    )

    correlation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "correlation_absolute_error.csv"
        )
    )

    print("\nReal correlation matrix:")

    print(
        pd.DataFrame(
            real_corr,
            index=FEATURES,
            columns=FEATURES
        ).round(4)
    )

    print("\nSynthetic correlation matrix:")

    print(
        pd.DataFrame(
            synthetic_corr,
            index=FEATURES,
            columns=FEATURES
        ).round(4)
    )

    correlation_mae = mean_absolute_correlation_error(
        real,
        synthetic
    )

    print(
        f"\nMean absolute correlation error: "
        f"{correlation_mae:.6f}"
    )

    # ========================================================
    # 6. LAG-1 TEMPORAL DYNAMICS
    # ========================================================

    print("\n6. TEMPORAL DYNAMICS")
    print("-" * 80)

    (
        real_lag,
        synthetic_lag,
        lag_errors
    ) = lag_one_error(
        real,
        synthetic
    )

    lag_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        lag_rows.append({

            "Feature":
                feature,

            "Real_Lag1":
                real_lag[i],

            "Synthetic_Lag1":
                synthetic_lag[i],

            "Absolute_Error":
                lag_errors[i]
        })

        print(
            f"{feature:<25} "
            f"Real={real_lag[i]: .6f} | "
            f"Synthetic={synthetic_lag[i]: .6f} | "
            f"Error={lag_errors[i]: .6f}"
        )

    lag_df = pd.DataFrame(
        lag_rows
    )

    lag_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "lag1_temporal_comparison.csv"
        ),
        index=False
    )

    avg_lag_error = float(
        np.nanmean(
            lag_errors
        )
    )

    print(
        f"\nAverage lag-1 error: "
        f"{avg_lag_error:.6f}"
    )

    # ========================================================
    # 7. 30-DAY VOLATILITY
    # ========================================================

    print("\n7. 30-DAY VOLATILITY")
    print("-" * 80)

    volatility_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        real_vol = feature_rolling_volatility(
            real,
            i,
            window=30
        )

        synthetic_vol = feature_rolling_volatility(
            synthetic,
            i,
            window=30
        )

        real_vol_mean = float(
            real_vol.mean()
        )

        synthetic_vol_mean = float(
            synthetic_vol.mean()
        )

        difference = abs(
            real_vol_mean
            - synthetic_vol_mean
        )

        volatility_rows.append({

            "Feature":
                feature,

            "Real_30D_Volatility":
                real_vol_mean,

            "Synthetic_30D_Volatility":
                synthetic_vol_mean,

            "Absolute_Difference":
                difference
        })

        print(
            f"{feature:<25} "
            f"Real={real_vol_mean:.8f} | "
            f"Synthetic={synthetic_vol_mean:.8f} | "
            f"Diff={difference:.8f}"
        )

    volatility_df = pd.DataFrame(
        volatility_rows
    )

    volatility_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "volatility_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # 8. KS TEST
    # ========================================================

    print("\n8. DISTRIBUTION TEST — KS STATISTIC")
    print("-" * 80)

    ks_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        result = ks_2samp(
            real_flat[:, i],
            synthetic_flat[:, i]
        )

        ks_rows.append({

            "Feature":
                feature,

            "KS_Statistic":
                float(result.statistic),

            "P_Value":
                float(result.pvalue)
        })

        print(
            f"{feature:<25} "
            f"KS={result.statistic:.6f} | "
            f"p={result.pvalue:.6g}"
        )

    ks_df = pd.DataFrame(
        ks_rows
    )

    ks_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ks_test.csv"
        ),
        index=False
    )

    # ========================================================
    # 9. QUANTILES
    # ========================================================

    print("\n9. QUANTILE ANALYSIS")
    print("-" * 80)

    quantile_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        real_quantiles = np.quantile(
            real_flat[:, i],
            QUANTILES
        )

        synthetic_quantiles = np.quantile(
            synthetic_flat[:, i],
            QUANTILES
        )

        for q, real_q, synthetic_q in zip(
            QUANTILES,
            real_quantiles,
            synthetic_quantiles
        ):

            quantile_rows.append({

                "Feature":
                    feature,

                "Quantile":
                    q,

                "Real":
                    real_q,

                "Synthetic":
                    synthetic_q,

                "Absolute_Error":
                    abs(
                        real_q
                        - synthetic_q
                    )
            })

            print(
                f"{feature:<25} "
                f"q={q:.2f} "
                f"Real={real_q: .8f} "
                f"Synthetic={synthetic_q: .8f}"
            )

    quantile_df = pd.DataFrame(
        quantile_rows
    )

    quantile_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "quantile_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # 10. NIFTY VaR
    # ========================================================

    print("\n10. NIFTY VaR / EXPECTED SHORTFALL")
    print("-" * 80)

    real_nifty = real_flat[
        :,
        NIFTY
    ]

    synthetic_nifty = synthetic_flat[
        :,
        NIFTY
    ]

    real_var95 = calculate_var(
        real_nifty,
        confidence=0.95
    )

    synthetic_var95 = calculate_var(
        synthetic_nifty,
        confidence=0.95
    )

    real_es95 = calculate_expected_shortfall(
        real_nifty,
        confidence=0.95
    )

    synthetic_es95 = calculate_expected_shortfall(
        synthetic_nifty,
        confidence=0.95
    )

    print(
        f"VaR95 Real      : {real_var95:.8f}"
    )

    print(
        f"VaR95 Synthetic : {synthetic_var95:.8f}"
    )

    print(
        f"ES95 Real       : {real_es95:.8f}"
    )

    print(
        f"ES95 Synthetic  : {synthetic_es95:.8f}"
    )

    risk_df = pd.DataFrame([{

        "Metric":
            "VaR95",

        "Real":
            real_var95,

        "Synthetic":
            synthetic_var95,

        "Absolute_Error":
            abs(
                real_var95
                - synthetic_var95
            )
    }, {

        "Metric":
            "Expected_Shortfall95",

        "Real":
            real_es95,

        "Synthetic":
            synthetic_es95,

        "Absolute_Error":
            abs(
                real_es95
                - synthetic_es95
            )
    }])

    risk_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "risk_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # 11. MAXIMUM DRAWDOWN
    # ========================================================

    print("\n11. NIFTY MAXIMUM DRAWDOWN")
    print("-" * 80)

    real_drawdown = summarize_drawdown(
        real
    )

    synthetic_drawdown = summarize_drawdown(
        synthetic
    )

    drawdown_rows = []

    for metric in [
        "mean_mdd",
        "median_mdd",
        "worst_mdd",
        "best_mdd"
    ]:

        drawdown_rows.append({

            "Metric":
                metric,

            "Real":
                real_drawdown[metric],

            "Synthetic":
                synthetic_drawdown[metric],

            "Absolute_Error":
                abs(
                    real_drawdown[metric]
                    -
                    synthetic_drawdown[metric]
                )
        })

        print(
            f"{metric:<15} "
            f"Real={real_drawdown[metric]: .8f} | "
            f"Synthetic={synthetic_drawdown[metric]: .8f}"
        )

    drawdown_df = pd.DataFrame(
        drawdown_rows
    )

    drawdown_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "drawdown_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # 12. OVERALL SUMMARY
    # ========================================================

    print("\n12. OVERALL SUMMARY")
    print("-" * 80)

    average_mean_error = float(
        mean_std_df[
            "Absolute_Mean_Error"
        ].mean()
    )

    average_std_error = float(
        mean_std_df[
            "Absolute_Std_Error"
        ].mean()
    )

    average_volatility_error = float(
        volatility_df[
            "Absolute_Difference"
        ].mean()
    )

    average_ks = float(
        ks_df[
            "KS_Statistic"
        ].mean()
    )

    average_quantile_error = float(
        quantile_df[
            "Absolute_Error"
        ].mean()
    )

    summary = {

        "model":
            "MacroStress-GAN V3.4",

        "validation_type":
            "Complete Statistical Validation",

        "real_shape":
            list(real.shape),

        "synthetic_shape":
            list(synthetic.shape),

        "features":
            FEATURES,

        "mean_absolute_error":
            average_mean_error,

        "std_absolute_error":
            average_std_error,

        "mean_absolute_correlation_error":
            correlation_mae,

        "average_lag1_error":
            avg_lag_error,

        "average_30d_volatility_error":
            average_volatility_error,

        "average_ks_statistic":
            average_ks,

        "average_quantile_absolute_error":
            average_quantile_error,

        "nifty_var95_real":
            real_var95,

        "nifty_var95_synthetic":
            synthetic_var95,

        "nifty_es95_real":
            real_es95,

        "nifty_es95_synthetic":
            synthetic_es95,

        "nifty_mean_mdd_real":
            real_drawdown["mean_mdd"],

        "nifty_mean_mdd_synthetic":
            synthetic_drawdown["mean_mdd"],

        "nifty_median_mdd_real":
            real_drawdown["median_mdd"],

        "nifty_median_mdd_synthetic":
            synthetic_drawdown["median_mdd"],

        "nifty_worst_mdd_real":
            real_drawdown["worst_mdd"],

        "nifty_worst_mdd_synthetic":
            synthetic_drawdown["worst_mdd"]
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "validation_summary_v34.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    summary_table = pd.DataFrame([{

        "Metric":
            "Mean Absolute Error",

        "Value":
            average_mean_error
    }, {

        "Metric":
            "Std Absolute Error",

        "Value":
            average_std_error
    }, {

        "Metric":
            "Mean Absolute Correlation Error",

        "Value":
            correlation_mae
    }, {

        "Metric":
            "Average Lag-1 Error",

        "Value":
            avg_lag_error
    }, {

        "Metric":
            "Average 30D Volatility Error",

        "Value":
            average_volatility_error
    }, {

        "Metric":
            "Average KS Statistic",

        "Value":
            average_ks
    }, {

        "Metric":
            "Average Quantile Error",

        "Value":
            average_quantile_error
    }, {

        "Metric":
            "VaR95 Real",

        "Value":
            real_var95
    }, {

        "Metric":
            "VaR95 Synthetic",

        "Value":
            synthetic_var95
    }, {

        "Metric":
            "ES95 Real",

        "Value":
            real_es95
    }, {

        "Metric":
            "ES95 Synthetic",

        "Value":
            synthetic_es95
    }, {

        "Metric":
            "Mean MDD Real",

        "Value":
            real_drawdown["mean_mdd"]
    }, {

        "Metric":
            "Mean MDD Synthetic",

        "Value":
            synthetic_drawdown["mean_mdd"]
    }, {

        "Metric":
            "Median MDD Real",

        "Value":
            real_drawdown["median_mdd"]
    }, {

        "Metric":
            "Median MDD Synthetic",

        "Value":
            synthetic_drawdown["median_mdd"]
    }, {

        "Metric":
            "Worst MDD Real",

        "Value":
            real_drawdown["worst_mdd"]
    }, {

        "Metric":
            "Worst MDD Synthetic",

        "Value":
            synthetic_drawdown["worst_mdd"]
    }])

    summary_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_summary_v34.csv"
        ),
        index=False
    )

    print(
        f"Mean absolute correlation error: "
        f"{correlation_mae:.6f}"
    )

    print(
        f"Average lag-1 error: "
        f"{avg_lag_error:.6f}"
    )

    print(
        f"Average KS statistic: "
        f"{average_ks:.6f}"
    )

    print(
        f"VaR95: "
        f"Real={real_var95:.6f}, "
        f"Synthetic={synthetic_var95:.6f}"
    )

    print(
        f"ES95: "
        f"Real={real_es95:.6f}, "
        f"Synthetic={synthetic_es95:.6f}"
    )

    print(
        f"Mean MDD: "
        f"Real={real_drawdown['mean_mdd']:.6f}, "
        f"Synthetic={synthetic_drawdown['mean_mdd']:.6f}"
    )

    print(
        f"Worst MDD: "
        f"Real={real_drawdown['worst_mdd']:.6f}, "
        f"Synthetic={synthetic_drawdown['worst_mdd']:.6f}"
    )

    print(
        f"\nSummary saved:\n"
        f"{summary_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("V3.4 COMPLETE STATISTICAL VALIDATION FINISHED")
    print("=" * 80)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f"  - {filename}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()