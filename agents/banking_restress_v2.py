from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# MACROSTRESS-GAN
# BANKING INDEPENDENT RE-STRESS V2
# =============================================================================
#
# Purpose:
#   Independently verify the optimized banking sensitivity weights.
#
# Improvements over V1:
#
#   1. Expected Loss remains based on adverse-only stress.
#   2. VaR and ES remain based on adverse-only stress.
#   3. Maximum Drawdown is now calculated from a SIGNED path.
#   4. Favorable monthly movements can recover portfolio value.
#   5. Maximum Drawdown therefore becomes path-dependent.
#
# Important:
#   These are MODELING SENSITIVITY WEIGHTS, not actual bank
#   balance-sheet exposures.
#
# TimeGAN V4:
#   Event-aware TimeGAN-style hybrid.
#   It is NOT claimed to be canonical TimeGAN.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "banking"
)

SYNTHETIC_PATH = (
    OUTPUT_DIR
    / "banking_timegan_v4_synthetic.csv"
)

OPTIMIZATION_JSON = (
    OUTPUT_DIR
    / "banking_optimization_v1_results.json"
)

RESULT_JSON = (
    OUTPUT_DIR
    / "banking_restress_v2_results.json"
)

SCENARIO_CSV = (
    OUTPUT_DIR
    / "banking_restress_v2_scenarios.csv"
)

MONTHLY_CSV = (
    OUTPUT_DIR
    / "banking_restress_v2_monthly_paths.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "banking_restress_v2_summary.csv"
)

VALIDATION_JSON = (
    OUTPUT_DIR
    / "banking_restress_v2_validation.json"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Banking_Independent_ReStress_V2"

TIMEGAN_VERSION = "TimeGAN_Banking_V4"

PORTFOLIO_VALUE = 100_000_000.0

VAR_CONFIDENCE = 0.95

LOSS_SCALE = 0.01

MAX_MONTHLY_LOSS_RATE = 0.05

SEQ_LEN = 30

EXPECTED_SCENARIOS = 1000


FEATURES = [
    "BANK_NIFTY_Return",
    "REPO_RATE_Change",
    "CPI_INFLATION_Change",
    "CREDIT_GROWTH_Change",
    "USD_INR_Return",
]


ADVERSE_DIRECTION = {
    "BANK_NIFTY_Return": -1.0,
    "REPO_RATE_Change": 1.0,
    "CPI_INFLATION_Change": 1.0,
    "CREDIT_GROWTH_Change": -1.0,
    "USD_INR_Return": 1.0,
}


# =============================================================================
# JSON SAFETY
# =============================================================================

def make_json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return make_json_safe(
            obj.tolist()
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):

        value = float(obj)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, (float, int, str, bool)):
        return obj

    if obj is None:
        return None

    return str(obj)


# =============================================================================
# LOAD SYNTHETIC DATA
# =============================================================================

def load_synthetic_data():

    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(
            f"TimeGAN V4 synthetic dataset not found:\n"
            f"{SYNTHETIC_PATH}"
        )

    df = pd.read_csv(
        SYNTHETIC_PATH
    )

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    if df[FEATURES].isna().any().any():

        raise ValueError(
            "Synthetic data contains NaN."
        )

    if np.isinf(
        df[FEATURES].to_numpy()
    ).any():

        raise ValueError(
            "Synthetic data contains infinity."
        )

    if len(df) % SEQ_LEN != 0:

        raise ValueError(
            "Synthetic rows are not divisible "
            f"by sequence length {SEQ_LEN}."
        )

    return df.reset_index(drop=True)


# =============================================================================
# LOAD OPTIMIZATION RESULT
# =============================================================================

def load_optimization_result():

    if not OPTIMIZATION_JSON.exists():

        raise FileNotFoundError(
            f"Optimization result not found:\n"
            f"{OPTIMIZATION_JSON}"
        )

    with open(
        OPTIMIZATION_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        result = json.load(f)

    baseline_weights = result[
        "baseline_weights"
    ]

    optimized_weights = result[
        "optimized_weights"
    ]

    return (
        baseline_weights,
        optimized_weights,
    )


# =============================================================================
# REAL-DATA CALIBRATION
# =============================================================================

def load_real_calibration():

    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "institutions"
        / "banking"
        / "banking_monthly_dataset_long.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Real banking dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df = df.dropna(
        subset=FEATURES
    )

    calibration = {}

    for feature in FEATURES:

        values = df[
            feature
        ].to_numpy(
            dtype=float
        )

        std = float(
            np.std(
                values,
                ddof=1
            )
        )

        if std <= 0:

            raise ValueError(
                f"Invalid standard deviation "
                f"for {feature}"
            )

        calibration[feature] = {
            "mean": float(
                np.mean(values)
            ),
            "std": std,
        }

    return calibration


# =============================================================================
# CALCULATE SIGNED AND ADVERSE SHOCKS
# =============================================================================

def calculate_shocks(
    scenario_df,
    calibration,
):

    signed = pd.DataFrame(
        index=scenario_df.index
    )

    adverse = pd.DataFrame(
        index=scenario_df.index
    )

    for feature in FEATURES:

        mean = calibration[
            feature
        ]["mean"]

        std = calibration[
            feature
        ]["std"]

        direction = ADVERSE_DIRECTION[
            feature
        ]

        z = (
            scenario_df[feature].to_numpy(
                dtype=float
            )
            - mean
        ) / std

        # ---------------------------------------------------------------------
        # Signed adverse-direction shock
        #
        # Positive value = adverse
        # Negative value = favorable
        # ---------------------------------------------------------------------

        signed_shock = (
            direction * z
        )

        signed[feature] = (
            signed_shock
        )

        # ---------------------------------------------------------------------
        # Adverse-only shock
        # ---------------------------------------------------------------------

        adverse[feature] = np.maximum(
            signed_shock,
            0.0,
        )

    return signed, adverse


# =============================================================================
# CALCULATE PORTFOLIO PATH
# =============================================================================

def calculate_scenario_metrics(
    scenario_df,
    calibration,
    weights,
):

    signed_z, adverse_z = (
        calculate_shocks(
            scenario_df,
            calibration,
        )
    )

    weight_vector = np.array(
        [
            weights[feature]
            for feature in FEATURES
        ],
        dtype=float,
    )

    signed_matrix = (
        signed_z[FEATURES]
        .to_numpy(dtype=float)
    )

    adverse_matrix = (
        adverse_z[FEATURES]
        .to_numpy(dtype=float)
    )

    # =========================================================================
    # ADVERSE STRESS
    # =========================================================================

    adverse_score = (
        adverse_matrix
        @ weight_vector
    )

    adverse_monthly_loss = np.clip(
        adverse_score * LOSS_SCALE,
        0.0,
        MAX_MONTHLY_LOSS_RATE,
    )

    # =========================================================================
    # SIGNED STRESS
    # =========================================================================
    #
    # This is deliberately NOT clipped at zero.
    #
    # Positive signed stress:
    #     loss
    #
    # Negative signed stress:
    #     favorable movement / recovery
    #
    # =========================================================================

    signed_score = (
        signed_matrix
        @ weight_vector
    )

    signed_monthly_return = (
        -signed_score
        * LOSS_SCALE
    )

    # Prevent a single monthly movement from producing
    # an impossible total return.
    signed_monthly_return = np.clip(
        signed_monthly_return,
        -MAX_MONTHLY_LOSS_RATE,
        MAX_MONTHLY_LOSS_RATE,
    )

    n_rows = len(
        scenario_df
    )

    n_scenarios = (
        n_rows // SEQ_LEN
    )

    scenario_rows = []

    monthly_rows = []

    for scenario_id in range(
        n_scenarios
    ):

        start = (
            scenario_id
            * SEQ_LEN
        )

        end = (
            start
            + SEQ_LEN
        )

        adverse_losses = (
            adverse_monthly_loss[
                start:end
            ]
        )

        signed_returns = (
            signed_monthly_return[
                start:end
            ]
        )

        # =====================================================================
        # ADVERSE-ONLY PORTFOLIO
        # =====================================================================

        adverse_values = np.empty(
            SEQ_LEN + 1,
            dtype=float,
        )

        adverse_values[0] = (
            PORTFOLIO_VALUE
        )

        for i, loss_rate in enumerate(
            adverse_losses,
            start=1,
        ):

            adverse_values[i] = (
                adverse_values[i - 1]
                * (1.0 - loss_rate)
            )

        adverse_final = (
            adverse_values[-1]
        )

        total_loss = (
            PORTFOLIO_VALUE
            - adverse_final
        )

        # =====================================================================
        # SIGNED MARK-TO-MARKET PATH
        # =====================================================================

        signed_values = np.empty(
            SEQ_LEN + 1,
            dtype=float,
        )

        signed_values[0] = (
            PORTFOLIO_VALUE
        )

        for i, monthly_return in enumerate(
            signed_returns,
            start=1,
        ):

            signed_values[i] = (
                signed_values[i - 1]
                * (1.0 + monthly_return)
            )

        # =====================================================================
        # PATH-DEPENDENT DRAWDOWN
        # =====================================================================

        running_max = np.maximum.accumulate(
            signed_values
        )

        drawdown = (
            running_max
            - signed_values
        ) / running_max

        max_drawdown = float(
            np.max(drawdown)
        )

        max_drawdown_month = int(
            np.argmax(drawdown)
        )

        signed_final_return = (
            signed_values[-1]
            / PORTFOLIO_VALUE
        ) - 1.0

        worst_monthly_return = float(
            np.min(signed_returns)
        )

        best_monthly_return = float(
            np.max(signed_returns)
        )

        scenario_rows.append(
            {
                "scenario_id": scenario_id,

                "adverse_loss_rate": float(
                    total_loss
                    / PORTFOLIO_VALUE
                ),

                "adverse_loss_inr": float(
                    total_loss
                ),

                "signed_final_return": float(
                    signed_final_return
                ),

                "signed_final_value": float(
                    signed_values[-1]
                ),

                "maximum_drawdown": (
                    max_drawdown
                ),

                "maximum_drawdown_inr": (
                    max_drawdown
                    * PORTFOLIO_VALUE
                ),

                "maximum_drawdown_month": (
                    max_drawdown_month
                ),

                "worst_monthly_return": (
                    worst_monthly_return
                ),

                "best_monthly_return": (
                    best_monthly_return
                ),
            }
        )

        # =====================================================================
        # SAVE MONTHLY PATH
        # =====================================================================

        for month in range(
            SEQ_LEN
        ):

            monthly_rows.append(
                {
                    "scenario_id":
                        scenario_id,

                    "month":
                        month + 1,

                    "adverse_monthly_loss_rate":
                        float(
                            adverse_losses[month]
                        ),

                    "signed_monthly_return":
                        float(
                            signed_returns[month]
                        ),

                    "adverse_portfolio_value":
                        float(
                            adverse_values[
                                month + 1
                            ]
                        ),

                    "signed_portfolio_value":
                        float(
                            signed_values[
                                month + 1
                            ]
                        ),

                    "running_peak":
                        float(
                            running_max[
                                month + 1
                            ]
                        ),

                    "drawdown":
                        float(
                            drawdown[
                                month + 1
                            ]
                        ),
                }
            )

    scenario_df_out = pd.DataFrame(
        scenario_rows
    )

    monthly_df_out = pd.DataFrame(
        monthly_rows
    )

    return (
        scenario_df_out,
        monthly_df_out,
    )


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_metrics(
    scenario_results,
):

    losses = (
        scenario_results[
            "adverse_loss_rate"
        ]
        .to_numpy(
            dtype=float
        )
    )

    drawdowns = (
        scenario_results[
            "maximum_drawdown"
        ]
        .to_numpy(
            dtype=float
        )
    )

    var95 = float(
        np.quantile(
            losses,
            VAR_CONFIDENCE,
        )
    )

    tail = losses[
        losses >= var95
    ]

    if len(tail) == 0:
        tail = np.array(
            [var95]
        )

    es95 = float(
        np.mean(tail)
    )

    expected_loss = float(
        np.mean(losses)
    )

    max_drawdown = float(
        np.max(drawdowns)
    )

    worst_loss = float(
        np.max(losses)
    )

    worst_scenario = int(
        np.argmax(losses)
    )

    worst_drawdown_scenario = int(
        np.argmax(drawdowns)
    )

    return {

        "Scenario_Count":
            int(len(losses)),

        "VaR_95":
            var95,

        "VaR_95_INR":
            var95
            * PORTFOLIO_VALUE,

        "ES_95":
            es95,

        "ES_95_INR":
            es95
            * PORTFOLIO_VALUE,

        "Expected_Loss":
            expected_loss,

        "Expected_Loss_INR":
            expected_loss
            * PORTFOLIO_VALUE,

        "Maximum_Drawdown":
            max_drawdown,

        "Maximum_Drawdown_INR":
            max_drawdown
            * PORTFOLIO_VALUE,

        "Worst_Scenario_Loss":
            worst_loss,

        "Worst_Scenario_Loss_INR":
            worst_loss
            * PORTFOLIO_VALUE,

        "Worst_Scenario_ID":
            worst_scenario,

        "Worst_Drawdown_Scenario_ID":
            worst_drawdown_scenario,

        "Mean_Path_Final_Return":
            float(
                scenario_results[
                    "signed_final_return"
                ].mean()
            ),

        "Worst_Path_Final_Return":
            float(
                scenario_results[
                    "signed_final_return"
                ].min()
            ),
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate(
    baseline,
    optimized,
    baseline_weights,
    optimized_weights,
    scenario_count,
):

    checks = {}

    checks[
        "scenario_count"
    ] = (
        scenario_count
        == EXPECTED_SCENARIOS
    )

    checks[
        "baseline_weight_sum"
    ] = (
        abs(
            sum(
                baseline_weights.values()
            )
            - 1.0
        )
        < 1e-8
    )

    checks[
        "optimized_weight_sum"
    ] = (
        abs(
            sum(
                optimized_weights.values()
            )
            - 1.0
        )
        < 1e-8
    )

    checks[
        "optimized_weights_finite"
    ] = all(
        np.isfinite(
            value
        )
        for value in
        optimized_weights.values()
    )

    checks[
        "baseline_es_ge_var"
    ] = (
        baseline["ES_95"]
        >= baseline["VaR_95"]
    )

    checks[
        "optimized_es_ge_var"
    ] = (
        optimized["ES_95"]
        >= optimized["VaR_95"]
    )

    checks[
        "baseline_worst_ge_es"
    ] = (
        baseline[
            "Worst_Scenario_Loss"
        ]
        >= baseline["ES_95"]
    )

    checks[
        "optimized_worst_ge_es"
    ] = (
        optimized[
            "Worst_Scenario_Loss"
        ]
        >= optimized["ES_95"]
    )

    checks[
        "baseline_drawdown_nonnegative"
    ] = (
        baseline[
            "Maximum_Drawdown"
        ]
        >= 0
    )

    checks[
        "optimized_drawdown_nonnegative"
    ] = (
        optimized[
            "Maximum_Drawdown"
        ]
        >= 0
    )

    # Critical improvement:
    #
    # Drawdown should NOT simply be identical to worst cumulative loss
    # for this signed path formulation.
    #
    checks[
        "baseline_drawdown_independent"
    ] = (
        abs(
            baseline[
                "Maximum_Drawdown"
            ]
            - baseline[
                "Worst_Scenario_Loss"
            ]
        )
        > 1e-10
    )

    checks[
        "optimized_drawdown_independent"
    ] = (
        abs(
            optimized[
                "Maximum_Drawdown"
            ]
            - optimized[
                "Worst_Scenario_Loss"
            ]
        )
        > 1e-10
    )

    checks[
        "monetary_consistency"
    ] = (
        abs(
            optimized[
                "Expected_Loss_INR"
            ]
            -
            optimized[
                "Expected_Loss"
            ]
            * PORTFOLIO_VALUE
        )
        < 1e-6
    )

    checks[
        "no_nan_metrics"
    ] = all(
        np.isfinite(
            float(value)
        )
        for key, value
        in optimized.items()
        if isinstance(
            value,
            (int, float)
        )
    )

    checks[
        "all_checks_pass"
    ] = all(
        bool(value)
        for value in checks.values()
    )

    return checks


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "MACROSTRESS-GAN"
    )
    print(
        "BANKING INDEPENDENT RE-STRESS V2"
    )
    print("=" * 80)

    print(
        f"Project Root : {PROJECT_ROOT}"
    )

    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"TimeGAN      : {TIMEGAN_VERSION}"
    )

    print(
        f"Portfolio    : ₹{PORTFOLIO_VALUE:,.2f}"
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING BANKING TIMEGAN V4")
    print("=" * 80)

    scenario_df = (
        load_synthetic_data()
    )

    scenario_count = (
        len(scenario_df)
        // SEQ_LEN
    )

    print(
        f"Rows       : {len(scenario_df):,}"
    )

    print(
        f"Scenarios  : {scenario_count:,}"
    )

    print(
        f"Sequence   : {SEQ_LEN}"
    )

    # -------------------------------------------------------------------------
    # LOAD WEIGHTS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING OPTIMIZATION V1 WEIGHTS")
    print("=" * 80)

    (
        baseline_weights,
        optimized_weights,
    ) = load_optimization_result()

    print()
    print("Baseline weights:")

    for feature in FEATURES:

        print(
            f"{feature:<28}"
            f"{baseline_weights[feature]:.6f}"
        )

    print()
    print("Optimized weights:")

    for feature in FEATURES:

        print(
            f"{feature:<28}"
            f"{optimized_weights[feature]:.6f}"
        )

    # -------------------------------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING REAL-DATA CALIBRATION")
    print("=" * 80)

    calibration = (
        load_real_calibration()
    )

    for feature in FEATURES:

        print(
            f"{feature:<28}"
            f"mean="
            f"{calibration[feature]['mean']: .6f} "
            f"std="
            f"{calibration[feature]['std']: .6f}"
        )

    # -------------------------------------------------------------------------
    # BASELINE RESTRESS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("INDEPENDENT BASELINE RE-STRESS")
    print("=" * 80)

    (
        baseline_scenarios,
        baseline_monthly,
    ) = calculate_scenario_metrics(
        scenario_df,
        calibration,
        baseline_weights,
    )

    baseline_metrics = (
        calculate_metrics(
            baseline_scenarios
        )
    )

    print(
        f"VaR 95%            : "
        f"₹{baseline_metrics['VaR_95_INR']:,.2f}"
    )

    print(
        f"ES 95%             : "
        f"₹{baseline_metrics['ES_95_INR']:,.2f}"
    )

    print(
        f"Expected Loss      : "
        f"₹{baseline_metrics['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Max Drawdown       : "
        f"₹{baseline_metrics['Maximum_Drawdown_INR']:,.2f}"
    )

    print(
        f"Worst Scenario     : "
        f"₹{baseline_metrics['Worst_Scenario_Loss_INR']:,.2f}"
    )

    print(
        f"Mean Final Return  : "
        f"{baseline_metrics['Mean_Path_Final_Return']:.6f}"
    )

    print(
        f"Worst Final Return : "
        f"{baseline_metrics['Worst_Path_Final_Return']:.6f}"
    )

    # -------------------------------------------------------------------------
    # OPTIMIZED RESTRESS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("INDEPENDENT OPTIMIZED RE-STRESS")
    print("=" * 80)

    (
        optimized_scenarios,
        optimized_monthly,
    ) = calculate_scenario_metrics(
        scenario_df,
        calibration,
        optimized_weights,
    )

    optimized_metrics = (
        calculate_metrics(
            optimized_scenarios
        )
    )

    print(
        f"VaR 95%            : "
        f"₹{optimized_metrics['VaR_95_INR']:,.2f}"
    )

    print(
        f"ES 95%             : "
        f"₹{optimized_metrics['ES_95_INR']:,.2f}"
    )

    print(
        f"Expected Loss      : "
        f"₹{optimized_metrics['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Max Drawdown       : "
        f"₹{optimized_metrics['Maximum_Drawdown_INR']:,.2f}"
    )

    print(
        f"Worst Scenario     : "
        f"₹{optimized_metrics['Worst_Scenario_Loss_INR']:,.2f}"
    )

    print(
        f"Mean Final Return  : "
        f"{optimized_metrics['Mean_Path_Final_Return']:.6f}"
    )

    print(
        f"Worst Final Return : "
        f"{optimized_metrics['Worst_Path_Final_Return']:.6f}"
    )

    # -------------------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------------------

    expected_loss_reduction = (
        baseline_metrics[
            "Expected_Loss_INR"
        ]
        -
        optimized_metrics[
            "Expected_Loss_INR"
        ]
    )

    expected_loss_reduction_pct = (
        expected_loss_reduction
        /
        baseline_metrics[
            "Expected_Loss_INR"
        ]
    )

    var_reduction = (
        baseline_metrics[
            "VaR_95_INR"
        ]
        -
        optimized_metrics[
            "VaR_95_INR"
        ]
    )

    es_reduction = (
        baseline_metrics[
            "ES_95_INR"
        ]
        -
        optimized_metrics[
            "ES_95_INR"
        ]
    )

    drawdown_reduction = (
        baseline_metrics[
            "Maximum_Drawdown_INR"
        ]
        -
        optimized_metrics[
            "Maximum_Drawdown_INR"
        ]
    )

    worst_loss_reduction = (
        baseline_metrics[
            "Worst_Scenario_Loss_INR"
        ]
        -
        optimized_metrics[
            "Worst_Scenario_Loss_INR"
        ]
    )

    print()
    print("=" * 80)
    print("BASELINE VS OPTIMIZED RE-STRESS")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'Baseline':>18}"
        f"{'Optimized':>18}"
    )

    print("-" * 68)

    print(
        f"{'VaR 95%':<30}"
        f"₹{baseline_metrics['VaR_95_INR']:>15,.2f}"
        f"₹{optimized_metrics['VaR_95_INR']:>15,.2f}"
    )

    print(
        f"{'ES 95%':<30}"
        f"₹{baseline_metrics['ES_95_INR']:>15,.2f}"
        f"₹{optimized_metrics['ES_95_INR']:>15,.2f}"
    )

    print(
        f"{'Expected Loss':<30}"
        f"₹{baseline_metrics['Expected_Loss_INR']:>15,.2f}"
        f"₹{optimized_metrics['Expected_Loss_INR']:>15,.2f}"
    )

    print(
        f"{'Maximum Drawdown':<30}"
        f"₹{baseline_metrics['Maximum_Drawdown_INR']:>15,.2f}"
        f"₹{optimized_metrics['Maximum_Drawdown_INR']:>15,.2f}"
    )

    print(
        f"{'Worst Scenario Loss':<30}"
        f"₹{baseline_metrics['Worst_Scenario_Loss_INR']:>15,.2f}"
        f"₹{optimized_metrics['Worst_Scenario_Loss_INR']:>15,.2f}"
    )

    print()
    print(
        f"Expected Loss Reduction : "
        f"₹{expected_loss_reduction:,.2f}"
    )

    print(
        f"Expected Loss Reduction : "
        f"{expected_loss_reduction_pct * 100:.4f}%"
    )

    print(
        f"VaR Reduction           : "
        f"₹{var_reduction:,.2f}"
    )

    print(
        f"ES Reduction            : "
        f"₹{es_reduction:,.2f}"
    )

    print(
        f"Drawdown Reduction      : "
        f"₹{drawdown_reduction:,.2f}"
    )

    print(
        f"Worst Loss Reduction    : "
        f"₹{worst_loss_reduction:,.2f}"
    )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RE-STRESS V2 VALIDATION")
    print("=" * 80)

    checks = validate(
        baseline_metrics,
        optimized_metrics,
        baseline_weights,
        optimized_weights,
        scenario_count,
    )

    for name, value in checks.items():

        print(
            f"{name:<40}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    status = (
        "PASS"
        if checks["all_checks_pass"]
        else "REVIEW REQUIRED"
    )

    print()
    print(
        f"BANKING INDEPENDENT RE-STRESS V2 : "
        f"{status}"
    )

    # -------------------------------------------------------------------------
    # SAVE SCENARIOS
    # -------------------------------------------------------------------------

    scenario_output = optimized_scenarios.copy()

    scenario_output[
        "baseline_expected_loss_rate"
    ] = baseline_scenarios[
        "adverse_loss_rate"
    ]

    scenario_output[
        "baseline_expected_loss_inr"
    ] = baseline_scenarios[
        "adverse_loss_inr"
    ]

    scenario_output[
        "baseline_max_drawdown"
    ] = baseline_scenarios[
        "maximum_drawdown"
    ]

    scenario_output[
        "optimized_expected_loss_rate"
    ] = optimized_scenarios[
        "adverse_loss_rate"
    ]

    scenario_output[
        "optimized_expected_loss_inr"
    ] = optimized_scenarios[
        "adverse_loss_inr"
    ]

    scenario_output[
        "optimized_max_drawdown"
    ] = optimized_scenarios[
        "maximum_drawdown"
    ]

    scenario_output.to_csv(
        SCENARIO_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE MONTHLY
    # -------------------------------------------------------------------------

    monthly_output = optimized_monthly.copy()

    monthly_output[
        "baseline_signed_monthly_return"
    ] = baseline_monthly[
        "signed_monthly_return"
    ]

    monthly_output[
        "baseline_signed_portfolio_value"
    ] = baseline_monthly[
        "signed_portfolio_value"
    ]

    monthly_output[
        "baseline_drawdown"
    ] = baseline_monthly[
        "drawdown"
    ]

    monthly_output[
        "optimized_signed_monthly_return"
    ] = optimized_monthly[
        "signed_monthly_return"
    ]

    monthly_output[
        "optimized_signed_portfolio_value"
    ] = optimized_monthly[
        "signed_portfolio_value"
    ]

    monthly_output[
        "optimized_drawdown"
    ] = optimized_monthly[
        "drawdown"
    ]

    monthly_output.to_csv(
        MONTHLY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "metric": "VaR_95",
                "baseline": baseline_metrics[
                    "VaR_95"
                ],
                "optimized": optimized_metrics[
                    "VaR_95"
                ],
                "baseline_inr": baseline_metrics[
                    "VaR_95_INR"
                ],
                "optimized_inr": optimized_metrics[
                    "VaR_95_INR"
                ],
            },
            {
                "metric": "ES_95",
                "baseline": baseline_metrics[
                    "ES_95"
                ],
                "optimized": optimized_metrics[
                    "ES_95"
                ],
                "baseline_inr": baseline_metrics[
                    "ES_95_INR"
                ],
                "optimized_inr": optimized_metrics[
                    "ES_95_INR"
                ],
            },
            {
                "metric": "Expected_Loss",
                "baseline": baseline_metrics[
                    "Expected_Loss"
                ],
                "optimized": optimized_metrics[
                    "Expected_Loss"
                ],
                "baseline_inr": baseline_metrics[
                    "Expected_Loss_INR"
                ],
                "optimized_inr": optimized_metrics[
                    "Expected_Loss_INR"
                ],
            },
            {
                "metric": "Maximum_Drawdown",
                "baseline": baseline_metrics[
                    "Maximum_Drawdown"
                ],
                "optimized": optimized_metrics[
                    "Maximum_Drawdown"
                ],
                "baseline_inr": baseline_metrics[
                    "Maximum_Drawdown_INR"
                ],
                "optimized_inr": optimized_metrics[
                    "Maximum_Drawdown_INR"
                ],
            },
            {
                "metric": "Worst_Scenario_Loss",
                "baseline": baseline_metrics[
                    "Worst_Scenario_Loss"
                ],
                "optimized": optimized_metrics[
                    "Worst_Scenario_Loss"
                ],
                "baseline_inr": baseline_metrics[
                    "Worst_Scenario_Loss_INR"
                ],
                "optimized_inr": optimized_metrics[
                    "Worst_Scenario_Loss_INR"
                ],
            },
        ]
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE JSON
    # -------------------------------------------------------------------------

    result = {

        "model":
            MODEL_NAME,

        "timegan_version":
            TIMEGAN_VERSION,

        "portfolio_value_inr":
            PORTFOLIO_VALUE,

        "scenario_count":
            scenario_count,

        "sequence_length":
            SEQ_LEN,

        "methodology": {

            "adverse_loss":
                "Adverse-only weighted standardized shocks "
                "with 1% scaling and 5% monthly loss cap.",

            "drawdown":
                "Path-dependent maximum drawdown calculated "
                "from signed monthly stress returns.",

            "signed_path":
                "Favorable movements are retained and can "
                "increase portfolio value.",

            "optimization":
                "Weights are imported from Banking Portfolio "
                "Optimizer V1.",

            "timegan_note":
                "Banking V4 is an event-aware "
                "TimeGAN-style hybrid, not canonical TimeGAN.",

            "exposure_note":
                "Weights represent explicit modeling "
                "sensitivity assumptions and are not claimed "
                "to be actual bank balance-sheet exposures.",
        },

        "baseline_weights":
            baseline_weights,

        "optimized_weights":
            optimized_weights,

        "baseline_metrics":
            baseline_metrics,

        "optimized_metrics":
            optimized_metrics,

        "comparison": {

            "expected_loss_reduction_inr":
                expected_loss_reduction,

            "expected_loss_reduction_pct":
                expected_loss_reduction_pct,

            "var_reduction_inr":
                var_reduction,

            "es_reduction_inr":
                es_reduction,

            "drawdown_reduction_inr":
                drawdown_reduction,

            "worst_loss_reduction_inr":
                worst_loss_reduction,
        },

        "validation":
            checks,

        "status":
            status,
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_safe(result),
            f,
            indent=2,
        )

    with open(
        VALIDATION_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_safe(checks),
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "BANKING INDEPENDENT RE-STRESS V2 COMPLETE"
    )
    print("=" * 80)

    print(
        f"Expected Loss Reduction : "
        f"₹{expected_loss_reduction:,.2f}"
    )

    print(
        f"Expected Loss Reduction : "
        f"{expected_loss_reduction_pct * 100:.4f}%"
    )

    print()

    print(
        f"Baseline Maximum Drawdown : "
        f"{baseline_metrics['Maximum_Drawdown']:.6f}"
    )

    print(
        f"Optimized Maximum Drawdown : "
        f"{optimized_metrics['Maximum_Drawdown']:.6f}"
    )

    print()

    print(
        f"Baseline Worst Loss : "
        f"{baseline_metrics['Worst_Scenario_Loss']:.6f}"
    )

    print(
        f"Optimized Worst Loss : "
        f"{optimized_metrics['Worst_Scenario_Loss']:.6f}"
    )

    print()

    print(
        f"STATUS : {status}"
    )

    print()
    print("OUTPUT FILES")

    print(
        f"Results JSON : {RESULT_JSON}"
    )

    print(
        f"Scenario CSV : {SCENARIO_CSV}"
    )

    print(
        f"Monthly CSV  : {MONTHLY_CSV}"
    )

    print(
        f"Summary CSV  : {SUMMARY_CSV}"
    )

    print(
        f"Validation   : {VALIDATION_JSON}"
    )


if __name__ == "__main__":
    main()