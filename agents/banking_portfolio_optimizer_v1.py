from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# MACROSTRESS-GAN
# BANKING PORTFOLIO OPTIMIZER V1
# =============================================================================
#
# Purpose:
#   Optimize banking risk-sensitivity weights using the already validated
#   Banking Risk Engine V1 scenario framework.
#
# Important:
#   These are MODELING SENSITIVITY WEIGHTS.
#   They are NOT claimed to be actual bank balance-sheet exposures.
#
# Pipeline:
#
#   TimeGAN V4 scenarios
#           ↓
#   Real-data calibration
#           ↓
#   Baseline risk
#           ↓
#   Constrained optimization
#           ↓
#   Optimized risk
#           ↓
#   Independent re-stress
#           ↓
#   Validation
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "banking"

RISK_JSON = OUTPUT_DIR / "banking_risk_v1_results.json"
RISK_SCENARIO_CSV = OUTPUT_DIR / "banking_risk_v1_scenario_losses.csv"
RISK_MONTHLY_CSV = OUTPUT_DIR / "banking_risk_v1_monthly_scenario_details.csv"

OPTIMIZED_SCENARIO_CSV = (
    OUTPUT_DIR / "banking_optimization_v1_scenario_losses.csv"
)

OPTIMIZED_MONTHLY_CSV = (
    OUTPUT_DIR / "banking_optimization_v1_monthly_details.csv"
)

WEIGHT_CSV = OUTPUT_DIR / "banking_optimization_v1_weights.csv"

SUMMARY_CSV = OUTPUT_DIR / "banking_optimization_v1_summary.csv"

RESULT_JSON = OUTPUT_DIR / "banking_optimization_v1_results.json"

RESTRESS_JSON = OUTPUT_DIR / "banking_optimization_v1_restress.json"

VALIDATION_JSON = OUTPUT_DIR / "banking_optimization_v1_validation.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Banking_Portfolio_Optimizer_V1"

TIMEGAN_VERSION = "TimeGAN_Banking_V4"

PORTFOLIO_VALUE = 100_000_000.0

VAR_CONFIDENCE = 0.95

LOSS_SCALE = 0.01

MAX_MONTHLY_LOSS_RATE = 0.05

N_SCENARIOS_EXPECTED = 1000

SEQ_LEN_EXPECTED = 30


FEATURES = [
    "BANK_NIFTY_Return",
    "REPO_RATE_Change",
    "CPI_INFLATION_Change",
    "CREDIT_GROWTH_Change",
    "USD_INR_Return",
]


# =============================================================================
# BASELINE MODELING WEIGHTS
# =============================================================================

BASELINE_WEIGHTS = {
    "BANK_NIFTY_Return": 0.30,
    "REPO_RATE_Change": 0.20,
    "CPI_INFLATION_Change": 0.15,
    "CREDIT_GROWTH_Change": 0.20,
    "USD_INR_Return": 0.15,
}


# =============================================================================
# ADVERSE DIRECTIONS
# =============================================================================
#
# +1 means higher value is adverse.
# -1 means lower value is adverse.
#
# BANK_NIFTY:
#     negative return = adverse
#
# REPO:
#     positive rate change = adverse
#
# CPI:
#     positive inflation change = adverse
#
# CREDIT:
#     negative credit-growth change = adverse
#
# USD/INR:
#     positive return = adverse
#
# =============================================================================

ADVERSE_DIRECTION = {
    "BANK_NIFTY_Return": -1.0,
    "REPO_RATE_Change": 1.0,
    "CPI_INFLATION_Change": 1.0,
    "CREDIT_GROWTH_Change": -1.0,
    "USD_INR_Return": 1.0,
}


# =============================================================================
# OPTIMIZATION BOUNDS
# =============================================================================
#
# These are MODELING CONSTRAINTS, not actual regulatory limits.
#
# Every variable must remain between 5% and 40%.
# All weights must sum to 100%.
#
# =============================================================================

LOWER_WEIGHT = 0.05
UPPER_WEIGHT = 0.40


# =============================================================================
# JSON SAFETY
# =============================================================================

def make_json_safe(obj):
    """
    Recursively convert NumPy/Pandas values into native Python values.
    """

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj

    return str(obj)


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_metric(name: str, value: float, monetary: bool = False):
    if monetary:
        print(f"{name:<30}: ₹{value:,.2f}")
    else:
        print(f"{name:<30}: {value:.6f}")


# =============================================================================
# LOAD REAL CALIBRATION DATA
# =============================================================================

def load_real_data() -> pd.DataFrame:

    real_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "institutions"
        / "banking"
        / "banking_monthly_dataset_long.csv"
    )

    if not real_path.exists():
        raise FileNotFoundError(
            f"Real banking dataset not found:\n{real_path}"
        )

    df = pd.read_csv(real_path)

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required real-data features: {missing}"
        )

    df = df[FEATURES].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            "Insufficient real observations for calibration."
        )

    return df


# =============================================================================
# LOAD TIMEGAN SCENARIOS
# =============================================================================

def load_scenarios() -> pd.DataFrame:

    synthetic_path = (
        OUTPUT_DIR
        / "banking_timegan_v4_synthetic.csv"
    )

    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Banking TimeGAN V4 synthetic dataset not found:\n"
            f"{synthetic_path}"
        )

    df = pd.read_csv(synthetic_path)

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required scenario features: {missing}"
        )

    df = df.copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Synthetic scenario dataset contains NaN values."
        )

    if np.isinf(df[FEATURES].to_numpy()).any():
        raise ValueError(
            "Synthetic scenario dataset contains infinite values."
        )

    if len(df) % SEQ_LEN_EXPECTED != 0:
        raise ValueError(
            "Synthetic rows are not divisible by the expected "
            f"sequence length of {SEQ_LEN_EXPECTED}."
        )

    return df.reset_index(drop=True)


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_real_data(
    real_df: pd.DataFrame,
) -> dict:

    calibration = {}

    for feature in FEATURES:

        values = real_df[feature].to_numpy(dtype=float)

        calibration[feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }

    return calibration


# =============================================================================
# PREPARE ADVERSE Z-SCORES
# =============================================================================

def calculate_adverse_z_scores(
    scenario_df: pd.DataFrame,
    calibration: dict,
) -> pd.DataFrame:

    result = pd.DataFrame(index=scenario_df.index)

    for feature in FEATURES:

        mean = calibration[feature]["mean"]
        std = calibration[feature]["std"]
        direction = ADVERSE_DIRECTION[feature]

        if std <= 0:
            raise ValueError(
                f"Non-positive standard deviation for {feature}."
            )

        z = (
            scenario_df[feature].to_numpy(dtype=float)
            - mean
        ) / std

        adverse_z = direction * z

        # Loss cannot become negative from a favorable movement.
        adverse_z = np.maximum(adverse_z, 0.0)

        result[feature] = adverse_z

    return result


# =============================================================================
# SCENARIO LOSS ENGINE
# =============================================================================

def calculate_losses(
    scenario_df: pd.DataFrame,
    calibration: dict,
    weights: dict,
):
    """
    Calculate scenario losses and monthly portfolio paths.

    The formulation follows the Banking Risk Engine V1 assumptions.

    For each month:

        weighted adverse stress
        ↓
        LOSS_SCALE
        ↓
        monthly loss capped at 5%
        ↓
        compound portfolio value
    """

    z_df = calculate_adverse_z_scores(
        scenario_df,
        calibration,
    )

    weight_vector = np.array(
        [weights[f] for f in FEATURES],
        dtype=float,
    )

    z_matrix = z_df[FEATURES].to_numpy(dtype=float)

    weighted_stress = z_matrix @ weight_vector

    monthly_loss_rate = np.clip(
        weighted_stress * LOSS_SCALE,
        0.0,
        MAX_MONTHLY_LOSS_RATE,
    )

    n_rows = len(scenario_df)

    n_scenarios = n_rows // SEQ_LEN_EXPECTED

    scenario_records = []

    monthly_records = []

    scenario_losses = []

    scenario_max_drawdowns = []

    scenario_worst_months = []

    for scenario_id in range(n_scenarios):

        start = scenario_id * SEQ_LEN_EXPECTED
        end = start + SEQ_LEN_EXPECTED

        monthly_rates = monthly_loss_rate[start:end]

        portfolio_values = np.empty(
            SEQ_LEN_EXPECTED + 1,
            dtype=float,
        )

        portfolio_values[0] = PORTFOLIO_VALUE

        for month_idx, loss_rate in enumerate(
            monthly_rates,
            start=1,
        ):
            portfolio_values[month_idx] = (
                portfolio_values[month_idx - 1]
                * (1.0 - loss_rate)
            )

        final_value = portfolio_values[-1]

        total_loss = (
            PORTFOLIO_VALUE
            - final_value
        )

        running_max = np.maximum.accumulate(
            portfolio_values
        )

        drawdowns = (
            running_max - portfolio_values
        ) / running_max

        max_drawdown = float(
            np.max(drawdowns)
        )

        worst_month = int(
            np.argmax(monthly_rates)
        ) + 1

        scenario_losses.append(
            float(total_loss / PORTFOLIO_VALUE)
        )

        scenario_max_drawdowns.append(
            max_drawdown
        )

        scenario_worst_months.append(
            worst_month
        )

        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "loss_rate": float(
                    total_loss / PORTFOLIO_VALUE
                ),
                "loss_inr": float(total_loss),
                "max_drawdown": max_drawdown,
                "worst_month": worst_month,
            }
        )

        for month_idx, loss_rate in enumerate(
            monthly_rates,
            start=1,
        ):

            monthly_records.append(
                {
                    "scenario_id": scenario_id,
                    "month": month_idx,
                    "monthly_loss_rate": float(loss_rate),
                    "portfolio_value": float(
                        portfolio_values[month_idx]
                    ),
                }
            )

    scenario_losses = np.asarray(
        scenario_losses,
        dtype=float,
    )

    scenario_max_drawdowns = np.asarray(
        scenario_max_drawdowns,
        dtype=float,
    )

    return (
        pd.DataFrame(scenario_records),
        pd.DataFrame(monthly_records),
        scenario_losses,
        scenario_max_drawdowns,
        z_df,
    )


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_risk_metrics(
    scenario_losses: np.ndarray,
    scenario_drawdowns: np.ndarray,
) -> dict:

    if len(scenario_losses) == 0:
        raise ValueError("No scenario losses available.")

    var_95 = float(
        np.quantile(
            scenario_losses,
            VAR_CONFIDENCE,
        )
    )

    tail = scenario_losses[
        scenario_losses >= var_95
    ]

    if len(tail) == 0:
        tail = np.array([var_95])

    es_95 = float(
        np.mean(tail)
    )

    expected_loss = float(
        np.mean(scenario_losses)
    )

    max_drawdown = float(
        np.max(scenario_drawdowns)
    )

    worst_loss = float(
        np.max(scenario_losses)
    )

    worst_scenario_id = int(
        np.argmax(scenario_losses)
    )

    return {
        "VaR_95": var_95,
        "VaR_95_INR": var_95 * PORTFOLIO_VALUE,
        "ES_95": es_95,
        "ES_95_INR": es_95 * PORTFOLIO_VALUE,
        "Expected_Loss": expected_loss,
        "Expected_Loss_INR": expected_loss * PORTFOLIO_VALUE,
        "Maximum_Drawdown": max_drawdown,
        "Maximum_Drawdown_INR": max_drawdown * PORTFOLIO_VALUE,
        "Worst_Scenario_Loss": worst_loss,
        "Worst_Scenario_Loss_INR": worst_loss * PORTFOLIO_VALUE,
        "Worst_Scenario_ID": worst_scenario_id,
        "Scenario_Count": int(len(scenario_losses)),
    }


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def weights_from_vector(x: np.ndarray) -> dict:

    return {
        feature: float(x[i])
        for i, feature in enumerate(FEATURES)
    }


def optimization_objective(
    x: np.ndarray,
    scenario_df: pd.DataFrame,
    calibration: dict,
) -> float:

    weights = weights_from_vector(x)

    (
        _scenario_df,
        _monthly_df,
        scenario_losses,
        scenario_drawdowns,
        _z_df,
    ) = calculate_losses(
        scenario_df,
        calibration,
        weights,
    )

    metrics = calculate_risk_metrics(
        scenario_losses,
        scenario_drawdowns,
    )

    # Multi-objective risk criterion.
    #
    # Main target:
    #     Expected Loss
    #
    # Secondary:
    #     VaR
    #
    # Tail:
    #     ES
    #
    objective = (
        0.50 * metrics["Expected_Loss"]
        + 0.30 * metrics["VaR_95"]
        + 0.20 * metrics["ES_95"]
    )

    return float(objective)


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_weights(
    scenario_df: pd.DataFrame,
    calibration: dict,
) -> dict:

    x0 = np.array(
        [
            BASELINE_WEIGHTS[f]
            for f in FEATURES
        ],
        dtype=float,
    )

    bounds = [
        (LOWER_WEIGHT, UPPER_WEIGHT)
        for _ in FEATURES
    ]

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1.0,
        }
    ]

    result = minimize(
        optimization_objective,
        x0,
        args=(
            scenario_df,
            calibration,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 200,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Portfolio optimization failed:\n"
            + str(result.message)
        )

    optimized_weights = weights_from_vector(
        result.x
    )

    return {
        "optimizer_success": True,
        "optimizer_message": str(
            result.message
        ),
        "iterations": int(
            getattr(result, "nit", 0)
        ),
        "objective_value": float(
            result.fun
        ),
        "weights": optimized_weights,
    }


# =============================================================================
# VARIABLE CONTRIBUTIONS
# =============================================================================

def calculate_variable_contributions(
    z_df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:

    contribution_rows = []

    for feature in FEATURES:

        weighted = (
            z_df[feature].to_numpy()
            * weights[feature]
        )

        positive_weighted = np.maximum(
            weighted,
            0.0,
        )

        total = float(
            np.sum(positive_weighted)
        )

        contribution_rows.append(
            {
                "feature": feature,
                "weight": float(weights[feature]),
                "mean_adverse_z": float(
                    np.mean(z_df[feature])
                ),
                "weighted_mean_contribution": float(
                    np.mean(positive_weighted)
                ),
                "total_contribution": total,
            }
        )

    result = pd.DataFrame(
        contribution_rows
    )

    contribution_sum = result[
        "total_contribution"
    ].sum()

    if contribution_sum > 0:

        result["contribution_pct"] = (
            result["total_contribution"]
            / contribution_sum
        )

    else:

        result["contribution_pct"] = 0.0

    return result


# =============================================================================
# RESTRESS
# =============================================================================

def independent_restress(
    scenario_df: pd.DataFrame,
    calibration: dict,
    optimized_weights: dict,
) -> dict:

    (
        optimized_scenarios,
        optimized_monthly,
        optimized_losses,
        optimized_drawdowns,
        optimized_z,
    ) = calculate_losses(
        scenario_df,
        calibration,
        optimized_weights,
    )

    metrics = calculate_risk_metrics(
        optimized_losses,
        optimized_drawdowns,
    )

    baseline_weights = BASELINE_WEIGHTS.copy()

    (
        baseline_scenarios,
        baseline_monthly,
        baseline_losses,
        baseline_drawdowns,
        baseline_z,
    ) = calculate_losses(
        scenario_df,
        calibration,
        baseline_weights,
    )

    baseline_metrics = calculate_risk_metrics(
        baseline_losses,
        baseline_drawdowns,
    )

    return {
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": metrics,
        "baseline_scenarios": baseline_scenarios,
        "optimized_scenarios": optimized_scenarios,
        "baseline_monthly": baseline_monthly,
        "optimized_monthly": optimized_monthly,
        "baseline_z": baseline_z,
        "optimized_z": optimized_z,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(
    baseline_metrics: dict,
    optimized_metrics: dict,
    optimized_weights: dict,
    optimizer_result: dict,
    scenario_count: int,
) -> dict:

    checks = {}

    checks["scenario_count"] = (
        scenario_count == N_SCENARIOS_EXPECTED
    )

    checks["weight_sum"] = (
        abs(
            sum(optimized_weights.values())
            - 1.0
        ) < 1e-8
    )

    checks["weights_lower_bound"] = all(
        w >= LOWER_WEIGHT - 1e-8
        for w in optimized_weights.values()
    )

    checks["weights_upper_bound"] = all(
        w <= UPPER_WEIGHT + 1e-8
        for w in optimized_weights.values()
    )

    checks["optimizer_success"] = bool(
        optimizer_result["optimizer_success"]
    )

    checks["baseline_var_valid"] = (
        baseline_metrics["VaR_95"] >= 0
    )

    checks["optimized_var_valid"] = (
        optimized_metrics["VaR_95"] >= 0
    )

    checks["baseline_es_ge_var"] = (
        baseline_metrics["ES_95"]
        >= baseline_metrics["VaR_95"]
    )

    checks["optimized_es_ge_var"] = (
        optimized_metrics["ES_95"]
        >= optimized_metrics["VaR_95"]
    )

    checks["baseline_worst_ge_es"] = (
        baseline_metrics["Worst_Scenario_Loss"]
        >= baseline_metrics["ES_95"]
    )

    checks["optimized_worst_ge_es"] = (
        optimized_metrics["Worst_Scenario_Loss"]
        >= optimized_metrics["ES_95"]
    )

    checks["monetary_consistency"] = (
        abs(
            optimized_metrics["Expected_Loss_INR"]
            - (
                optimized_metrics["Expected_Loss"]
                * PORTFOLIO_VALUE
            )
        ) < 1e-6
    )

    baseline_loss = (
        baseline_metrics["Expected_Loss_INR"]
    )

    optimized_loss = (
        optimized_metrics["Expected_Loss_INR"]
    )

    checks["non_negative_reduction"] = (
        baseline_loss >= optimized_loss - 1e-6
    )

    reduction = (
        baseline_loss
        - optimized_loss
    )

    reduction_pct = (
        reduction / baseline_loss
        if baseline_loss > 0
        else 0.0
    )

    checks["reduction_percentage_valid"] = (
        0.0 <= reduction_pct <= 1.0
    )

    checks["no_nan_weights"] = all(
        np.isfinite(w)
        for w in optimized_weights.values()
    )

    checks["all_checks_pass"] = all(
        bool(v)
        for v in checks.values()
    )

    return {
        "checks": checks,
        "expected_loss_reduction_inr": float(
            reduction
        ),
        "expected_loss_reduction_pct": float(
            reduction_pct
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_header(
        "MACROSTRESS-GAN\n"
        "BANKING PORTFOLIO OPTIMIZER V1"
    )

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_NAME}")
    print(f"TimeGAN      : {TIMEGAN_VERSION}")
    print(
        f"Portfolio    : ₹{PORTFOLIO_VALUE:,.2f}"
    )
    print(
        f"VaR Level    : {VAR_CONFIDENCE * 100:.1f}%"
    )

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    print_header(
        "LOADING REAL BANKING DATA"
    )

    real_df = load_real_data()

    print(
        f"Rows       : {len(real_df)}"
    )

    print(
        f"Features   : {len(FEATURES)}"
    )

    print(
        f"Missing    : {real_df[FEATURES].isna().sum().sum()}"
    )

    # -------------------------------------------------------------------------
    # LOAD SCENARIOS
    # -------------------------------------------------------------------------

    print_header(
        "LOADING BANKING TIMEGAN V4 SCENARIOS"
    )

    scenario_df = load_scenarios()

    scenario_count = (
        len(scenario_df)
        // SEQ_LEN_EXPECTED
    )

    print(
        f"Scenario rows : {len(scenario_df):,}"
    )

    print(
        f"Scenarios     : {scenario_count:,}"
    )

    print(
        f"Sequence len  : {SEQ_LEN_EXPECTED}"
    )

    # -------------------------------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------------------------------

    print_header(
        "CALIBRATING FROM REAL DATA"
    )

    calibration = calibrate_real_data(
        real_df
    )

    for feature in FEATURES:

        c = calibration[feature]

        print(
            f"{feature:<28}"
            f"mean={c['mean']: .6f} "
            f"std={c['std']: .6f} "
            f"q05={c['q05']: .6f} "
            f"q95={c['q95']: .6f}"
        )

    # -------------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------------

    print_header(
        "CALCULATING BASELINE RISK"
    )

    (
        baseline_scenarios,
        baseline_monthly,
        baseline_losses,
        baseline_drawdowns,
        baseline_z,
    ) = calculate_losses(
        scenario_df,
        calibration,
        BASELINE_WEIGHTS,
    )

    baseline_metrics = calculate_risk_metrics(
        baseline_losses,
        baseline_drawdowns,
    )

    print(
        f"Baseline VaR 95%       : "
        f"{baseline_metrics['VaR_95']:.6f}"
    )

    print(
        f"Baseline ES 95%        : "
        f"{baseline_metrics['ES_95']:.6f}"
    )

    print(
        f"Baseline Expected Loss : "
        f"{baseline_metrics['Expected_Loss']:.6f}"
    )

    print(
        f"Baseline Expected Loss : "
        f"₹{baseline_metrics['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Baseline Max Drawdown  : "
        f"{baseline_metrics['Maximum_Drawdown']:.6f}"
    )

    print(
        f"Baseline Worst Loss    : "
        f"{baseline_metrics['Worst_Scenario_Loss']:.6f}"
    )

    # -------------------------------------------------------------------------
    # OPTIMIZATION
    # -------------------------------------------------------------------------

    print_header(
        "OPTIMIZING BANKING RISK WEIGHTS"
    )

    print("Baseline weights:")

    for feature in FEATURES:

        print(
            f"  {feature:<28}"
            f"{BASELINE_WEIGHTS[feature]:.6f}"
        )

    optimization_result = optimize_weights(
        scenario_df,
        calibration,
    )

    optimized_weights = (
        optimization_result["weights"]
    )

    print()
    print(
        "Optimized weights:"
    )

    for feature in FEATURES:

        print(
            f"  {feature:<28}"
            f"{optimized_weights[feature]:.6f}"
        )

    print()
    print(
        f"Optimizer iterations : "
        f"{optimization_result['iterations']}"
    )

    print(
        f"Objective value      : "
        f"{optimization_result['objective_value']:.8f}"
    )

    # -------------------------------------------------------------------------
    # OPTIMIZED RISK
    # -------------------------------------------------------------------------

    print_header(
        "CALCULATING OPTIMIZED RISK"
    )

    (
        optimized_scenarios,
        optimized_monthly,
        optimized_losses,
        optimized_drawdowns,
        optimized_z,
    ) = calculate_losses(
        scenario_df,
        calibration,
        optimized_weights,
    )

    optimized_metrics = calculate_risk_metrics(
        optimized_losses,
        optimized_drawdowns,
    )

    print(
        f"Optimized VaR 95%       : "
        f"{optimized_metrics['VaR_95']:.6f}"
    )

    print(
        f"Optimized VaR 95%       : "
        f"₹{optimized_metrics['VaR_95_INR']:,.2f}"
    )

    print(
        f"Optimized ES 95%        : "
        f"{optimized_metrics['ES_95']:.6f}"
    )

    print(
        f"Optimized ES 95%        : "
        f"₹{optimized_metrics['ES_95_INR']:,.2f}"
    )

    print(
        f"Optimized Expected Loss : "
        f"{optimized_metrics['Expected_Loss']:.6f}"
    )

    print(
        f"Optimized Expected Loss : "
        f"₹{optimized_metrics['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Optimized Max Drawdown  : "
        f"{optimized_metrics['Maximum_Drawdown']:.6f}"
    )

    print(
        f"Optimized Worst Loss    : "
        f"{optimized_metrics['Worst_Scenario_Loss']:.6f}"
    )

    # -------------------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------------------

    baseline_expected_loss = (
        baseline_metrics["Expected_Loss_INR"]
    )

    optimized_expected_loss = (
        optimized_metrics["Expected_Loss_INR"]
    )

    loss_reduction = (
        baseline_expected_loss
        - optimized_expected_loss
    )

    loss_reduction_pct = (
        loss_reduction
        / baseline_expected_loss
        if baseline_expected_loss > 0
        else 0.0
    )

    var_reduction = (
        baseline_metrics["VaR_95_INR"]
        - optimized_metrics["VaR_95_INR"]
    )

    es_reduction = (
        baseline_metrics["ES_95_INR"]
        - optimized_metrics["ES_95_INR"]
    )

    drawdown_reduction = (
        baseline_metrics["Maximum_Drawdown_INR"]
        - optimized_metrics["Maximum_Drawdown_INR"]
    )

    worst_loss_reduction = (
        baseline_metrics["Worst_Scenario_Loss_INR"]
        - optimized_metrics["Worst_Scenario_Loss_INR"]
    )

    print_header(
        "BASELINE VS OPTIMIZED"
    )

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
        f"₹{baseline_expected_loss:>15,.2f}"
        f"₹{optimized_expected_loss:>15,.2f}"
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
        f"₹{loss_reduction:,.2f}"
    )

    print(
        f"Expected Loss Reduction : "
        f"{loss_reduction_pct * 100:.4f}%"
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
    # VARIABLE CONTRIBUTIONS
    # -------------------------------------------------------------------------

    print_header(
        "OPTIMIZED VARIABLE CONTRIBUTIONS"
    )

    contribution_df = calculate_variable_contributions(
        optimized_z,
        optimized_weights,
    )

    for _, row in contribution_df.iterrows():

        print(
            f"{row['feature']:<28}"
            f"weight={row['weight']:.4f} "
            f"contribution="
            f"{row['contribution_pct']:.4f}"
        )

    # -------------------------------------------------------------------------
    # INDEPENDENT RESTRESS
    # -------------------------------------------------------------------------

    print_header(
        "INDEPENDENT RE-STRESS VERIFICATION"
    )

    restress = independent_restress(
        scenario_df,
        calibration,
        optimized_weights,
    )

    restress_baseline = (
        restress["baseline_metrics"]
    )

    restress_optimized = (
        restress["optimized_metrics"]
    )

    restress_reduction = (
        restress_baseline["Expected_Loss_INR"]
        - restress_optimized["Expected_Loss_INR"]
    )

    restress_reduction_pct = (
        restress_reduction
        / restress_baseline["Expected_Loss_INR"]
        if restress_baseline["Expected_Loss_INR"] > 0
        else 0.0
    )

    print(
        f"Independent baseline loss : "
        f"₹{restress_baseline['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Independent optimized loss : "
        f"₹{restress_optimized['Expected_Loss_INR']:,.2f}"
    )

    print(
        f"Independent reduction      : "
        f"₹{restress_reduction:,.2f}"
    )

    print(
        f"Independent reduction      : "
        f"{restress_reduction_pct * 100:.4f}%"
    )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    print_header(
        "OPTIMIZATION VALIDATION"
    )

    validation = validate_results(
        baseline_metrics,
        optimized_metrics,
        optimized_weights,
        optimization_result,
        scenario_count,
    )

    for check_name, status in validation[
        "checks"
    ].items():

        print(
            f"{check_name:<35}: "
            f"{'PASS' if status else 'FAIL'}"
        )

    overall_status = (
        "PASS"
        if validation["checks"]["all_checks_pass"]
        else "REVIEW REQUIRED"
    )

    print()
    print(
        f"BANKING PORTFOLIO OPTIMIZER V1 : "
        f"{overall_status}"
    )

    # -------------------------------------------------------------------------
    # SAVE SCENARIOS
    # -------------------------------------------------------------------------

    baseline_scenarios.to_csv(
        OUTPUT_DIR
        / "banking_optimization_v1_baseline_scenarios.csv",
        index=False,
    )

    optimized_scenarios.to_csv(
        OPTIMIZED_SCENARIO_CSV,
        index=False,
    )

    optimized_monthly.to_csv(
        OPTIMIZED_MONTHLY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE WEIGHTS
    # -------------------------------------------------------------------------

    weight_rows = []

    for feature in FEATURES:

        weight_rows.append(
            {
                "feature": feature,
                "baseline_weight": BASELINE_WEIGHTS[
                    feature
                ],
                "optimized_weight": optimized_weights[
                    feature
                ],
                "weight_change": (
                    optimized_weights[feature]
                    - BASELINE_WEIGHTS[feature]
                ),
            }
        )

    weight_df = pd.DataFrame(
        weight_rows
    )

    weight_df.to_csv(
        WEIGHT_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    summary_rows = [

        {
            "metric": "VaR_95",
            "baseline": baseline_metrics["VaR_95"],
            "optimized": optimized_metrics["VaR_95"],
            "baseline_inr": baseline_metrics[
                "VaR_95_INR"
            ],
            "optimized_inr": optimized_metrics[
                "VaR_95_INR"
            ],
        },

        {
            "metric": "ES_95",
            "baseline": baseline_metrics["ES_95"],
            "optimized": optimized_metrics["ES_95"],
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

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE RESTRESS
    # -------------------------------------------------------------------------

    restress_payload = {

        "model": MODEL_NAME,

        "timegan_version": TIMEGAN_VERSION,

        "portfolio_value_inr": PORTFOLIO_VALUE,

        "scenario_count": scenario_count,

        "sequence_length": SEQ_LEN_EXPECTED,

        "optimized_weights": optimized_weights,

        "baseline_expected_loss_inr":
            restress_baseline[
                "Expected_Loss_INR"
            ],

        "optimized_expected_loss_inr":
            restress_optimized[
                "Expected_Loss_INR"
            ],

        "expected_loss_reduction_inr":
            restress_reduction,

        "expected_loss_reduction_pct":
            restress_reduction_pct,

        "baseline_metrics":
            restress_baseline,

        "optimized_metrics":
            restress_optimized,

        "validation":
            {
                "independent_reduction_non_negative":
                    restress_reduction >= -1e-6
            },
    }

    with open(
        RESTRESS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_safe(restress_payload),
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # SAVE MAIN RESULT JSON
    # -------------------------------------------------------------------------

    result_payload = {

        "model": MODEL_NAME,

        "timegan_version": TIMEGAN_VERSION,

        "portfolio_value_inr": PORTFOLIO_VALUE,

        "var_confidence": VAR_CONFIDENCE,

        "loss_scale": LOSS_SCALE,

        "max_monthly_loss_rate":
            MAX_MONTHLY_LOSS_RATE,

        "scenario_count": scenario_count,

        "sequence_length": SEQ_LEN_EXPECTED,

        "features": FEATURES,

        "adverse_direction":
            ADVERSE_DIRECTION,

        "optimization_bounds": {
            "lower": LOWER_WEIGHT,
            "upper": UPPER_WEIGHT,
        },

        "baseline_weights":
            BASELINE_WEIGHTS,

        "optimized_weights":
            optimized_weights,

        "optimizer": optimization_result,

        "baseline_metrics":
            baseline_metrics,

        "optimized_metrics":
            optimized_metrics,

        "comparison": {

            "expected_loss_reduction_inr":
                loss_reduction,

            "expected_loss_reduction_pct":
                loss_reduction_pct,

            "var_reduction_inr":
                var_reduction,

            "es_reduction_inr":
                es_reduction,

            "drawdown_reduction_inr":
                drawdown_reduction,

            "worst_loss_reduction_inr":
                worst_loss_reduction,
        },

        "independent_restress": {

            "baseline_expected_loss_inr":
                restress_baseline[
                    "Expected_Loss_INR"
                ],

            "optimized_expected_loss_inr":
                restress_optimized[
                    "Expected_Loss_INR"
                ],

            "reduction_inr":
                restress_reduction,

            "reduction_pct":
                restress_reduction_pct,
        },

        "validation":
            validation,

        "status":
            overall_status,

        "methodology_note":
            "Optimization uses explicit modeling "
            "sensitivity weights and does not claim "
            "to represent actual bank balance-sheet "
            "exposures. TimeGAN V4 is an event-aware "
            "TimeGAN-style hybrid, not canonical "
            "TimeGAN.",
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_safe(result_payload),
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # SAVE VALIDATION JSON
    # -------------------------------------------------------------------------

    with open(
        VALIDATION_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_safe(
                validation
            ),
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print_header(
        "BANKING PORTFOLIO OPTIMIZATION V1 COMPLETE"
    )

    print(
        f"Baseline Expected Loss : "
        f"₹{baseline_expected_loss:,.2f}"
    )

    print(
        f"Optimized Expected Loss : "
        f"₹{optimized_expected_loss:,.2f}"
    )

    print(
        f"Expected Loss Reduction : "
        f"₹{loss_reduction:,.2f}"
    )

    print(
        f"Expected Loss Reduction : "
        f"{loss_reduction_pct * 100:.4f}%"
    )

    print(
        f"Baseline VaR 95% : "
        f"₹{baseline_metrics['VaR_95_INR']:,.2f}"
    )

    print(
        f"Optimized VaR 95% : "
        f"₹{optimized_metrics['VaR_95_INR']:,.2f}"
    )

    print(
        f"Baseline ES 95% : "
        f"₹{baseline_metrics['ES_95_INR']:,.2f}"
    )

    print(
        f"Optimized ES 95% : "
        f"₹{optimized_metrics['ES_95_INR']:,.2f}"
    )

    print()
    print(
        f"STATUS : {overall_status}"
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"Results JSON : {RESULT_JSON}"
    )

    print(
        f"Summary CSV  : {SUMMARY_CSV}"
    )

    print(
        f"Weights CSV  : {WEIGHT_CSV}"
    )

    print(
        f"Re-stress    : {RESTRESS_JSON}"
    )

    print(
        f"Validation   : {VALIDATION_JSON}"
    )


if __name__ == "__main__":
    main()