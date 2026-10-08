"""
===============================================================================
MACROSTRESS-GAN
BANKING RISK ENGINE V1
===============================================================================

Purpose
-------
Calculate banking stress-test risk metrics from Banking TimeGAN V4 scenarios.

Pipeline
--------
Real Banking Dataset
        |
        v
Banking TimeGAN V4 Synthetic Scenarios
        |
        v
Scenario Stress Scoring
        |
        +---- VaR 95%
        +---- ES 95%
        +---- Expected Loss
        +---- Maximum Drawdown
        +---- Worst Scenario Loss
        +---- Monetary Loss
        +---- Variable Contributions
        |
        v
Risk Report

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
TimeGAN V4 is an event-aware TimeGAN-style hybrid:
    - neural generation for continuous variables
    - historical event-aware reconstruction for repo-rate changes

This script does NOT modify the V4 model or real input data.

The portfolio value, variable sensitivities and loss scale below are
explicit MODELING ASSUMPTIONS. They should later be calibrated with
institution-specific exposure data.

No synthetic observations are added to the real training dataset.
===============================================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

REAL_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "institutions"
    / "banking"
    / "banking_monthly_dataset_long.csv"
)

SYNTHETIC_DATA_PATH = (
    ROOT
    / "outputs"
    / "banking"
    / "banking_timegan_v4_synthetic.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "banking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# MODEL INFORMATION
# =============================================================================

MODEL_VERSION = "Banking_Risk_Engine_V1"

TIMEGAN_VERSION = "TimeGAN_Banking_V4"


# =============================================================================
# RISK CONFIGURATION
# =============================================================================

# Explicit modeling assumption.
# ₹10 crore.
PORTFOLIO_VALUE_INR = 100_000_000.0

# 95% VaR.
VAR_CONFIDENCE = 0.95

# Stress score -> loss conversion.
#
# Example:
# stress score = 1.0
# loss = 1.0 × 0.01 = 1%
#
# This is an explicit modeling assumption.
LOSS_SCALE = 0.01

# Maximum monthly loss rate.
MAX_MONTHLY_LOSS_RATE = 0.05


# =============================================================================
# ADVERSE DIRECTIONS
# =============================================================================
#
# -1 means negative movement is adverse.
# +1 means positive movement is adverse.
#
# These represent stress-test directions, not causal estimates.
# =============================================================================

ADVERSE_DIRECTION = {

    # Falling Bank Nifty = stress
    "BANK_NIFTY_Return": -1.0,

    # Rising repo rate = stress
    "REPO_RATE_Change": 1.0,

    # Rising inflation = stress
    "CPI_INFLATION_Change": 1.0,

    # Falling credit growth = stress
    "CREDIT_GROWTH_Change": -1.0,

    # Rising USD/INR = INR depreciation = stress
    "USD_INR_Return": 1.0,
}


# =============================================================================
# VARIABLE WEIGHTS
# =============================================================================
#
# These are explicit modeling assumptions.
#
# They should later be replaced/calibrated with institution-specific
# exposure/sensitivity information.
# =============================================================================

VARIABLE_WEIGHTS = {

    "BANK_NIFTY_Return": 0.30,

    "REPO_RATE_Change": 0.20,

    "CPI_INFLATION_Change": 0.15,

    "CREDIT_GROWTH_Change": 0.20,

    "USD_INR_Return": 0.15,
}


# =============================================================================
# REQUIRED FEATURES
# =============================================================================

FEATURES = [

    "BANK_NIFTY_Return",

    "REPO_RATE_Change",

    "CPI_INFLATION_Change",

    "CREDIT_GROWTH_Change",

    "USD_INR_Return",
]


# =============================================================================
# PRINTING
# =============================================================================

def print_header(title: str) -> None:

    print()

    print(
        "=" * 90
    )

    print(title)

    print(
        "=" * 90
    )


# =============================================================================
# SAFE FLOAT
# =============================================================================

def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):

            return None

        return value

    except Exception:

        return None


# =============================================================================
# DATAFRAME COLUMN FINDER
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
):

    normalized = {

        str(column)
        .strip()
        .lower(): column

        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in normalized:

            return normalized[key]

    return None


# =============================================================================
# LOAD REAL DATA
# =============================================================================

def load_real_data() -> pd.DataFrame:

    print_header(
        "LOADING REAL BANKING DATASET"
    )

    if not REAL_DATA_PATH.exists():

        raise FileNotFoundError(
            "Real banking dataset not found:\n"
            f"{REAL_DATA_PATH}"
        )

    df = pd.read_csv(
        REAL_DATA_PATH
    )

    print(
        f"Path       : {REAL_DATA_PATH}"
    )

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    missing_features = [

        feature

        for feature in FEATURES

        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required real-data features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    df = df.copy()

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.sort_values(
            "Date"
        )

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    before = len(df)

    df = df.dropna(
        subset=FEATURES
    ).reset_index(
        drop=True
    )

    dropped = (
        before - len(df)
    )

    print(
        f"Usable rows: {len(df):,}"
    )

    print(
        f"Dropped    : {dropped:,}"
    )

    if len(df) < 30:

        raise ValueError(
            "Insufficient real observations."
        )

    return df


# =============================================================================
# LOAD V4 SYNTHETIC DATA
# =============================================================================

def load_synthetic_data() -> pd.DataFrame:

    print_header(
        "LOADING BANKING TIMEGAN V4 SCENARIOS"
    )

    if not SYNTHETIC_DATA_PATH.exists():

        raise FileNotFoundError(
            "Banking TimeGAN V4 synthetic file not found:\n"
            f"{SYNTHETIC_DATA_PATH}"
        )

    df = pd.read_csv(
        SYNTHETIC_DATA_PATH
    )

    print(
        f"Path       : {SYNTHETIC_DATA_PATH}"
    )

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    missing_features = [

        feature

        for feature in FEATURES

        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required V4 features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # -------------------------------------------------------------------------
    # NaN check
    # -------------------------------------------------------------------------

    nan_counts = (
        df[FEATURES]
        .isna()
        .sum()
    )

    if nan_counts.sum() > 0:

        print()

        print(
            "NaN values detected:"
        )

        print(
            nan_counts
        )

        raise ValueError(
            "Synthetic V4 dataset contains NaN values."
        )

    # -------------------------------------------------------------------------
    # Infinity check
    # -------------------------------------------------------------------------

    if np.isinf(
        df[FEATURES]
        .to_numpy(
            dtype=float
        )
    ).any():

        raise ValueError(
            "Synthetic V4 dataset contains Infinity values."
        )

    return df


# =============================================================================
# IDENTIFY SCENARIO/TIMESTEP COLUMNS
# =============================================================================

def identify_scenario_columns(
    df: pd.DataFrame,
):

    scenario_col = find_column(
        df,
        [
            "sequence_id",
            "Sequence_ID",
            "scenario_id",
            "Scenario_ID",
            "sequence",
            "scenario",
            "sample_id",
            "Sample_ID",
        ],
    )

    timestep_col = find_column(
        df,
        [
            "timestep",
            "Timestep",
            "time_step",
            "Time_Step",
            "step",
            "Step",
        ],
    )

    return (
        scenario_col,
        timestep_col,
    )


# =============================================================================
# PREPARE SCENARIO STRUCTURE
# =============================================================================

def prepare_scenario_structure(
    synthetic: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "VALIDATING SCENARIO STRUCTURE"
    )

    df = synthetic.copy()

    (
        scenario_col,
        timestep_col,
    ) = identify_scenario_columns(
        df
    )

    # -------------------------------------------------------------------------
    # Scenario ID
    # -------------------------------------------------------------------------

    if scenario_col is not None:

        df["_scenario_id"] = (
            df[scenario_col]
        )

        unique_ids = {

            value: index

            for index, value

            in enumerate(
                pd.unique(
                    df["_scenario_id"]
                )
            )
        }

        df["_scenario_id"] = (
            df["_scenario_id"]
            .map(
                unique_ids
            )
        )

    else:

        sequence_length = 30

        if len(df) % sequence_length != 0:

            raise ValueError(
                "Synthetic dataset does not contain "
                "an explicit scenario ID and its row "
                "count is not divisible by 30."
            )

        df["_scenario_id"] = (

            np.arange(
                len(df)
            )

            // sequence_length
        )

    # -------------------------------------------------------------------------
    # Timestep
    # -------------------------------------------------------------------------

    if timestep_col is not None:

        df["_timestep"] = pd.to_numeric(
            df[timestep_col],
            errors="coerce"
        )

        if df["_timestep"].isna().any():

            raise ValueError(
                "Timestep column contains invalid values."
            )

    else:

        df["_timestep"] = (

            df.groupby(
                "_scenario_id"
            )
            .cumcount()
        )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    df = df.sort_values(
        [
            "_scenario_id",
            "_timestep",
        ]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Sequence size validation
    # -------------------------------------------------------------------------

    scenario_sizes = (

        df.groupby(
            "_scenario_id"
        )
        .size()
    )

    print(
        f"Scenarios       : "
        f"{len(scenario_sizes):,}"
    )

    print(
        f"Min sequence len: "
        f"{scenario_sizes.min()}"
    )

    print(
        f"Max sequence len: "
        f"{scenario_sizes.max()}"
    )

    if not (
        scenario_sizes == 30
    ).all():

        invalid = (
            scenario_sizes[
                scenario_sizes != 30
            ]
        )

        print(
            "Invalid sequence lengths:"
        )

        print(
            invalid.head(20)
        )

        raise ValueError(
            "Not every scenario contains exactly 30 months."
        )

    print(
        "Every sequence = 30 : PASS"
    )

    # -------------------------------------------------------------------------
    # Timestep continuity
    # -------------------------------------------------------------------------

    expected_steps = np.arange(
        30
    )

    continuity_ok = True

    for (
        scenario_id,
        group,
    ) in df.groupby(
        "_scenario_id"
    ):

        actual_steps = (
            group[
                "_timestep"
            ]
            .to_numpy()
        )

        if not np.array_equal(
            actual_steps,
            expected_steps
        ):

            continuity_ok = False

            print(
                "Invalid timestep sequence "
                f"for scenario {scenario_id}"
            )

            break

    print(
        "Timestep continuity : "
        + (
            "PASS"
            if continuity_ok
            else "FAIL"
        )
    )

    if not continuity_ok:

        raise ValueError(
            "Scenario timestep continuity failed."
        )

    print(
        f"Total rows          : "
        f"{len(df):,}"
    )

    print(
        f"Expected rows       : "
        f"{len(scenario_sizes) * 30:,}"
    )

    return df


# =============================================================================
# REAL DATA CALIBRATION
# =============================================================================

def calculate_real_statistics(
    real: pd.DataFrame,
) -> dict:

    print_header(
        "CALIBRATING RISK DISTRIBUTIONS FROM REAL DATA"
    )

    statistics = {}

    for feature in FEATURES:

        values = (
            real[feature]
            .to_numpy(
                dtype=float
            )
        )

        mean = float(
            np.mean(
                values
            )
        )

        std = float(
            np.std(
                values,
                ddof=1
            )
        )

        q05 = float(
            np.quantile(
                values,
                0.05
            )
        )

        q95 = float(
            np.quantile(
                values,
                0.95
            )
        )

        if std <= 1e-12:

            warnings.warn(
                f"{feature} has near-zero standard deviation."
            )

        statistics[feature] = {

            "mean": mean,

            "std": std,

            "q05": q05,

            "q95": q95,
        }

        print(
            f"{feature:<28} "
            f"mean={mean: .6f} "
            f"std={std: .6f} "
            f"q05={q05: .6f} "
            f"q95={q95: .6f}"
        )

    return statistics


# =============================================================================
# ADVERSE Z-SCORES
# =============================================================================

def calculate_adverse_z_scores(
    scenario: pd.DataFrame,
    real_statistics: dict,
) -> pd.DataFrame:

    output = pd.DataFrame(
        index=scenario.index
    )

    for feature in FEATURES:

        mean = (
            real_statistics[
                feature
            ]["mean"]
        )

        std = (
            real_statistics[
                feature
            ]["std"]
        )

        if std <= 1e-12:

            z = np.zeros(
                len(scenario)
            )

        else:

            z = (

                scenario[feature]
                .to_numpy(
                    dtype=float
                )

                - mean

            ) / std

        direction = (
            ADVERSE_DIRECTION[
                feature
            ]
        )

        adverse_z = (
            z * direction
        )

        # Only adverse movement contributes
        # to stress losses.
        adverse_z = np.maximum(
            adverse_z,
            0.0
        )

        output[
            f"{feature}_adverse_z"
        ] = adverse_z

    return output


# =============================================================================
# VARIABLE CONTRIBUTIONS
# =============================================================================

def calculate_variable_contributions(
    adverse_z: pd.DataFrame,
) -> pd.DataFrame:

    contributions = pd.DataFrame(
        index=adverse_z.index
    )

    for feature in FEATURES:

        z_column = (
            f"{feature}_adverse_z"
        )

        contributions[
            feature
        ] = (

            adverse_z[
                z_column
            ]

            * VARIABLE_WEIGHTS[
                feature
            ]
        )

    contributions[
        "total_stress_score"
    ] = contributions.sum(
        axis=1
    )

    return contributions


# =============================================================================
# MONTHLY LOSS RATE
# =============================================================================

def calculate_monthly_loss_rate(
    stress_score: pd.Series,
) -> pd.Series:

    loss_rate = (

        stress_score
        * LOSS_SCALE
    )

    loss_rate = np.clip(
        loss_rate,
        0.0,
        MAX_MONTHLY_LOSS_RATE
    )

    return pd.Series(
        loss_rate,
        index=stress_score.index,
        name="monthly_loss_rate",
    )


# =============================================================================
# SCENARIO LOSS CALCULATION
# =============================================================================

def calculate_scenario_metrics(
    prepared: pd.DataFrame,
    real_statistics: dict,
):

    print_header(
        "CALCULATING SCENARIO LOSSES"
    )

    # -------------------------------------------------------------------------
    # Calculate adverse shocks
    # -------------------------------------------------------------------------

    adverse_z = (
        calculate_adverse_z_scores(
            prepared,
            real_statistics
        )
    )

    # -------------------------------------------------------------------------
    # Calculate variable contributions
    # -------------------------------------------------------------------------

    contributions = (
        calculate_variable_contributions(
            adverse_z
        )
    )

    # -------------------------------------------------------------------------
    # Calculate monthly loss
    # -------------------------------------------------------------------------

    monthly_loss_rate = (
        calculate_monthly_loss_rate(
            contributions[
                "total_stress_score"
            ]
        )
    )

    working = prepared.copy()

    working[
        "stress_score"
    ] = contributions[
        "total_stress_score"
    ]

    working[
        "monthly_loss_rate"
    ] = monthly_loss_rate

    # -------------------------------------------------------------------------
    # Add individual contribution columns
    # -------------------------------------------------------------------------

    for feature in FEATURES:

        working[
            f"{feature}_contribution"
        ] = contributions[
            feature
        ]

    # -------------------------------------------------------------------------
    # Calculate each scenario
    # -------------------------------------------------------------------------

    scenario_rows = []

    for (
        scenario_id,
        group,
    ) in working.groupby(
        "_scenario_id"
    ):

        group = group.sort_values(
            "_timestep"
        )

        monthly_rates = (

            group[
                "monthly_loss_rate"
            ]
            .to_numpy(
                dtype=float
            )
        )

        stress_scores = (

            group[
                "stress_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ---------------------------------------------------------------------
        # Portfolio value path
        # ---------------------------------------------------------------------

        portfolio_path = np.empty(
            len(monthly_rates) + 1,
            dtype=float
        )

        portfolio_path[0] = (
            PORTFOLIO_VALUE_INR
        )

        for index, loss_rate in enumerate(
            monthly_rates
        ):

            portfolio_path[
                index + 1
            ] = (

                portfolio_path[index]

                * (
                    1.0
                    - loss_rate
                )
            )

        # ---------------------------------------------------------------------
        # Final loss
        # ---------------------------------------------------------------------

        final_value = (
            portfolio_path[-1]
        )

        total_loss_inr = (

            PORTFOLIO_VALUE_INR
            - final_value
        )

        total_loss_pct = (

            total_loss_inr
            / PORTFOLIO_VALUE_INR
        )

        # ---------------------------------------------------------------------
        # Maximum drawdown
        # ---------------------------------------------------------------------

        running_peak = (
            np.maximum.accumulate(
                portfolio_path
            )
        )

        drawdown = (

            portfolio_path
            / running_peak
            - 1.0
        )

        max_drawdown = abs(
            float(
                np.min(
                    drawdown
                )
            )
        )

        # ---------------------------------------------------------------------
        # Other scenario statistics
        # ---------------------------------------------------------------------

        worst_monthly_loss = float(
            np.max(
                monthly_rates
            )
        )

        mean_stress = float(
            np.mean(
                stress_scores
            )
        )

        maximum_stress = float(
            np.max(
                stress_scores
            )
        )

        row = {

            "scenario_id":
                int(scenario_id),

            "months":
                int(len(group)),

            "expected_loss_pct":
                float(total_loss_pct),

            "expected_loss_inr":
                float(total_loss_inr),

            "maximum_drawdown_pct":
                float(max_drawdown),

            "worst_monthly_loss_pct":
                worst_monthly_loss,

            "mean_stress_score":
                mean_stress,

            "maximum_stress_score":
                maximum_stress,
        }

        # ---------------------------------------------------------------------
        # Average variable contributions
        # ---------------------------------------------------------------------

        for feature in FEATURES:

            contribution_column = (
                f"{feature}_contribution"
            )

            row[
                f"{feature}_mean_contribution"
            ] = float(
                group[
                    contribution_column
                ].mean()
            )

        scenario_rows.append(
            row
        )

    scenario_results = pd.DataFrame(
        scenario_rows
    )

    # -------------------------------------------------------------------------
    # Contribution percentages
    # -------------------------------------------------------------------------

    contribution_columns = [

        f"{feature}_mean_contribution"

        for feature in FEATURES
    ]

    contribution_total = (

        scenario_results[
            contribution_columns
        ]
        .sum(
            axis=1
        )
    )

    for feature in FEATURES:

        contribution_column = (
            f"{feature}_mean_contribution"
        )

        percentage_column = (
            f"{feature}_contribution_pct"
        )

        scenario_results[
            percentage_column
        ] = np.where(

            contribution_total > 0,

            scenario_results[
                contribution_column
            ]
            / contribution_total,

            0.0,
        )

    print(
        f"Scenario count : "
        f"{len(scenario_results):,}"
    )

    print(
        f"Mean loss %    : "
        f"{scenario_results['expected_loss_pct'].mean():.6f}"
    )

    print(
        f"Mean loss INR  : ₹"
        f"{scenario_results['expected_loss_inr'].mean():,.2f}"
    )

    return (
        scenario_results,
        working,
    )


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_risk_metrics(
    scenario_results: pd.DataFrame,
) -> dict:

    print_header(
        "BANKING RISK METRICS"
    )

    losses_pct = (

        scenario_results[
            "expected_loss_pct"
        ]
        .to_numpy(
            dtype=float
        )
    )

    losses_inr = (

        scenario_results[
            "expected_loss_inr"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # =========================================================================
    # VaR 95%
    # =========================================================================

    var_pct = float(
        np.quantile(
            losses_pct,
            VAR_CONFIDENCE
        )
    )

    var_inr = (

        var_pct
        * PORTFOLIO_VALUE_INR
    )

    # =========================================================================
    # Expected Shortfall 95%
    # =========================================================================

    tail_mask = (
        losses_pct >= var_pct
    )

    tail_losses_pct = (
        losses_pct[
            tail_mask
        ]
    )

    tail_losses_inr = (
        losses_inr[
            tail_mask
        ]
    )

    if len(
        tail_losses_pct
    ) == 0:

        es_pct = var_pct

        es_inr = var_inr

    else:

        es_pct = float(
            np.mean(
                tail_losses_pct
            )
        )

        es_inr = float(
            np.mean(
                tail_losses_inr
            )
        )

    # =========================================================================
    # Expected Loss
    # =========================================================================

    expected_loss_pct = float(
        np.mean(
            losses_pct
        )
    )

    expected_loss_inr = float(
        np.mean(
            losses_inr
        )
    )

    # =========================================================================
    # Maximum Drawdown
    # =========================================================================

    maximum_drawdown_pct = float(
        scenario_results[
            "maximum_drawdown_pct"
        ].max()
    )

    maximum_drawdown_inr = (

        maximum_drawdown_pct
        * PORTFOLIO_VALUE_INR
    )

    # =========================================================================
    # Worst Scenario
    # =========================================================================

    worst_index = int(
        np.argmax(
            losses_pct
        )
    )

    worst_scenario_loss_pct = float(
        losses_pct[
            worst_index
        ]
    )

    worst_scenario_loss_inr = float(
        losses_inr[
            worst_index
        ]
    )

    worst_scenario_id = int(
        scenario_results.iloc[
            worst_index
        ]["scenario_id"]
    )

    # =========================================================================
    # Additional distribution statistics
    # =========================================================================

    median_loss_pct = float(
        np.median(
            losses_pct
        )
    )

    p75_loss_pct = float(
        np.quantile(
            losses_pct,
            0.75
        )
    )

    p90_loss_pct = float(
        np.quantile(
            losses_pct,
            0.90
        )
    )

    p99_loss_pct = float(
        np.quantile(
            losses_pct,
            0.99
        )
    )

    # =========================================================================
    # Metrics object
    # =========================================================================

    metrics = {

        "model_version":
            MODEL_VERSION,

        "timegan_version":
            TIMEGAN_VERSION,

        "portfolio_value_inr":
            float(PORTFOLIO_VALUE_INR),

        "var_confidence":
            float(VAR_CONFIDENCE),

        "loss_scale":
            float(LOSS_SCALE),

        "var_95_pct":
            var_pct,

        "var_95_inr":
            var_inr,

        "expected_shortfall_95_pct":
            es_pct,

        "expected_shortfall_95_inr":
            es_inr,

        "expected_loss_pct":
            expected_loss_pct,

        "expected_loss_inr":
            expected_loss_inr,

        "maximum_drawdown_pct":
            maximum_drawdown_pct,

        "maximum_drawdown_inr":
            maximum_drawdown_inr,

        "worst_scenario_loss_pct":
            worst_scenario_loss_pct,

        "worst_scenario_loss_inr":
            worst_scenario_loss_inr,

        "worst_scenario_id":
            worst_scenario_id,

        "median_loss_pct":
            median_loss_pct,

        "p75_loss_pct":
            p75_loss_pct,

        "p90_loss_pct":
            p90_loss_pct,

        "p99_loss_pct":
            p99_loss_pct,

        "scenario_count":
            int(
                len(
                    scenario_results
                )
            ),
    }

    # =========================================================================
    # Print results
    # =========================================================================

    print(
        f"Portfolio Value       : "
        f"₹{PORTFOLIO_VALUE_INR:,.2f}"
    )

    print(
        f"Scenario Count        : "
        f"{len(scenario_results):,}"
    )

    print()

    print(
        f"VaR 95%               : "
        f"{var_pct:.6f} "
        f"(₹{var_inr:,.2f})"
    )

    print(
        f"Expected Shortfall 95%: "
        f"{es_pct:.6f} "
        f"(₹{es_inr:,.2f})"
    )

    print(
        f"Expected Loss         : "
        f"{expected_loss_pct:.6f} "
        f"(₹{expected_loss_inr:,.2f})"
    )

    print(
        f"Maximum Drawdown      : "
        f"{maximum_drawdown_pct:.6f} "
        f"(₹{maximum_drawdown_inr:,.2f})"
    )

    print(
        f"Worst Scenario Loss   : "
        f"{worst_scenario_loss_pct:.6f} "
        f"(₹{worst_scenario_loss_inr:,.2f})"
    )

    print(
        f"Worst Scenario ID    : "
        f"{worst_scenario_id}"
    )

    return metrics


# =============================================================================
# VARIABLE IMPACT
# =============================================================================

def calculate_variable_impact(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "VARIABLE-LEVEL RISK CONTRIBUTION"
    )

    rows = []

    for feature in FEATURES:

        contribution_column = (
            f"{feature}_mean_contribution"
        )

        percentage_column = (
            f"{feature}_contribution_pct"
        )

        mean_contribution = float(
            scenario_results[
                contribution_column
            ].mean()
        )

        mean_percentage = float(
            scenario_results[
                percentage_column
            ].mean()
        )

        configured_weight = float(
            VARIABLE_WEIGHTS[
                feature
            ]
        )

        if (
            ADVERSE_DIRECTION[
                feature
            ] < 0
        ):

            direction = (
                "Negative movement"
            )

        else:

            direction = (
                "Positive movement"
            )

        rows.append({

            "variable":
                feature,

            "direction_of_adverse_stress":
                direction,

            "configured_weight":
                configured_weight,

            "mean_stress_contribution":
                mean_contribution,

            "mean_contribution_percentage":
                mean_percentage,
        })

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "mean_contribution_percentage",
        ascending=False
    ).reset_index(
        drop=True
    )

    print()

    for _, row in result.iterrows():

        print(
            f"{row['variable']:<28} "
            f"weight="
            f"{row['configured_weight']:.3f} "
            f"contribution="
            f"{row['mean_contribution_percentage']:.4f}"
        )

    return result


# =============================================================================
# RISK VALIDATION
# =============================================================================

def validate_risk_output(
    scenario_results: pd.DataFrame,
    metrics: dict,
) -> dict:

    print_header(
        "RISK ENGINE VALIDATION"
    )

    checks = {}

    # -------------------------------------------------------------------------
    # Scenario count
    # -------------------------------------------------------------------------

    checks[
        "scenario_count"
    ] = bool(
        len(
            scenario_results
        ) == 1000
    )

    # -------------------------------------------------------------------------
    # Every scenario = 30 months
    # -------------------------------------------------------------------------

    checks[
        "30_month_scenarios"
    ] = bool(
        (
            scenario_results[
                "months"
            ]
            == 30
        ).all()
    )

    # -------------------------------------------------------------------------
    # No NaN
    # -------------------------------------------------------------------------

    checks[
        "no_nan_losses"
    ] = bool(
        not (
            scenario_results[
                "expected_loss_pct"
            ]
            .isna()
            .any()
        )
    )

    # -------------------------------------------------------------------------
    # No infinity
    # -------------------------------------------------------------------------

    checks[
        "no_infinite_losses"
    ] = bool(
        not np.isinf(
            scenario_results[
                "expected_loss_pct"
            ]
            .to_numpy(
                dtype=float
            )
        ).any()
    )

    # -------------------------------------------------------------------------
    # Non-negative losses
    # -------------------------------------------------------------------------

    checks[
        "non_negative_losses"
    ] = bool(
        (
            scenario_results[
                "expected_loss_pct"
            ]
            >= 0
        ).all()
    )

    # -------------------------------------------------------------------------
    # VaR >= Expected Loss
    # -------------------------------------------------------------------------

    checks[
        "var_ge_expected_loss"
    ] = bool(
        metrics[
            "var_95_pct"
        ]
        >=
        metrics[
            "expected_loss_pct"
        ]
    )

    # -------------------------------------------------------------------------
    # ES >= VaR
    # -------------------------------------------------------------------------

    checks[
        "es_ge_var"
    ] = bool(
        metrics[
            "expected_shortfall_95_pct"
        ]
        >=
        metrics[
            "var_95_pct"
        ]
    )

    # -------------------------------------------------------------------------
    # Worst >= ES
    # -------------------------------------------------------------------------

    checks[
        "worst_ge_es"
    ] = bool(
        metrics[
            "worst_scenario_loss_pct"
        ]
        >=
        metrics[
            "expected_shortfall_95_pct"
        ]
    )

    # -------------------------------------------------------------------------
    # Monetary consistency
    # -------------------------------------------------------------------------

    checks[
        "monetary_consistency"
    ] = bool(
        np.isclose(
            metrics[
                "expected_loss_inr"
            ],

            metrics[
                "expected_loss_pct"
            ]
            * PORTFOLIO_VALUE_INR,

            rtol=1e-8,

            atol=1e-5,
        )
    )

    # -------------------------------------------------------------------------
    # Print checks
    # -------------------------------------------------------------------------

    for name, passed in checks.items():

        print(
            f"{name:<32}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall = bool(
        all(
            checks.values()
        )
    )

    print()

    print(
        "BANKING RISK ENGINE V1 : "
        + (
            "PASS"
            if overall
            else "REVIEW REQUIRED"
        )
    )

    return {

        "checks":
            checks,

        "overall_status":
            (
                "PASS"
                if overall
                else "REVIEW REQUIRED"
            ),
    }


# =============================================================================
# JSON-SAFE CONVERTER
# =============================================================================
#
# This recursively converts:
#
# np.bool_
# np.int64
# np.float64
# numpy arrays
# pandas timestamps
#
# into normal Python JSON-compatible objects.
#
# This prevents the error:
#
# TypeError: Object of type bool is not JSON serializable
# =============================================================================

def make_json_safe(value):

    # -------------------------------------------------------------------------
    # None
    # -------------------------------------------------------------------------

    if value is None:

        return None

    # -------------------------------------------------------------------------
    # Python bool
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        bool
    ):

        return bool(value)

    # -------------------------------------------------------------------------
    # NumPy bool
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    # -------------------------------------------------------------------------
    # Python integer
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        int
    ):

        return int(value)

    # -------------------------------------------------------------------------
    # NumPy integer
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    # -------------------------------------------------------------------------
    # Python float
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        float
    ):

        if not np.isfinite(
            value
        ):

            return None

        return float(value)

    # -------------------------------------------------------------------------
    # NumPy floating
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        np.floating
    ):

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):

            return None

        return value

    # -------------------------------------------------------------------------
    # NumPy array
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        np.ndarray
    ):

        return [
            make_json_safe(
                item
            )

            for item in value.tolist()
        ]

    # -------------------------------------------------------------------------
    # Pandas Timestamp
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        pd.Timestamp
    ):

        return value.isoformat()

    # -------------------------------------------------------------------------
    # Pandas Series
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        pd.Series
    ):

        return {
            str(key):
                make_json_safe(item)

            for key, item
            in value.to_dict().items()
        }

    # -------------------------------------------------------------------------
    # Pandas DataFrame
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        pd.DataFrame
    ):

        return [
            make_json_safe(row)

            for row
            in value.to_dict(
                orient="records"
            )
        ]

    # -------------------------------------------------------------------------
    # Dictionary
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        return {

            str(key):
                make_json_safe(item)

            for key, item
            in value.items()
        }

    # -------------------------------------------------------------------------
    # List / tuple
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
        )
    ):

        return [

            make_json_safe(item)

            for item
            in value
        ]

    # -------------------------------------------------------------------------
    # Pandas NA
    # -------------------------------------------------------------------------

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    # -------------------------------------------------------------------------
    # String / other
    # -------------------------------------------------------------------------

    return str(value)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    scenario_results: pd.DataFrame,
    working: pd.DataFrame,
    metrics: dict,
    variable_impact: pd.DataFrame,
    validation: dict,
    real_statistics: dict,
) -> None:

    print_header(
        "SAVING BANKING RISK OUTPUTS"
    )

    # =========================================================================
    # 1. Scenario losses
    # =========================================================================

    scenario_path = (

        OUTPUT_DIR
        / "banking_risk_v1_scenario_losses.csv"
    )

    scenario_results.to_csv(
        scenario_path,
        index=False
    )

    # =========================================================================
    # 2. Variable impact
    # =========================================================================

    variable_path = (

        OUTPUT_DIR
        / "banking_risk_v1_variable_impact.csv"
    )

    variable_impact.to_csv(
        variable_path,
        index=False
    )

    # =========================================================================
    # 3. Monthly scenario details
    # =========================================================================

    monthly_path = (

        OUTPUT_DIR
        / "banking_risk_v1_monthly_scenario_details.csv"
    )

    detail_columns = [

        "_scenario_id",

        "_timestep",

        *FEATURES,

        "stress_score",

        "monthly_loss_rate",
    ]

    available_columns = [

        column

        for column in detail_columns

        if column in working.columns
    ]

    working[
        available_columns
    ].to_csv(
        monthly_path,
        index=False
    )

    # =========================================================================
    # 4. Summary CSV
    # =========================================================================

    summary_rows = []

    for key, value in metrics.items():

        safe_value = (
            make_json_safe(
                value
            )
        )

        summary_rows.append({

            "metric":
                str(key),

            "value":
                safe_value,
        })

    summary_path = (

        OUTPUT_DIR
        / "banking_risk_v1_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False
    )

    # =========================================================================
    # 5. Build JSON object
    # =========================================================================

    final_json = {

        "model_version":
            MODEL_VERSION,

        "timegan_version":
            TIMEGAN_VERSION,

        "input_real_dataset":
            str(
                REAL_DATA_PATH
            ),

        "input_synthetic_dataset":
            str(
                SYNTHETIC_DATA_PATH
            ),

        "metrics":
            make_json_safe(
                metrics
            ),

        "variable_impact":
            make_json_safe(
                variable_impact
            ),

        "validation":
            make_json_safe(
                validation
            ),

        "configuration":
            make_json_safe({

                "portfolio_value_inr":
                    PORTFOLIO_VALUE_INR,

                "var_confidence":
                    VAR_CONFIDENCE,

                "loss_scale":
                    LOSS_SCALE,

                "max_monthly_loss_rate":
                    MAX_MONTHLY_LOSS_RATE,

                "variable_weights":
                    VARIABLE_WEIGHTS,

                "adverse_direction":
                    ADVERSE_DIRECTION,

                "methodology_note":
                    (
                        "Portfolio value, variable weights and "
                        "loss scale are explicit modeling assumptions. "
                        "They are not claimed to represent a specific "
                        "bank's actual balance sheet."
                    ),
            }),

        "real_data_calibration":
            make_json_safe(
                real_statistics
            ),
    }

    # =========================================================================
    # 6. Save JSON
    # =========================================================================

    json_path = (

        OUTPUT_DIR
        / "banking_risk_v1_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_json,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    # =========================================================================
    # 7. Verify JSON actually exists
    # =========================================================================

    if not json_path.exists():

        raise RuntimeError(
            "JSON output was not created."
        )

    json_size = (
        json_path.stat().st_size
    )

    if json_size <= 0:

        raise RuntimeError(
            "JSON output file is empty."
        )

    # =========================================================================
    # 8. Validate JSON can be loaded
    # =========================================================================

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        json.load(
            file
        )

    # =========================================================================
    # 9. Print output locations
    # =========================================================================

    print(
        f"Scenario losses : "
        f"{scenario_path}"
    )

    print(
        f"Variable impact : "
        f"{variable_path}"
    )

    print(
        f"Monthly details : "
        f"{monthly_path}"
    )

    print(
        f"Summary         : "
        f"{summary_path}"
    )

    print(
        f"JSON report     : "
        f"{json_path}"
    )

    print(
        f"JSON size       : "
        f"{json_size:,} bytes"
    )

    print()

    print(
        "JSON validation : PASS"
    )

    print(
        "All Banking Risk Engine V1 outputs saved successfully."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print_header(
        "MACROSTRESS-GAN\n"
        "BANKING RISK ENGINE V1"
    )

    print(
        f"Project Root : "
        f"{ROOT}"
    )

    print(
        f"Model        : "
        f"{MODEL_VERSION}"
    )

    print(
        f"TimeGAN      : "
        f"{TIMEGAN_VERSION}"
    )

    print(
        f"Portfolio    : "
        f"₹{PORTFOLIO_VALUE_INR:,.2f}"
    )

    print(
        f"VaR Level    : "
        f"{VAR_CONFIDENCE * 100:.1f}%"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Portfolio value, sensitivity weights and "
        "loss scale are explicit modeling assumptions."
    )

    print(
        "They should be calibrated using institution-specific "
        "exposure data in a later version."
    )

    # =========================================================================
    # LOAD REAL DATA
    # =========================================================================

    real = load_real_data()

    # =========================================================================
    # LOAD V4 SCENARIOS
    # =========================================================================

    synthetic = load_synthetic_data()

    # =========================================================================
    # VALIDATE SCENARIO STRUCTURE
    # =========================================================================

    prepared = (
        prepare_scenario_structure(
            synthetic
        )
    )

    # =========================================================================
    # REAL DATA CALIBRATION
    # =========================================================================

    real_statistics = (
        calculate_real_statistics(
            real
        )
    )

    # =========================================================================
    # SCENARIO LOSS CALCULATION
    # =========================================================================

    (
        scenario_results,
        working,
    ) = calculate_scenario_metrics(

        prepared,

        real_statistics,
    )

    # =========================================================================
    # RISK METRICS
    # =========================================================================

    metrics = (
        calculate_risk_metrics(
            scenario_results
        )
    )

    # =========================================================================
    # VARIABLE IMPACT
    # =========================================================================

    variable_impact = (
        calculate_variable_impact(
            scenario_results
        )
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    validation = (
        validate_risk_output(
            scenario_results,
            metrics
        )
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_outputs(

        scenario_results,

        working,

        metrics,

        variable_impact,

        validation,

        real_statistics,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header(
        "BANKING RISK ENGINE V1 COMPLETE"
    )

    print(
        f"VaR 95%                 : "
        f"{metrics['var_95_pct']:.6f}"
    )

    print(
        f"VaR 95% INR             : ₹"
        f"{metrics['var_95_inr']:,.2f}"
    )

    print(
        f"ES 95%                  : "
        f"{metrics['expected_shortfall_95_pct']:.6f}"
    )

    print(
        f"ES 95% INR              : ₹"
        f"{metrics['expected_shortfall_95_inr']:,.2f}"
    )

    print(
        f"Expected Loss           : "
        f"{metrics['expected_loss_pct']:.6f}"
    )

    print(
        f"Expected Loss INR       : ₹"
        f"{metrics['expected_loss_inr']:,.2f}"
    )

    print(
        f"Maximum Drawdown        : "
        f"{metrics['maximum_drawdown_pct']:.6f}"
    )

    print(
        f"Maximum Drawdown INR    : ₹"
        f"{metrics['maximum_drawdown_inr']:,.2f}"
    )

    print(
        f"Worst Scenario Loss     : "
        f"{metrics['worst_scenario_loss_pct']:.6f}"
    )

    print(
        f"Worst Scenario Loss INR : ₹"
        f"{metrics['worst_scenario_loss_inr']:,.2f}"
    )

    print(
        f"Worst Scenario ID       : "
        f"{metrics['worst_scenario_id']}"
    )

    print()

    print(
        "STATUS                  : "
        f"{validation['overall_status']}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()