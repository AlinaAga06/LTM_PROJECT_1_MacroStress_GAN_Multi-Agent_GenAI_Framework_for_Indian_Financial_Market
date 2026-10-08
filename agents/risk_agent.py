"""
MacroStress-GAN
Risk Agent - Phase 6

Institutional Financial Risk Engine
------------------------------------

Converts TimeGAN V3.4 financial feature paths into
portfolio-level risk metrics.

IMPORTANT:

The five TimeGAN features are NOT all portfolio returns.

Features:
    NIFTY50_Return
    CRUDE_OIL_Return
    USD_INR_Return
    INDIA_VIX_Change
    INDIA_10Y_YIELD_Change

Therefore, they must NOT simply be multiplied by positive
portfolio weights.

For the standardized institutional portfolio proxy:

    NIFTY50 Return       -> positive exposure
    Crude Oil Return     -> adverse when rising
    USD/INR Return       -> adverse when rising
    India VIX Change     -> adverse when rising
    India 10Y Yield      -> adverse when rising

The model converts the five market-risk drivers into a
signed portfolio stress-return proxy.

Monetary Risk:

    Portfolio Amount × Risk Percentage

Example:

    Portfolio = ₹10 crore
    Expected Loss = 0.003654

    Monetary Expected Loss
        = ₹10 crore × 0.003654
        = ₹3.654 lakh
"""


from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v2_scaler.pkl",
)

SYNTHETIC_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "timegan_v34_generated.npy",
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

FEATURE_DIM = 5

DEFAULT_CONFIDENCE = 0.95


# ============================================================
# INSTITUTIONAL PORTFOLIO RISK EXPOSURES
# ============================================================

"""
These are NOT simple portfolio-return weights.

They represent the relative contribution of each
market-risk driver to the standardized institutional
portfolio stress proxy.

NIFTY:
    Positive exposure.
    NIFTY fall -> negative portfolio return.

CRUDE:
    Negative exposure.
    Crude increase -> negative portfolio return.

USD/INR:
    Negative exposure.
    USD/INR increase -> INR depreciation -> negative return.

VIX:
    Negative exposure.
    VIX increase -> market stress -> negative return.

10Y Yield:
    Negative exposure.
    Yield increase -> valuation/rate pressure -> negative return.
"""

PORTFOLIO_EXPOSURES = np.array(
    [
        0.60,    # NIFTY50
        -0.10,   # CRUDE OIL
        -0.10,   # USD/INR
        -0.10,   # INDIA VIX
        -0.10,   # INDIA 10Y YIELD
    ],
    dtype=np.float64,
)


# ============================================================
# FEATURE SCALING FACTORS
# ============================================================

"""
The first three features are returns.

The last two are changes rather than returns.

Therefore VIX and Yield changes are scaled before being
converted into the portfolio-risk proxy.

These values intentionally remain conservative.

They prevent a raw +1.0 yield feature movement from being
interpreted as a literal +100% portfolio return.
"""

FEATURE_IMPACT_SCALE = np.array(
    [
        1.00,   # NIFTY Return
        1.00,   # Crude Return
        1.00,   # USD/INR Return
        0.10,   # VIX Change
        0.05,   # 10Y Yield Change
    ],
    dtype=np.float64,
)


# ============================================================
# RISK AGENT
# ============================================================

class RiskAgent:

    def __init__(
        self,
        scaler_path: str = SCALER_PATH,
        confidence_level: float = DEFAULT_CONFIDENCE,
    ):

        self.scaler_path = scaler_path

        self.confidence_level = float(
            confidence_level
        )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        self.scaler = None

        self.status = "initialized"

        self.error = None

        self._load_scaler()


    # ========================================================
    # LOAD SCALER
    # ========================================================

    def _load_scaler(self):

        if not os.path.exists(
            self.scaler_path
        ):

            self.status = "error"

            self.error = (
                f"Scaler not found: "
                f"{self.scaler_path}"
            )

            return

        try:

            self.scaler = joblib.load(
                self.scaler_path
            )

            if not hasattr(
                self.scaler,
                "inverse_transform",
            ):

                raise TypeError(
                    "Loaded scaler does not support "
                    "inverse_transform()."
                )

            if not hasattr(
                self.scaler,
                "transform",
            ):

                raise TypeError(
                    "Loaded scaler does not support "
                    "transform()."
                )

            self.status = "ready"

        except Exception as exc:

            self.status = "error"

            self.error = (
                f"Failed to load scaler: {exc}"
            )


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_input(
        self,
        sequences,
        allow_outside_scaled_range=False,
    ):

        if sequences is None:

            raise ValueError(
                "Generated sequences are None."
            )

        array = np.asarray(
            sequences,
            dtype=np.float64,
        )

        if array.ndim != 3:

            raise ValueError(
                "Expected 3D array "
                "(scenarios, horizon, features), "
                f"got {array.ndim}D."
            )

        if array.shape[2] != FEATURE_DIM:

            raise ValueError(
                f"Expected {FEATURE_DIM} features, "
                f"got {array.shape[2]}."
            )

        if array.shape[0] < 1:

            raise ValueError(
                "No scenarios supplied."
            )

        if array.shape[1] < 1:

            raise ValueError(
                "No time steps supplied."
            )

        if not np.isfinite(array).all():

            raise ValueError(
                "Input contains NaN or infinite values."
            )

        if not allow_outside_scaled_range:

            if (
                array.min() < -1e-5
                or array.max() > 1.00001
            ):

                raise ValueError(
                    "Input appears to be outside "
                    "the expected [0,1] TimeGAN range. "
                    f"Min={array.min():.6f}, "
                    f"Max={array.max():.6f}"
                )

        return array


    # ========================================================
    # INVERSE TRANSFORM
    # ========================================================

    def inverse_transform(
        self,
        sequences,
    ):

        array = self.validate_input(
            sequences
        )

        original_shape = array.shape

        flattened = array.reshape(
            -1,
            FEATURE_DIM,
        )

        restored = self.scaler.inverse_transform(
            flattened
        )

        restored = np.asarray(
            restored,
            dtype=np.float64,
        ).reshape(
            original_shape
        )

        if not np.isfinite(
            restored
        ).all():

            raise ValueError(
                "Inverse-transformed data contains "
                "NaN or infinite values."
            )

        return restored


    # ========================================================
    # PORTFOLIO RETURNS
    # ========================================================

    def calculate_portfolio_returns(
        self,
        restored_sequences,
    ):
        """
        Convert the five market-risk drivers into a
        signed portfolio stress-return proxy.

        Shape:

            input:
                (scenarios, horizon, 5)

            output:
                (scenarios, horizon)

        The important difference from the old implementation
        is that VIX and Yield are adverse when they increase.
        """

        restored_sequences = np.asarray(
            restored_sequences,
            dtype=np.float64,
        )

        if restored_sequences.ndim != 3:

            raise ValueError(
                "Expected restored sequences to be 3D."
            )

        if restored_sequences.shape[2] != FEATURE_DIM:

            raise ValueError(
                f"Expected {FEATURE_DIM} features."
            )

        if not np.isfinite(
            restored_sequences
        ).all():

            raise ValueError(
                "Restored sequences contain "
                "NaN or infinite values."
            )

        # ----------------------------------------------------
        # APPLY FEATURE-SPECIFIC IMPACT SCALE
        # ----------------------------------------------------

        adjusted_features = (
            restored_sequences
            * FEATURE_IMPACT_SCALE
        )

        # ----------------------------------------------------
        # SIGNED PORTFOLIO EXPOSURE
        # ----------------------------------------------------

        portfolio_returns = np.sum(
            adjusted_features
            * PORTFOLIO_EXPOSURES,
            axis=2,
        )

        portfolio_returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        )

        if not np.isfinite(
            portfolio_returns
        ).all():

            raise ValueError(
                "Portfolio returns contain "
                "NaN or infinite values."
            )

        return portfolio_returns


    # ========================================================
    # VAR
    # ========================================================

    def calculate_var(
        self,
        portfolio_returns,
        confidence_level=None,
    ):

        if confidence_level is None:

            confidence_level = (
                self.confidence_level
            )

        returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        ).reshape(-1)

        losses = -returns

        return float(
            np.quantile(
                losses,
                confidence_level,
            )
        )


    # ========================================================
    # EXPECTED SHORTFALL
    # ========================================================

    def calculate_expected_shortfall(
        self,
        portfolio_returns,
        confidence_level=None,
    ):

        if confidence_level is None:

            confidence_level = (
                self.confidence_level
            )

        returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        ).reshape(-1)

        losses = -returns

        var = np.quantile(
            losses,
            confidence_level,
        )

        tail_losses = losses[
            losses >= var
        ]

        if len(tail_losses) == 0:

            return float(var)

        return float(
            np.mean(
                tail_losses
            )
        )


    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    def calculate_max_drawdown(
        self,
        portfolio_returns,
    ):

        returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        )

        if returns.ndim != 2:

            raise ValueError(
                "Expected portfolio returns with shape "
                "(scenarios, horizon)."
            )

        scenario_drawdowns = []

        for scenario in returns:

            # ------------------------------------------------
            # SAFETY
            # ------------------------------------------------

            if np.any(
                1.0 + scenario <= 0.0
            ):

                scenario_drawdowns.append(
                    1.0
                )

                continue

            # ------------------------------------------------
            # WEALTH PATH
            # ------------------------------------------------

            wealth = np.cumprod(
                1.0 + scenario
            )

            running_peak = np.maximum.accumulate(
                wealth
            )

            drawdown = (
                wealth - running_peak
            ) / running_peak

            scenario_drawdowns.append(
                abs(
                    np.min(
                        drawdown
                    )
                )
            )

        if not scenario_drawdowns:

            return 0.0

        return float(
            np.max(
                scenario_drawdowns
            )
        )


    # ========================================================
    # EXPECTED LOSS
    # ========================================================

    def calculate_expected_loss(
        self,
        portfolio_returns,
    ):

        returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        ).reshape(-1)

        losses = -returns

        negative_losses = losses[
            losses > 0
        ]

        if len(negative_losses) == 0:

            return 0.0

        return float(
            np.mean(
                negative_losses
            )
        )


    # ========================================================
    # WORST SCENARIO
    # ========================================================

    def calculate_worst_scenario(
        self,
        portfolio_returns,
    ):

        returns = np.asarray(
            portfolio_returns,
            dtype=np.float64,
        )

        if returns.ndim != 2:

            raise ValueError(
                "Expected portfolio returns with shape "
                "(scenarios, horizon)."
            )

        scenario_returns = []

        for scenario in returns:

            if np.any(
                1.0 + scenario <= 0.0
            ):

                cumulative_return = -1.0

            else:

                cumulative_return = (
                    np.prod(
                        1.0 + scenario
                    )
                    - 1.0
                )

            scenario_returns.append(
                cumulative_return
            )

        scenario_returns = np.asarray(
            scenario_returns,
            dtype=np.float64,
        )

        worst_index = int(
            np.argmin(
                scenario_returns
            )
        )

        worst_return = float(
            scenario_returns[
                worst_index
            ]
        )

        return {
            "scenario_index": worst_index,

            "cumulative_return": (
                worst_return
            ),

            "loss_magnitude": max(
                0.0,
                -worst_return,
            ),
        }


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    def classify_risk(
        self,
        var,
        expected_shortfall,
        max_drawdown,
    ):

        var_abs = abs(
            float(var)
        )

        es_abs = abs(
            float(expected_shortfall)
        )

        mdd_abs = abs(
            float(max_drawdown)
        )

        if (
            var_abs >= 0.10
            or es_abs >= 0.15
            or mdd_abs >= 0.25
        ):

            return "High"

        if (
            var_abs >= 0.05
            or es_abs >= 0.08
            or mdd_abs >= 0.15
        ):

            return "Moderate"

        return "Low"


    # ========================================================
    # SUMMARY
    # ========================================================

    def calculate_summary(
        self,
        restored_sequences,
        portfolio_returns,
    ):

        flattened = restored_sequences.reshape(
            -1,
            FEATURE_DIM,
        )

        feature_summary = {}

        for i, feature_name in enumerate(
            FEATURE_NAMES
        ):

            values = flattened[:, i]

            feature_summary[
                feature_name
            ] = {

                "mean": float(
                    np.mean(values)
                ),

                "std": float(
                    np.std(values)
                ),

                "min": float(
                    np.min(values)
                ),

                "max": float(
                    np.max(values)
                ),

                "p05": float(
                    np.quantile(
                        values,
                        0.05,
                    )
                ),

                "p50": float(
                    np.quantile(
                        values,
                        0.50,
                    )
                ),

                "p95": float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
            }

        portfolio_flat = np.asarray(
            portfolio_returns
        ).reshape(-1)

        return {

            "features": feature_summary,

            "portfolio": {

                "mean": float(
                    np.mean(
                        portfolio_flat
                    )
                ),

                "std": float(
                    np.std(
                        portfolio_flat
                    )
                ),

                "min": float(
                    np.min(
                        portfolio_flat
                    )
                ),

                "max": float(
                    np.max(
                        portfolio_flat
                    )
                ),
            },

            "portfolio_exposure_model": {
                feature: float(
                    PORTFOLIO_EXPOSURES[i]
                )
                for i, feature in enumerate(
                    FEATURE_NAMES
                )
            },

            "feature_impact_scale": {
                feature: float(
                    FEATURE_IMPACT_SCALE[i]
                )
                for i, feature in enumerate(
                    FEATURE_NAMES
                )
            },
        }


    # ========================================================
    # METRIC BUNDLE
    # ========================================================

    def _calculate_metric_bundle(
        self,
        restored_sequences,
    ):

        portfolio_returns = (
            self.calculate_portfolio_returns(
                restored_sequences
            )
        )

        var = self.calculate_var(
            portfolio_returns
        )

        es = self.calculate_expected_shortfall(
            portfolio_returns
        )

        mdd = self.calculate_max_drawdown(
            portfolio_returns
        )

        expected_loss = (
            self.calculate_expected_loss(
                portfolio_returns
            )
        )

        worst = (
            self.calculate_worst_scenario(
                portfolio_returns
            )
        )

        risk_level = self.classify_risk(
            var,
            es,
            mdd,
        )

        summary = self.calculate_summary(
            restored_sequences,
            portfolio_returns,
        )

        return {

            "VaR_95": var,

            "Expected_Shortfall_95": es,

            "Maximum_Drawdown": mdd,

            "Expected_Loss": expected_loss,

            "risk_level": risk_level,

            "worst_scenario": worst,

            "summary": summary,

            "portfolio_returns":
                portfolio_returns,
        }


    # ========================================================
    # STRESS IMPACT
    # ========================================================

    def calculate_stress_impact(
        self,
        baseline_metrics,
        stressed_metrics,
    ):

        metric_names = [
            "VaR_95",
            "Expected_Shortfall_95",
            "Maximum_Drawdown",
            "Expected_Loss",
        ]

        result = {}

        for name in metric_names:

            baseline = float(
                baseline_metrics[name]
            )

            stressed = float(
                stressed_metrics[name]
            )

            delta = (
                stressed
                - baseline
            )

            if abs(
                baseline
            ) > 1e-12:

                change_pct = (
                    delta
                    / abs(baseline)
                ) * 100.0

            else:

                change_pct = None

            result[name] = {

                "baseline":
                    baseline,

                "stressed":
                    stressed,

                "delta":
                    delta,

                "change_pct":
                    change_pct,
            }

        return result


    # ========================================================
    # FEATURE IMPACT
    # ========================================================

    def calculate_feature_impact(
        self,
        baseline_restored,
        stressed_restored,
    ):

        baseline_flat = (
            baseline_restored.reshape(
                -1,
                FEATURE_DIM,
            )
        )

        stressed_flat = (
            stressed_restored.reshape(
                -1,
                FEATURE_DIM,
            )
        )

        result = {}

        for i, feature_name in enumerate(
            FEATURE_NAMES
        ):

            baseline_mean = float(
                np.mean(
                    baseline_flat[:, i]
                )
            )

            stressed_mean = float(
                np.mean(
                    stressed_flat[:, i]
                )
            )

            baseline_std = float(
                np.std(
                    baseline_flat[:, i]
                )
            )

            stressed_std = float(
                np.std(
                    stressed_flat[:, i]
                )
            )

            result[feature_name] = {

                "baseline_mean":
                    baseline_mean,

                "stressed_mean":
                    stressed_mean,

                "mean_delta":
                    stressed_mean
                    - baseline_mean,

                "baseline_std":
                    baseline_std,

                "stressed_std":
                    stressed_std,

                "std_delta":
                    stressed_std
                    - baseline_std,

                "portfolio_exposure":
                    float(
                        PORTFOLIO_EXPOSURES[i]
                    ),

                "impact_scale":
                    float(
                        FEATURE_IMPACT_SCALE[i]
                    ),
            }

        return result


    # ========================================================
    # MONETARY RISK
    # ========================================================

    def calculate_monetary_risk(
        self,
        metrics,
        portfolio_amount,
    ):

        portfolio_amount = float(
            portfolio_amount
        )

        if portfolio_amount <= 0:

            raise ValueError(
                "Portfolio amount must be greater than zero."
            )

        return {

            "portfolio_amount_inr":
                portfolio_amount,

            "VaR_95_inr":
                portfolio_amount
                * max(
                    0.0,
                    float(
                        metrics["VaR_95"]
                    ),
                ),

            "Expected_Shortfall_95_inr":
                portfolio_amount
                * max(
                    0.0,
                    float(
                        metrics[
                            "Expected_Shortfall_95"
                        ]
                    ),
                ),

            "Maximum_Drawdown_inr":
                portfolio_amount
                * max(
                    0.0,
                    float(
                        metrics[
                            "Maximum_Drawdown"
                        ]
                    ),
                ),

            "Expected_Loss_inr":
                portfolio_amount
                * max(
                    0.0,
                    float(
                        metrics[
                            "Expected_Loss"
                        ]
                    ),
                ),

            "Worst_Scenario_Loss_inr":
                portfolio_amount
                * max(
                    0.0,
                    float(
                        metrics[
                            "worst_scenario"
                        ][
                            "loss_magnitude"
                        ]
                    ),
                ),
        }


    # ========================================================
    # SCENARIO ANALYSIS
    # ========================================================

    def analyze_scenario(
        self,
        baseline_sequences,
        stressed_sequences,
        scenario_name="Stress Scenario",
        portfolio_amount=None,
    ):

        if self.status == "error":

            return {
                "status": "error",
                "agent": "Risk Agent",
                "error": self.error,
            }

        try:

            baseline_scaled = (
                self.validate_input(
                    baseline_sequences,
                    allow_outside_scaled_range=True,
                )
            )

            stressed_scaled = (
                self.validate_input(
                    stressed_sequences,
                    allow_outside_scaled_range=True,
                )
            )

            if (
                baseline_scaled.shape
                != stressed_scaled.shape
            ):

                raise ValueError(
                    "Baseline and stressed sequence "
                    "shapes must match. "
                    f"Baseline={baseline_scaled.shape}, "
                    f"Stressed={stressed_scaled.shape}"
                )

            # ------------------------------------------------
            # INVERSE TRANSFORM
            # ------------------------------------------------

            baseline_restored = (
                self.inverse_transform(
                    baseline_scaled
                )
            )

            stressed_restored = (
                self.inverse_transform(
                    stressed_scaled
                )
            )

            # ------------------------------------------------
            # CALCULATE BASELINE
            # ------------------------------------------------

            baseline_metrics = (
                self._calculate_metric_bundle(
                    baseline_restored
                )
            )

            # ------------------------------------------------
            # CALCULATE STRESSED
            # ------------------------------------------------

            stressed_metrics = (
                self._calculate_metric_bundle(
                    stressed_restored
                )
            )

            # ------------------------------------------------
            # STRESS IMPACT
            # ------------------------------------------------

            stress_impact = (
                self.calculate_stress_impact(
                    baseline_metrics,
                    stressed_metrics,
                )
            )

            # ------------------------------------------------
            # FEATURE IMPACT
            # ------------------------------------------------

            feature_impact = (
                self.calculate_feature_impact(
                    baseline_restored,
                    stressed_restored,
                )
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = {

                "status":
                    "success",

                "agent":
                    "Risk Agent",

                "phase":
                    "Phase 6",

                "analysis_type":
                    "Baseline vs Scenario Stress",

                "scenario":
                    scenario_name,

                "num_scenarios":
                    int(
                        baseline_scaled.shape[0]
                    ),

                "horizon_days":
                    int(
                        baseline_scaled.shape[1]
                    ),

                "feature_count":
                    FEATURE_DIM,

                "features":
                    FEATURE_NAMES,

                "scaling": {

                    "input_space":
                        "TimeGAN V3.4 scaled",

                    "inverse_scaler":
                        "timegan_v2_scaler.pkl",

                    "scaler_type":
                        type(
                            self.scaler
                        ).__name__,

                    "output_space":
                        "Original financial "
                        "feature space",
                },

                "portfolio_model": {

                    "type":
                        "Directional institutional "
                        "risk proxy",

                    "description":
                        "Market features are converted "
                        "to signed portfolio stress "
                        "returns instead of treating "
                        "VIX and yield changes as "
                        "positive asset returns.",

                    "exposures": {
                        feature: float(
                            PORTFOLIO_EXPOSURES[i]
                        )
                        for i, feature in enumerate(
                            FEATURE_NAMES
                        )
                    },

                    "impact_scale": {
                        feature: float(
                            FEATURE_IMPACT_SCALE[i]
                        )
                        for i, feature in enumerate(
                            FEATURE_NAMES
                        )
                    },
                },

                "baseline": {

                    "risk_metrics": {

                        "VaR_95":
                            baseline_metrics[
                                "VaR_95"
                            ],

                        "Expected_Shortfall_95":
                            baseline_metrics[
                                "Expected_Shortfall_95"
                            ],

                        "Maximum_Drawdown":
                            baseline_metrics[
                                "Maximum_Drawdown"
                            ],

                        "Expected_Loss":
                            baseline_metrics[
                                "Expected_Loss"
                            ],
                    },

                    "risk_level":
                        baseline_metrics[
                            "risk_level"
                        ],

                    "worst_scenario":
                        baseline_metrics[
                            "worst_scenario"
                        ],

                    "summary":
                        baseline_metrics[
                            "summary"
                        ],
                },

                "stressed": {

                    "risk_metrics": {

                        "VaR_95":
                            stressed_metrics[
                                "VaR_95"
                            ],

                        "Expected_Shortfall_95":
                            stressed_metrics[
                                "Expected_Shortfall_95"
                            ],

                        "Maximum_Drawdown":
                            stressed_metrics[
                                "Maximum_Drawdown"
                            ],

                        "Expected_Loss":
                            stressed_metrics[
                                "Expected_Loss"
                            ],
                    },

                    "risk_level":
                        stressed_metrics[
                            "risk_level"
                        ],

                    "worst_scenario":
                        stressed_metrics[
                            "worst_scenario"
                        ],

                    "summary":
                        stressed_metrics[
                            "summary"
                        ],
                },

                "stress_impact":
                    stress_impact,

                "feature_impact":
                    feature_impact,

                "risk_change": {

                    "VaR_delta":
                        stress_impact[
                            "VaR_95"
                        ]["delta"],

                    "Expected_Shortfall_delta":
                        stress_impact[
                            "Expected_Shortfall_95"
                        ]["delta"],

                    "Maximum_Drawdown_delta":
                        stress_impact[
                            "Maximum_Drawdown"
                        ]["delta"],

                    "Expected_Loss_delta":
                        stress_impact[
                            "Expected_Loss"
                        ]["delta"],
                },

                "portfolio": {

                    "amount_inr":
                        portfolio_amount,

                    "currency":
                        "INR",
                },

                "note":
                    "Positive deltas in VaR, "
                    "Expected Shortfall, Maximum "
                    "Drawdown, or Expected Loss "
                    "indicate increased loss "
                    "magnitude under the supplied "
                    "stress scenario.",
            }

            # =================================================
            # MONETARY RISK
            # =================================================

            if portfolio_amount is not None:

                baseline_monetary = (
                    self.calculate_monetary_risk(
                        baseline_metrics,
                        portfolio_amount,
                    )
                )

                stressed_monetary = (
                    self.calculate_monetary_risk(
                        stressed_metrics,
                        portfolio_amount,
                    )
                )

                result[
                    "monetary_risk"
                ] = {

                    "currency":
                        "INR",

                    "portfolio_amount_inr":
                        float(
                            portfolio_amount
                        ),

                    "baseline":
                        baseline_monetary,

                    "stressed":
                        stressed_monetary,

                    "stress_impact": {

                        "VaR_95_inr":
                            stressed_monetary[
                                "VaR_95_inr"
                            ]
                            - baseline_monetary[
                                "VaR_95_inr"
                            ],

                        "Expected_Shortfall_95_inr":
                            stressed_monetary[
                                "Expected_Shortfall_95_inr"
                            ]
                            - baseline_monetary[
                                "Expected_Shortfall_95_inr"
                            ],

                        "Maximum_Drawdown_inr":
                            stressed_monetary[
                                "Maximum_Drawdown_inr"
                            ]
                            - baseline_monetary[
                                "Maximum_Drawdown_inr"
                            ],

                        "Expected_Loss_inr":
                            stressed_monetary[
                                "Expected_Loss_inr"
                            ]
                            - baseline_monetary[
                                "Expected_Loss_inr"
                            ],

                        "Worst_Scenario_Loss_inr":
                            stressed_monetary[
                                "Worst_Scenario_Loss_inr"
                            ]
                            - baseline_monetary[
                                "Worst_Scenario_Loss_inr"
                            ],
                    },
                }

            return result

        except Exception as exc:

            return {

                "status":
                    "error",

                "agent":
                    "Risk Agent",

                "error":
                    str(exc),
            }


    # ========================================================
    # PUBLIC RUN
    # ========================================================

    def run(
        self,
        sequences=None,
        scenario_name="Generated Stress Scenario",
        baseline_sequences=None,
        stressed_sequences=None,
        portfolio_amount=None,
    ):

        if (
            baseline_sequences is not None
            and stressed_sequences is not None
        ):

            return self.analyze_scenario(

                baseline_sequences=
                    baseline_sequences,

                stressed_sequences=
                    stressed_sequences,

                scenario_name=
                    scenario_name,

                portfolio_amount=
                    portfolio_amount,
            )

        if sequences is None:

            return {

                "status":
                    "error",

                "agent":
                    "Risk Agent",

                "error":
                    "No sequences supplied.",
            }

        if self.status == "error":

            return {

                "status":
                    "error",

                "agent":
                    "Risk Agent",

                "error":
                    self.error,
            }

        try:

            scaled = self.validate_input(
                sequences,
                allow_outside_scaled_range=True,
            )

            restored = self.inverse_transform(
                scaled
            )

            metrics = (
                self._calculate_metric_bundle(
                    restored
                )
            )

            result = {

                "status":
                    "success",

                "agent":
                    "Risk Agent",

                "phase":
                    "Phase 6",

                "scenario":
                    scenario_name,

                "num_scenarios":
                    int(
                        scaled.shape[0]
                    ),

                "horizon_days":
                    int(
                        scaled.shape[1]
                    ),

                "feature_count":
                    FEATURE_DIM,

                "features":
                    FEATURE_NAMES,

                "risk_metrics": {

                    "VaR_95":
                        metrics[
                            "VaR_95"
                        ],

                    "Expected_Shortfall_95":
                        metrics[
                            "Expected_Shortfall_95"
                        ],

                    "Maximum_Drawdown":
                        metrics[
                            "Maximum_Drawdown"
                        ],

                    "Expected_Loss":
                        metrics[
                            "Expected_Loss"
                        ],
                },

                "risk_level":
                    metrics[
                        "risk_level"
                    ],

                "worst_scenario":
                    metrics[
                        "worst_scenario"
                    ],

                "summary":
                    metrics[
                        "summary"
                    ],
            }

            if portfolio_amount is not None:

                result[
                    "monetary_risk"
                ] = self.calculate_monetary_risk(
                    metrics,
                    portfolio_amount,
                )

            return result

        except Exception as exc:

            return {

                "status":
                    "error",

                "agent":
                    "Risk Agent",

                "error":
                    str(exc),
            }


# ============================================================
# STANDALONE TEST
# ============================================================

def load_generated_sequences():

    if not os.path.exists(
        SYNTHETIC_OUTPUT_PATH
    ):

        raise FileNotFoundError(
            f"Generated file not found: "
            f"{SYNTHETIC_OUTPUT_PATH}"
        )

    return np.load(
        SYNTHETIC_OUTPUT_PATH
    )


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 78)

    print(
        "MACROSTRESS-GAN - RISK AGENT"
    )

    print("=" * 78)

    agent = RiskAgent()

    print(
        f"Scaler status: {agent.status}"
    )

    if agent.status == "error":

        print(
            agent.error
        )

        sys.exit(1)

    baseline = (
        load_generated_sequences()
    )

    stressed = (
        baseline.copy()
    )

    result = agent.run(

        baseline_sequences=
            baseline,

        stressed_sequences=
            stressed,

        scenario_name=
            "Test Scenario",

        portfolio_amount=
            100000000,
    )

    print(
        "\nRisk Result:"
    )

    print(
        result
    )


if __name__ == "__main__":

    main()