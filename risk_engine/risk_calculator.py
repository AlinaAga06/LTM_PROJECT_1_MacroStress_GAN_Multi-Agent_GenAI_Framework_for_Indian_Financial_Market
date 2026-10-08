from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# MACROSTRESS-GAN
# PRODUCTION RISK CALCULATOR
#
# Input:
#   Scenario paths / portfolio returns
#
# Output:
#   VaR 95%
#   Expected Shortfall 95%
#   Expected Loss
#   Maximum Drawdown
#   Worst Scenario Loss
#   Monetary Risk
#
# IMPORTANT:
#   This module does NOT read pre-calculated risk JSON files.
#   It calculates the risk metrics from supplied scenario data.
# =============================================================================


class RiskCalculationError(Exception):
    """Raised when risk calculation cannot be completed safely."""


class RiskCalculator:
    """
    Production-level portfolio risk calculator.

    The calculator expects scenario returns.

    Supported shapes:

        1D:
            [r1, r2, r3, ...]

        2D:
            [scenario, time]

        3D:
            [scenario, time, asset]

    If 3D data is supplied, portfolio weights are applied across assets.

    Returns are expected as decimal returns.

    Example:
        -0.05 = -5%
         0.02 = +2%
    """

    def __init__(
        self,
        confidence: float = 0.95,
        portfolio_value: float = 100_000_000.0,
    ):
        if not 0.5 < confidence < 1.0:
            raise ValueError(
                "confidence must be between 0.5 and 1.0"
            )

        if portfolio_value <= 0:
            raise ValueError(
                "portfolio_value must be positive"
            )

        self.confidence = float(confidence)

        self.portfolio_value = float(
            portfolio_value
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def calculate(
        self,
        scenario_returns: Any,
        weights: Optional[Any] = None,
        portfolio_value: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate complete risk profile.
        """

        if portfolio_value is None:
            portfolio_value = self.portfolio_value

        if confidence is None:
            confidence = self.confidence

        if portfolio_value <= 0:
            raise RiskCalculationError(
                "Portfolio value must be positive."
            )

        if not 0.5 < confidence < 1.0:
            raise RiskCalculationError(
                "Confidence must be between 0.5 and 1.0."
            )

        array = self._prepare_scenario_data(
            scenario_returns
        )

        portfolio_paths = self._convert_to_portfolio_paths(
            array,
            weights=weights,
        )

        scenario_returns_1d = (
            self._calculate_scenario_returns(
                portfolio_paths
            )
        )

        losses = self._returns_to_losses(
            scenario_returns_1d
        )

        var_loss = self._calculate_var(
            losses,
            confidence,
        )

        es_loss = self._calculate_expected_shortfall(
            losses,
            confidence,
        )

        expected_loss = float(
            np.mean(
                np.maximum(
                    losses,
                    0.0,
                )
            )
        )

        maximum_drawdown = self._calculate_maximum_drawdown(
            portfolio_paths
        )

        worst_loss = float(
            np.max(
                losses
            )
        )

        result = {

            "risk_metrics": {

                "var_95_pct":
                    self._ratio(
                        var_loss
                    ),

                "var_95_inr":
                    float(
                        var_loss
                        * portfolio_value
                    ),

                "expected_shortfall_95_pct":
                    self._ratio(
                        es_loss
                    ),

                "expected_shortfall_95_inr":
                    float(
                        es_loss
                        * portfolio_value
                    ),

                "expected_loss_pct":
                    self._ratio(
                        expected_loss
                    ),

                "expected_loss_inr":
                    float(
                        expected_loss
                        * portfolio_value
                    ),

                "maximum_drawdown_pct":
                    self._ratio(
                        maximum_drawdown
                    ),

                "maximum_drawdown_inr":
                    float(
                        maximum_drawdown
                        * portfolio_value
                    ),

                "worst_scenario_loss_pct":
                    self._ratio(
                        worst_loss
                    ),

                "worst_scenario_loss_inr":
                    float(
                        worst_loss
                        * portfolio_value
                    ),
            },

            "portfolio_value_inr":
                float(
                    portfolio_value
                ),

            "confidence_level":
                float(
                    confidence
                ),

            "scenario_count":
                int(
                    len(
                        scenario_returns_1d
                    )
                ),

            "sequence_length":
                int(
                    portfolio_paths.shape[1]
                ),

            "calculation_engine":
                "MacroStress-GAN Production Risk Engine",

            "calculation_method":
                "Historical empirical tail risk over generated stress scenarios",

        }

        return result

    # =========================================================================
    # INPUT PREPARATION
    # =========================================================================

    def _prepare_scenario_data(
        self,
        data: Any,
    ) -> np.ndarray:

        if isinstance(
            data,
            pd.DataFrame,
        ):

            data = data.select_dtypes(
                include=[np.number]
            ).values

        elif isinstance(
            data,
            pd.Series,
        ):

            data = data.values

        elif isinstance(
            data,
            list,
        ):

            data = np.asarray(
                data,
                dtype=float,
            )

        elif isinstance(
            data,
            np.ndarray,
        ):

            data = np.asarray(
                data,
                dtype=float,
            )

        else:

            raise RiskCalculationError(
                "Unsupported scenario data type."
            )

        if data.size == 0:

            raise RiskCalculationError(
                "Scenario data is empty."
            )

        if not np.isfinite(
            data
        ).all():

            raise RiskCalculationError(
                "Scenario data contains NaN or infinite values."
            )

        if data.ndim not in (
            1,
            2,
            3,
        ):

            raise RiskCalculationError(
                f"Unsupported scenario dimensions: {data.ndim}"
            )

        return data

    # =========================================================================
    # PORTFOLIO PATH CONSTRUCTION
    # =========================================================================

    def _convert_to_portfolio_paths(
        self,
        data: np.ndarray,
        weights: Optional[Any] = None,
    ) -> np.ndarray:

        # ---------------------------------------------------------------------
        # 1D
        # ---------------------------------------------------------------------

        if data.ndim == 1:

            return data.reshape(
                1,
                -1,
            )

        # ---------------------------------------------------------------------
        # 2D
        #
        # Ambiguous:
        #
        #   scenario x time
        #
        # This is interpreted as scenario paths.
        # ---------------------------------------------------------------------

        if data.ndim == 2:

            return data

        # ---------------------------------------------------------------------
        # 3D
        #
        # scenario x time x asset
        # ---------------------------------------------------------------------

        scenario_count = data.shape[0]

        time_steps = data.shape[1]

        asset_count = data.shape[2]

        if weights is None:

            weights_array = np.ones(
                asset_count,
                dtype=float,
            ) / asset_count

        else:

            weights_array = np.asarray(
                weights,
                dtype=float,
            )

            if len(
                weights_array
            ) != asset_count:

                raise RiskCalculationError(
                    "Number of weights does not match number of assets."
                )

            if np.any(
                weights_array < 0
            ):

                raise RiskCalculationError(
                    "Portfolio weights cannot be negative."
                )

            total = weights_array.sum()

            if total <= 0:

                raise RiskCalculationError(
                    "Portfolio weights must have positive sum."
                )

            weights_array = (
                weights_array / total
            )

        portfolio_paths = np.sum(
            data
            * weights_array.reshape(
                1,
                1,
                -1,
            ),
            axis=2,
        )

        return portfolio_paths

    # =========================================================================
    # RETURNS
    # =========================================================================

    def _calculate_scenario_returns(
        self,
        portfolio_paths: np.ndarray,
    ) -> np.ndarray:

        portfolio_paths = np.asarray(
            portfolio_paths,
            dtype=float,
        )

        if portfolio_paths.ndim != 2:

            raise RiskCalculationError(
                "Portfolio paths must be 2-dimensional."
            )

        # ---------------------------------------------------------------------
        # Detect whether input represents:
        #
        # A) returns
        # B) price/index paths
        #
        # If values are mostly inside a reasonable return range,
        # treat them as returns.
        # ---------------------------------------------------------------------

        max_abs = np.max(
            np.abs(
                portfolio_paths
            )
        )

        if max_abs <= 2.0:

            # Treat each row as a sequence of returns.
            cumulative = np.prod(
                1.0
                + portfolio_paths,
                axis=1,
            )

            scenario_return = (
                cumulative - 1.0
            )

            return scenario_return

        # Otherwise treat as price/index paths.

        initial = portfolio_paths[
            :,
            0,
        ]

        final = portfolio_paths[
            :,
            -1,
        ]

        safe_initial = np.where(
            np.abs(initial) < 1e-12,
            1e-12,
            initial,
        )

        scenario_return = (
            final
            / safe_initial
            - 1.0
        )

        return scenario_return

    # =========================================================================
    # LOSS
    # =========================================================================

    @staticmethod
    def _returns_to_losses(
        returns: np.ndarray,
    ) -> np.ndarray:

        returns = np.asarray(
            returns,
            dtype=float,
        )

        # Positive loss means portfolio lost money.
        losses = -returns

        return losses

    # =========================================================================
    # VAR
    # =========================================================================

    @staticmethod
    def _calculate_var(
        losses: np.ndarray,
        confidence: float,
    ) -> float:

        percentile = (
            confidence * 100.0
        )

        return float(
            np.percentile(
                losses,
                percentile,
            )
        )

    # =========================================================================
    # EXPECTED SHORTFALL
    # =========================================================================

    @staticmethod
    def _calculate_expected_shortfall(
        losses: np.ndarray,
        confidence: float,
    ) -> float:

        var = RiskCalculator._calculate_var(
            losses,
            confidence,
        )

        tail = losses[
            losses >= var
        ]

        if len(tail) == 0:

            return var

        return float(
            np.mean(
                tail
            )
        )

    # =========================================================================
    # MAXIMUM DRAWDOWN
    # =========================================================================

    @staticmethod
    def _calculate_maximum_drawdown(
        portfolio_paths: np.ndarray,
    ) -> float:

        maximum = 0.0

        for path in portfolio_paths:

            path = np.asarray(
                path,
                dtype=float,
            )

            if path.size == 0:
                continue

            # If this is return data, create wealth curve.
            if np.max(
                np.abs(path)
            ) <= 2.0:

                wealth = np.cumprod(
                    1.0 + path
                )

            else:

                wealth = path

            running_max = np.maximum.accumulate(
                wealth
            )

            safe_running_max = np.where(
                np.abs(running_max) < 1e-12,
                1e-12,
                running_max,
            )

            drawdown = (
                running_max
                - wealth
            ) / np.abs(
                safe_running_max
            )

            current = float(
                np.max(
                    drawdown
                )
            )

            maximum = max(
                maximum,
                current,
            )

        return float(
            maximum
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _ratio(
        value: float,
    ) -> float:

        if not math.isfinite(
            value
        ):

            return 0.0

        return float(
            max(
                value,
                0.0,
            )
        )


# =============================================================================
# FILE HELPERS
# =============================================================================

def calculate_and_save_risk(
    scenario_data,
    output_path,
    portfolio_value,
    weights=None,
    confidence=0.95,
):

    calculator = RiskCalculator(
        confidence=confidence,
        portfolio_value=portfolio_value,
    )

    result = calculator.calculate(
        scenario_returns=scenario_data,
        weights=weights,
        portfolio_value=portfolio_value,
        confidence=confidence,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    return result