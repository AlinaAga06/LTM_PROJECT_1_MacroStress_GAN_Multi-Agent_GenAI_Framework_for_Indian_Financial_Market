"""
MacroStress-GAN
MONETARY RISK CALCULATOR
============================================================

Purpose
-------
Converts percentage/decimal portfolio risk metrics produced
by the Risk Agent into monetary exposure.

Compatible with RiskAgent Phase 6 output:

baseline
    └── risk_metrics
         ├── VaR_95
         ├── Expected_Shortfall_95
         ├── Maximum_Drawdown
         └── Expected_Loss

baseline
    └── worst_scenario
         └── loss_magnitude

Example:

Portfolio = ₹10 crore
Expected Loss = 0.003654

Monetary Expected Loss:

₹10,00,00,000 × 0.003654
= ₹3,65,400

IMPORTANT
---------
This module does NOT modify:

- TimeGAN
- Simulation Agent
- Risk Agent calculations
- VaR methodology
- Expected Shortfall methodology
- Maximum Drawdown methodology

It only converts existing risk metrics
into monetary values.
"""

from __future__ import annotations

from typing import Dict, Any


# ============================================================
# MONETARY RISK CALCULATOR
# ============================================================

class MonetaryRiskCalculator:

    def __init__(
        self,
        portfolio_amount: float,
        currency: str = "INR",
    ):

        try:
            self.portfolio_amount = float(
                portfolio_amount
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "portfolio_amount must be numeric."
            )

        if self.portfolio_amount <= 0:

            raise ValueError(
                "portfolio_amount must be greater than zero."
            )

        self.currency = (
            str(currency).upper()
            if currency
            else "INR"
        )


    # ========================================================
    # FORMAT INDIAN RUPEE
    # ========================================================

    @staticmethod
    def format_inr(
        amount: float,
    ) -> str:

        amount = float(amount)

        sign = "-" if amount < 0 else ""

        amount = abs(amount)

        integer_part = int(amount)

        decimal_part = round(
            amount - integer_part,
            2,
        )

        number = str(integer_part)

        if len(number) > 3:

            last_three = number[-3:]

            remaining = number[:-3]

            groups = []

            while len(remaining) > 2:

                groups.insert(
                    0,
                    remaining[-2:],
                )

                remaining = remaining[:-2]

            if remaining:

                groups.insert(
                    0,
                    remaining,
                )

            formatted_integer = (
                ",".join(groups)
                + ","
                + last_three
            )

        else:

            formatted_integer = number

        if decimal_part > 0:

            decimal_string = (
                f"{decimal_part:.2f}"
                .split(".")[1]
            )

            return (
                f"{sign}₹"
                f"{formatted_integer}."
                f"{decimal_string}"
            )

        return (
            f"{sign}₹"
            f"{formatted_integer}"
        )


    # ========================================================
    # GENERIC CURRENCY FORMAT
    # ========================================================

    def format_currency(
        self,
        amount: float,
    ) -> str:

        if self.currency == "INR":

            return self.format_inr(
                amount
            )

        return (
            f"{self.currency} "
            f"{amount:,.2f}"
        )


    # ========================================================
    # CRORE
    # ========================================================

    @staticmethod
    def to_crore(
        amount: float,
    ) -> float:

        return float(amount) / 10_000_000


    # ========================================================
    # LAKH
    # ========================================================

    @staticmethod
    def to_lakh(
        amount: float,
    ) -> float:

        return float(amount) / 100_000


    # ========================================================
    # HUMAN READABLE SCALE
    # ========================================================

    def format_indian_scale(
        self,
        amount: float,
    ) -> str:

        amount = float(amount)

        absolute_amount = abs(
            amount
        )

        sign = "-" if amount < 0 else ""

        if absolute_amount >= 10_000_000:

            crore = (
                absolute_amount
                / 10_000_000
            )

            return (
                f"{sign}{self.currency} "
                f"{crore:,.2f} crore"
            )

        if absolute_amount >= 100_000:

            lakh = (
                absolute_amount
                / 100_000
            )

            return (
                f"{sign}{self.currency} "
                f"{lakh:,.2f} lakh"
            )

        if absolute_amount >= 1_000:

            thousand = (
                absolute_amount
                / 1_000
            )

            return (
                f"{sign}{self.currency} "
                f"{thousand:,.2f} thousand"
            )

        return self.format_currency(
            amount
        )


    # ========================================================
    # GET METRIC
    # ========================================================

    @staticmethod
    def _get_metric(
        metrics: Dict[str, Any],
        possible_names,
        default=0.0,
    ) -> float:

        if not isinstance(
            metrics,
            dict,
        ):

            return float(default)

        # ----------------------------------------------------
        # 1. Direct / flat format
        # ----------------------------------------------------

        for name in possible_names:

            if name in metrics:

                value = metrics[name]

                if value is None:

                    continue

                try:

                    return float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

        # ----------------------------------------------------
        # 2. Current RiskAgent nested format
        #
        # baseline
        #    └── risk_metrics
        # ----------------------------------------------------

        nested_metrics = metrics.get(
            "risk_metrics"
        )

        if isinstance(
            nested_metrics,
            dict,
        ):

            for name in possible_names:

                if name in nested_metrics:

                    value = nested_metrics[name]

                    if value is None:

                        continue

                    try:

                        return float(
                            value
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

        return float(default)


    # ========================================================
    # GET WORST SCENARIO LOSS
    # ========================================================

    @staticmethod
    def _get_worst_scenario_loss(
        metrics: Dict[str, Any],
    ) -> float:

        if not isinstance(
            metrics,
            dict,
        ):

            return 0.0

        # ----------------------------------------------------
        # Flat format
        # ----------------------------------------------------

        direct_names = [
            "Worst_Scenario_Loss",
            "Worst_Loss",
            "worst_scenario_loss",
        ]

        for name in direct_names:

            if name in metrics:

                try:

                    return float(
                        metrics[name]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        # ----------------------------------------------------
        # RiskAgent format
        #
        # worst_scenario:
        #     cumulative_return
        #     loss_magnitude
        # ----------------------------------------------------

        worst_scenario = metrics.get(
            "worst_scenario"
        )

        if isinstance(
            worst_scenario,
            dict,
        ):

            if (
                "loss_magnitude"
                in worst_scenario
            ):

                try:

                    return float(
                        worst_scenario[
                            "loss_magnitude"
                        ]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        return 0.0


    # ========================================================
    # NORMALIZE RISK VALUE
    # ========================================================

    @staticmethod
    def normalize_risk_value(
        value: float,
    ) -> float:

        value = float(value)

        # ----------------------------------------------------
        # Decimal format
        #
        # 0.05 -> 0.05
        # ----------------------------------------------------

        if abs(value) <= 1:

            return value

        # ----------------------------------------------------
        # Percentage format
        #
        # 5 -> 0.05
        # ----------------------------------------------------

        return value / 100.0


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    def calculate_metrics(
        self,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(
            metrics,
            dict,
        ):

            raise ValueError(
                "Risk metrics must be a dictionary."
            )

        # ----------------------------------------------------
        # Extract metrics
        # ----------------------------------------------------

        var_95 = self._get_metric(
            metrics,
            [
                "VaR_95",
                "VaR",
                "var_95",
                "var",
            ],
        )

        expected_shortfall = (
            self._get_metric(
                metrics,
                [
                    "Expected_Shortfall_95",
                    "Expected_Shortfall",
                    "ES_95",
                    "ES",
                    "expected_shortfall",
                ],
            )
        )

        maximum_drawdown = (
            self._get_metric(
                metrics,
                [
                    "Maximum_Drawdown",
                    "Max_Drawdown",
                    "MDD",
                    "maximum_drawdown",
                ],
            )
        )

        expected_loss = (
            self._get_metric(
                metrics,
                [
                    "Expected_Loss",
                    "expected_loss",
                ],
            )
        )

        worst_scenario_loss = (
            self._get_worst_scenario_loss(
                metrics
            )
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        var_95 = self.normalize_risk_value(
            var_95
        )

        expected_shortfall = (
            self.normalize_risk_value(
                expected_shortfall
            )
        )

        maximum_drawdown = (
            self.normalize_risk_value(
                maximum_drawdown
            )
        )

        expected_loss = (
            self.normalize_risk_value(
                expected_loss
            )
        )

        worst_scenario_loss = (
            self.normalize_risk_value(
                worst_scenario_loss
            )
        )

        # ----------------------------------------------------
        # Convert to monetary values
        # ----------------------------------------------------

        var_95_amount = (
            self.portfolio_amount
            * abs(var_95)
        )

        expected_shortfall_amount = (
            self.portfolio_amount
            * abs(expected_shortfall)
        )

        maximum_drawdown_amount = (
            self.portfolio_amount
            * abs(maximum_drawdown)
        )

        expected_loss_amount = (
            self.portfolio_amount
            * abs(expected_loss)
        )

        worst_scenario_loss_amount = (
            self.portfolio_amount
            * abs(worst_scenario_loss)
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {

            # =================================================
            # DECIMAL RISK VALUES
            # =================================================

            "VaR_95":
                var_95,

            "Expected_Shortfall_95":
                expected_shortfall,

            "Maximum_Drawdown":
                maximum_drawdown,

            "Expected_Loss":
                expected_loss,

            "Worst_Scenario_Loss":
                worst_scenario_loss,

            # =================================================
            # MONETARY VALUES
            # =================================================

            "VaR_95_amount":
                var_95_amount,

            "Expected_Shortfall_95_amount":
                expected_shortfall_amount,

            "Maximum_Drawdown_amount":
                maximum_drawdown_amount,

            "Expected_Loss_amount":
                expected_loss_amount,

            "Worst_Scenario_Loss_amount":
                worst_scenario_loss_amount,

            # =================================================
            # FORMATTED
            # =================================================

            "VaR_95_formatted":
                self.format_currency(
                    var_95_amount
                ),

            "Expected_Shortfall_95_formatted":
                self.format_currency(
                    expected_shortfall_amount
                ),

            "Maximum_Drawdown_formatted":
                self.format_currency(
                    maximum_drawdown_amount
                ),

            "Expected_Loss_formatted":
                self.format_currency(
                    expected_loss_amount
                ),

            "Worst_Scenario_Loss_formatted":
                self.format_currency(
                    worst_scenario_loss_amount
                ),

            # =================================================
            # INDIAN SCALE
            # =================================================

            "VaR_95_scale":
                self.format_indian_scale(
                    var_95_amount
                ),

            "Expected_Shortfall_95_scale":
                self.format_indian_scale(
                    expected_shortfall_amount
                ),

            "Maximum_Drawdown_scale":
                self.format_indian_scale(
                    maximum_drawdown_amount
                ),

            "Expected_Loss_scale":
                self.format_indian_scale(
                    expected_loss_amount
                ),

            "Worst_Scenario_Loss_scale":
                self.format_indian_scale(
                    worst_scenario_loss_amount
                ),
        }


    # ========================================================
    # COMPLETE CALCULATION
    # ========================================================

    def calculate(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(
            risk_result,
            dict,
        ):

            raise ValueError(
                "risk_result must be a dictionary."
            )

        # ====================================================
        # BASELINE
        # ====================================================

        baseline = risk_result.get(
            "baseline"
        )

        if baseline is None:

            baseline = risk_result.get(
                "baseline_metrics"
            )

        # ====================================================
        # STRESSED
        # ====================================================

        stressed = risk_result.get(
            "stressed"
        )

        if stressed is None:

            stressed = risk_result.get(
                "stressed_metrics"
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        if not isinstance(
            baseline,
            dict,
        ):

            raise RuntimeError(
                "Risk result does not contain "
                "valid baseline metrics."
            )

        if not isinstance(
            stressed,
            dict,
        ):

            raise RuntimeError(
                "Risk result does not contain "
                "valid stressed metrics."
            )

        # ====================================================
        # CALCULATE
        # ====================================================

        baseline_result = (
            self.calculate_metrics(
                baseline
            )
        )

        stressed_result = (
            self.calculate_metrics(
                stressed
            )
        )

        # ====================================================
        # STRESS IMPACT
        # ====================================================

        expected_loss_change = (
            stressed_result[
                "Expected_Loss_amount"
            ]
            -
            baseline_result[
                "Expected_Loss_amount"
            ]
        )

        var_change = (
            stressed_result[
                "VaR_95_amount"
            ]
            -
            baseline_result[
                "VaR_95_amount"
            ]
        )

        es_change = (
            stressed_result[
                "Expected_Shortfall_95_amount"
            ]
            -
            baseline_result[
                "Expected_Shortfall_95_amount"
            ]
        )

        drawdown_change = (
            stressed_result[
                "Maximum_Drawdown_amount"
            ]
            -
            baseline_result[
                "Maximum_Drawdown_amount"
            ]
        )

        worst_loss_change = (
            stressed_result[
                "Worst_Scenario_Loss_amount"
            ]
            -
            baseline_result[
                "Worst_Scenario_Loss_amount"
            ]
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "portfolio_amount":
                self.portfolio_amount,

            "portfolio_amount_formatted":
                self.format_currency(
                    self.portfolio_amount
                ),

            "portfolio_amount_scale":
                self.format_indian_scale(
                    self.portfolio_amount
                ),

            "currency":
                self.currency,

            "baseline":
                baseline_result,

            "stressed":
                stressed_result,

            "stress_impact": {

                "Expected_Loss_change":
                    expected_loss_change,

                "VaR_95_change":
                    var_change,

                "Expected_Shortfall_95_change":
                    es_change,

                "Maximum_Drawdown_change":
                    drawdown_change,

                "Worst_Scenario_Loss_change":
                    worst_loss_change,

                "Expected_Loss_change_formatted":
                    self.format_currency(
                        expected_loss_change
                    ),

                "VaR_95_change_formatted":
                    self.format_currency(
                        var_change
                    ),

                "Expected_Shortfall_95_change_formatted":
                    self.format_currency(
                        es_change
                    ),

                "Maximum_Drawdown_change_formatted":
                    self.format_currency(
                        drawdown_change
                    ),

                "Worst_Scenario_Loss_change_formatted":
                    self.format_currency(
                        worst_loss_change
                    ),

                "Expected_Loss_change_scale":
                    self.format_indian_scale(
                        expected_loss_change
                    ),

                "VaR_95_change_scale":
                    self.format_indian_scale(
                        var_change
                    ),

                "Expected_Shortfall_95_change_scale":
                    self.format_indian_scale(
                        es_change
                    ),

                "Maximum_Drawdown_change_scale":
                    self.format_indian_scale(
                        drawdown_change
                    ),

                "Worst_Scenario_Loss_change_scale":
                    self.format_indian_scale(
                        worst_loss_change
                    ),
            },
        }


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "MONETARY RISK CALCULATOR TEST"
    )

    print(
        "=" * 78
    )

    portfolio_amount = 100_000_000

    calculator = MonetaryRiskCalculator(
        portfolio_amount=portfolio_amount,
        currency="INR",
    )

    # --------------------------------------------------------
    # TEST 1
    # Current RiskAgent nested structure
    # --------------------------------------------------------

    nested_risk_result = {

        "baseline": {

            "risk_metrics": {

                "VaR_95":
                    0.050,

                "Expected_Shortfall_95":
                    0.080,

                "Maximum_Drawdown":
                    0.120,

                "Expected_Loss":
                    0.003654,
            },

            "worst_scenario": {

                "loss_magnitude":
                    0.060,
            },
        },

        "stressed": {

            "risk_metrics": {

                "VaR_95":
                    0.100,

                "Expected_Shortfall_95":
                    0.150,

                "Maximum_Drawdown":
                    0.250,

                "Expected_Loss":
                    0.080,
            },

            "worst_scenario": {

                "loss_magnitude":
                    0.150,
            },
        },
    }

    result = calculator.calculate(
        nested_risk_result
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "\nPortfolio:"
    )

    print(
        result[
            "portfolio_amount_formatted"
        ]
    )

    print(
        "\nPortfolio scale:"
    )

    print(
        result[
            "portfolio_amount_scale"
        ]
    )

    print(
        "\nBASELINE"
    )

    print(
        "VaR 95%:",
        result[
            "baseline"
        ][
            "VaR_95_formatted"
        ],
    )

    print(
        "Expected Shortfall:",
        result[
            "baseline"
        ][
            "Expected_Shortfall_95_formatted"
        ],
    )

    print(
        "Maximum Drawdown:",
        result[
            "baseline"
        ][
            "Maximum_Drawdown_formatted"
        ],
    )

    print(
        "Expected Loss:",
        result[
            "baseline"
        ][
            "Expected_Loss_formatted"
        ],
    )

    print(
        "Worst Scenario Loss:",
        result[
            "baseline"
        ][
            "Worst_Scenario_Loss_formatted"
        ],
    )

    print(
        "\nSTRESSED"
    )

    print(
        "VaR 95%:",
        result[
            "stressed"
        ][
            "VaR_95_formatted"
        ],
    )

    print(
        "Expected Shortfall:",
        result[
            "stressed"
        ][
            "Expected_Shortfall_95_formatted"
        ],
    )

    print(
        "Maximum Drawdown:",
        result[
            "stressed"
        ][
            "Maximum_Drawdown_formatted"
        ],
    )

    print(
        "Expected Loss:",
        result[
            "stressed"
        ][
            "Expected_Loss_formatted"
        ],
    )

    print(
        "Worst Scenario Loss:",
        result[
            "stressed"
        ][
            "Worst_Scenario_Loss_formatted"
        ],
    )

    print(
        "\nSTRESS IMPACT"
    )

    print(
        "Expected Loss Increase:",
        result[
            "stress_impact"
        ][
            "Expected_Loss_change_formatted"
        ],
    )

    print(
        "VaR Increase:",
        result[
            "stress_impact"
        ][
            "VaR_95_change_formatted"
        ],
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TEST COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()