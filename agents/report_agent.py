"""
MacroStress-GAN
Institutional Daily Risk System

Phase 9 - Report Agent
----------------------

Consumes validated outputs from:

    Phase 5 - Scenario-Aware Simulation Agent
    Phase 6 - Risk Agent
    Phase 7 - Orchestrator
    Phase 8 - Critic / Validation Agent

The Report Agent does not recalculate risk metrics.
It converts the validated results into a structured
institutional risk report.
"""

from typing import Dict, Any


class ReportAgent:
    """Generates an institutional risk report."""

    def __init__(self):
        print("Initializing Report Agent...")
        print("Report Agent loaded successfully.")

    # ------------------------------------------------------------------
    # SAFE VALUE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _percent(value: Any) -> str:
        """Convert decimal value to percentage text."""

        return f"{ReportAgent._number(value) * 100:.2f}%"

    # ------------------------------------------------------------------
    # SCENARIO SUMMARY
    # ------------------------------------------------------------------

    def build_scenario_summary(
        self,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the scenario section."""

        return {
            "scenario_name": scenario.get(
                "scenario_name",
                "Unknown Scenario",
            ),
            "horizon_days": int(
                scenario.get(
                    "horizon_days",
                    0,
                )
            ),
            "shocks": {
                "NIFTY50": self._number(
                    scenario.get(
                        "NIFTY50_shock_pct",
                        0.0,
                    )
                ),
                "CRUDE_OIL": self._number(
                    scenario.get(
                        "CRUDE_OIL_shock_pct",
                        0.0,
                    )
                ),
                "USD_INR": self._number(
                    scenario.get(
                        "USD_INR_shock_pct",
                        0.0,
                    )
                ),
                "INDIA_VIX": self._number(
                    scenario.get(
                        "INDIA_VIX_shock_pct",
                        0.0,
                    )
                ),
                "INDIA_10Y_YIELD": self._number(
                    scenario.get(
                        "INDIA_10Y_YIELD_shock_pct",
                        0.0,
                    )
                ),
            },
        }

    # ------------------------------------------------------------------
    # RISK SUMMARY
    # ------------------------------------------------------------------

    def build_risk_summary(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build baseline and stressed risk sections."""

        baseline = risk_result.get(
            "baseline",
            {},
        )

        stressed = risk_result.get(
            "stressed",
            {},
        )

        baseline_metrics = baseline.get(
            "risk_metrics",
            {},
        )

        stressed_metrics = stressed.get(
            "risk_metrics",
            {},
        )

        metric_names = [
            "VaR_95",
            "Expected_Shortfall_95",
            "Maximum_Drawdown",
            "Expected_Loss",
        ]

        baseline_metrics_clean = {}
        stressed_metrics_clean = {}

        for metric in metric_names:

            baseline_metrics_clean[metric] = (
                self._number(
                    baseline_metrics.get(
                        metric,
                        0.0,
                    )
                )
            )

            stressed_metrics_clean[metric] = (
                self._number(
                    stressed_metrics.get(
                        metric,
                        0.0,
                    )
                )
            )

        return {
            "baseline": {
                "risk_level": baseline.get(
                    "risk_level",
                    "Unknown",
                ),
                "metrics": baseline_metrics_clean,
            },
            "stressed": {
                "risk_level": stressed.get(
                    "risk_level",
                    "Unknown",
                ),
                "metrics": stressed_metrics_clean,
            },
        }

    # ------------------------------------------------------------------
    # STRESS IMPACT
    # ------------------------------------------------------------------

    def build_stress_impact(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract stress impact from the Risk Agent."""

        impact = risk_result.get(
            "stress_impact",
            {},
        )

        result = {}

        metric_names = [
            "VaR_95",
            "Expected_Shortfall_95",
            "Maximum_Drawdown",
            "Expected_Loss",
            "Portfolio_Mean_Return",
        ]

        for metric in metric_names:

            values = impact.get(
                metric,
                {},
            )

            result[metric] = {
                "baseline": self._number(
                    values.get(
                        "baseline",
                        0.0,
                    )
                ),
                "stressed": self._number(
                    values.get(
                        "stressed",
                        0.0,
                    )
                ),
                "delta": self._number(
                    values.get(
                        "delta",
                        0.0,
                    )
                ),
                "percentage_change": self._number(
                    values.get(
                        "percentage_change",
                        0.0,
                    )
                ),
            }

        return result

    # ------------------------------------------------------------------
    # FEATURE IMPACT
    # ------------------------------------------------------------------

    def build_feature_impact(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract feature-level stress impact."""

        feature_impact = risk_result.get(
            "feature_impact",
            {},
        )

        result = {}

        for feature, values in feature_impact.items():

            result[feature] = {
                "baseline_mean": self._number(
                    values.get(
                        "baseline_mean",
                        0.0,
                    )
                ),
                "stressed_mean": self._number(
                    values.get(
                        "stressed_mean",
                        0.0,
                    )
                ),
                "mean_delta": self._number(
                    values.get(
                        "mean_delta",
                        0.0,
                    )
                ),
                "baseline_std": self._number(
                    values.get(
                        "baseline_std",
                        0.0,
                    )
                ),
                "stressed_std": self._number(
                    values.get(
                        "stressed_std",
                        0.0,
                    )
                ),
                "std_delta": self._number(
                    values.get(
                        "std_delta",
                        0.0,
                    )
                ),
            }

        return result

    # ------------------------------------------------------------------
    # WORST SCENARIO
    # ------------------------------------------------------------------

    def build_worst_scenario(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        baseline = risk_result.get(
            "baseline",
            {}
        )

        stressed = risk_result.get(
            "stressed",
            {}
        )

        baseline_worst = baseline.get(
            "worst_scenario",
            {}
        )

        stressed_worst = stressed.get(
            "worst_scenario",
            {}
        )

        return {
            "baseline": {
                "scenario_index": baseline_worst.get(
                    "scenario_index"
                ),
                "cumulative_return": self._number(
                    baseline_worst.get(
                        "cumulative_return",
                        0.0,
                    )
                ),
                "loss_magnitude": self._number(
                    baseline_worst.get(
                        "loss_magnitude",
                        0.0,
                    )
                ),
            },
            "stressed": {
                "scenario_index": stressed_worst.get(
                    "scenario_index"
                ),
                "cumulative_return": self._number(
                    stressed_worst.get(
                        "cumulative_return",
                        0.0,
                    )
                ),
                "loss_magnitude": self._number(
                    stressed_worst.get(
                        "loss_magnitude",
                        0.0,
                    )
                ),
            },
        }

    # ------------------------------------------------------------------
    # VALIDATION SUMMARY
    # ------------------------------------------------------------------

    def build_validation_summary(
        self,
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "overall_status": validation_result.get(
                "overall_status",
                "UNKNOWN",
            ),
            "validation_score": self._number(
                validation_result.get(
                    "validation_score",
                    0.0,
                )
            ),
            "passed_checks": validation_result.get(
                "passed_checks",
                0,
            ),
            "total_checks": validation_result.get(
                "total_checks",
                0,
            ),
        }

    # ------------------------------------------------------------------
    # KEY FINDINGS
    # ------------------------------------------------------------------

    def generate_key_findings(
        self,
        scenario: Dict[str, Any],
        risk_result: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> list:

        findings = []

        scenario_name = scenario.get(
            "scenario_name",
            "Unknown Scenario",
        )

        findings.append(
            f"The analysis evaluates a "
            f"{scenario_name} over a "
            f"{scenario.get('horizon_days', 0)}-day horizon."
        )

        baseline = risk_result.get(
            "baseline",
            {},
        )

        stressed = risk_result.get(
            "stressed",
            {},
        )

        findings.append(
            "Baseline risk classification: "
            f"{baseline.get('risk_level', 'Unknown')}."
        )

        findings.append(
            "Stressed risk classification: "
            f"{stressed.get('risk_level', 'Unknown')}."
        )

        impact = risk_result.get(
            "stress_impact",
            {},
        )

        var_delta = self._number(
            impact.get(
                "VaR_95",
                {},
            ).get(
                "delta",
                0.0,
            )
        )

        es_delta = self._number(
            impact.get(
                "Expected_Shortfall_95",
                {},
            ).get(
                "delta",
                0.0,
            )
        )

        mdd_delta = self._number(
            impact.get(
                "Maximum_Drawdown",
                {},
            ).get(
                "delta",
                0.0,
            )
        )

        el_delta = self._number(
            impact.get(
                "Expected_Loss",
                {},
            ).get(
                "delta",
                0.0,
            )
        )

        findings.append(
            f"VaR change under the scenario: "
            f"{var_delta:.6f} "
            f"({var_delta * 100:.2f} percentage points)."
        )

        findings.append(
            f"Expected Shortfall change: "
            f"{es_delta:.6f} "
            f"({es_delta * 100:.2f} percentage points)."
        )

        findings.append(
            f"Maximum Drawdown change: "
            f"{mdd_delta:.6f} "
            f"({mdd_delta * 100:.2f} percentage points)."
        )

        findings.append(
            f"Expected Loss change: "
            f"{el_delta:.6f} "
            f"({el_delta * 100:.2f} percentage points)."
        )

        validation_status = validation_result.get(
            "overall_status",
            "UNKNOWN",
        )

        findings.append(
            "Critic / Validation Agent status: "
            f"{validation_status}."
        )

        return findings

    # ------------------------------------------------------------------
    # COMPLETE REPORT
    # ------------------------------------------------------------------

    def generate_report(
        self,
        scenario: Dict[str, Any],
        simulation_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate the complete structured institutional report.

        simulation_result is included to preserve the complete
        multi-agent pipeline context.
        """

        report = {
            "status": "success",
            "agent": "Report Agent",
            "phase": "Phase 9",

            "report_metadata": {
                "report_type": (
                    "Institutional Stress Test Risk Report"
                ),
                "model": "TimeGAN V3.4",
                "simulation_scenarios": (
                    simulation_result.get(
                        "num_scenarios",
                        10,
                    )
                ),
                "horizon_days": scenario.get(
                    "horizon_days",
                    30,
                ),
            },

            "scenario": self.build_scenario_summary(
                scenario
            ),

            "risk_summary": self.build_risk_summary(
                risk_result
            ),

            "stress_impact": self.build_stress_impact(
                risk_result
            ),

            "feature_impact": self.build_feature_impact(
                risk_result
            ),

            "worst_scenario": self.build_worst_scenario(
                risk_result
            ),

            "validation": self.build_validation_summary(
                validation_result
            ),

            "key_findings": self.generate_key_findings(
                scenario,
                risk_result,
                validation_result,
            ),
        }

        return report

    # ------------------------------------------------------------------
    # PUBLIC RUN METHOD
    # ------------------------------------------------------------------

    def run(
        self,
        scenario: Dict[str, Any],
        simulation_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        return self.generate_report(
            scenario=scenario,
            simulation_result=simulation_result,
            risk_result=risk_result,
            validation_result=validation_result,
        )

    # ------------------------------------------------------------------
    # REPORT PRINTING
    # ------------------------------------------------------------------

    def print_report(
        self,
        report: Dict[str, Any],
    ) -> None:

        scenario = report["scenario"]
        risk_summary = report["risk_summary"]
        stress_impact = report["stress_impact"]
        validation = report["validation"]

        print("\n" + "=" * 78)
        print("MACROSTRESS-GAN - PHASE 9")
        print("INSTITUTIONAL STRESS TEST RISK REPORT")
        print("=" * 78)

        # --------------------------------------------------------------
        # REPORT METADATA
        # --------------------------------------------------------------

        metadata = report["report_metadata"]

        print("\nREPORT METADATA")
        print("-" * 78)

        print(
            f"Report Type        : "
            f"{metadata['report_type']}"
        )

        print(
            f"Model              : "
            f"{metadata['model']}"
        )

        print(
            f"Scenarios          : "
            f"{metadata['simulation_scenarios']}"
        )

        print(
            f"Horizon            : "
            f"{metadata['horizon_days']} days"
        )

        # --------------------------------------------------------------
        # SCENARIO
        # --------------------------------------------------------------

        print("\nSCENARIO")
        print("-" * 78)

        print(
            f"Scenario Name      : "
            f"{scenario['scenario_name']}"
        )

        print(
            f"Horizon            : "
            f"{scenario['horizon_days']} days"
        )

        shocks = scenario["shocks"]

        print(
            f"NIFTY50 Shock      : "
            f"{shocks['NIFTY50']:.2f}%"
        )

        print(
            f"Crude Oil Shock    : "
            f"{shocks['CRUDE_OIL']:.2f}%"
        )

        print(
            f"USD/INR Shock      : "
            f"{shocks['USD_INR']:.2f}%"
        )

        print(
            f"India VIX Shock    : "
            f"{shocks['INDIA_VIX']:.2f}%"
        )

        print(
            f"10Y Yield Shock     : "
            f"{shocks['INDIA_10Y_YIELD']:.2f}%"
        )

        # --------------------------------------------------------------
        # RISK SUMMARY
        # --------------------------------------------------------------

        print("\nRISK SUMMARY")
        print("-" * 78)

        print(
            f"{'Metric':<35}"
            f"{'Baseline':>15}"
            f"{'Stressed':>15}"
        )

        print("-" * 78)

        metric_labels = [
            ("VaR (95%)", "VaR_95"),
            (
                "Expected Shortfall",
                "Expected_Shortfall_95",
            ),
            (
                "Maximum Drawdown",
                "Maximum_Drawdown",
            ),
            (
                "Expected Loss",
                "Expected_Loss",
            ),
        ]

        for label, key in metric_labels:

            base = risk_summary[
                "baseline"
            ]["metrics"][key]

            stress = risk_summary[
                "stressed"
            ]["metrics"][key]

            print(
                f"{label:<35}"
                f"{base:>15.6f}"
                f"{stress:>15.6f}"
            )

        print(
            f"\nBaseline Risk Level : "
            f"{risk_summary['baseline']['risk_level']}"
        )

        print(
            f"Stressed Risk Level : "
            f"{risk_summary['stressed']['risk_level']}"
        )

        # --------------------------------------------------------------
        # STRESS IMPACT
        # --------------------------------------------------------------

        print("\nSTRESS IMPACT")
        print("-" * 78)

        for label, key in metric_labels:

            values = stress_impact[key]

            print(
                f"{label:<28}"
                f"Delta: {values['delta']:>12.6f}   "
                f"Change: {values['percentage_change']:>9.2f}%"
            )

        # --------------------------------------------------------------
        # FEATURE IMPACT
        # --------------------------------------------------------------

        print("\nFEATURE-LEVEL IMPACT")
        print("-" * 78)

        print(
            f"{'Feature':<35}"
            f"{'Baseline':>15}"
            f"{'Stressed':>15}"
            f"{'Delta':>15}"
        )

        print("-" * 78)

        for feature, values in report[
            "feature_impact"
        ].items():

            print(
                f"{feature:<35}"
                f"{values['baseline_mean']:>15.6f}"
                f"{values['stressed_mean']:>15.6f}"
                f"{values['mean_delta']:>15.6f}"
            )

        # --------------------------------------------------------------
        # WORST SCENARIO
        # --------------------------------------------------------------

        print("\nWORST SCENARIO")
        print("-" * 78)

        worst = report["worst_scenario"]

        print("Baseline:")

        print(
            f"  Scenario Index    : "
            f"{worst['baseline']['scenario_index']}"
        )

        print(
            f"  Cumulative Return : "
            f"{worst['baseline']['cumulative_return']:.6f} "
            f"({worst['baseline']['cumulative_return'] * 100:.2f}%)"
        )

        print("\nStressed:")

        print(
            f"  Scenario Index    : "
            f"{worst['stressed']['scenario_index']}"
        )

        print(
            f"  Cumulative Return : "
            f"{worst['stressed']['cumulative_return']:.6f} "
            f"({worst['stressed']['cumulative_return'] * 100:.2f}%)"
        )

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        print("\nVALIDATION")
        print("-" * 78)

        print(
            f"Overall Status     : "
            f"{validation['overall_status']}"
        )

        print(
            f"Validation Score   : "
            f"{validation['validation_score']:.2%}"
        )

        print(
            f"Checks Passed      : "
            f"{validation['passed_checks']}/"
            f"{validation['total_checks']}"
        )

        # --------------------------------------------------------------
        # KEY FINDINGS
        # --------------------------------------------------------------

        print("\nKEY FINDINGS")
        print("-" * 78)

        for index, finding in enumerate(
            report["key_findings"],
            start=1,
        ):

            print(
                f"{index}. {finding}"
            )

        print("\n" + "=" * 78)
        print("PHASE 9 REPORT GENERATION COMPLETED")
        print("=" * 78)


# ==========================================================================
# STANDALONE TEST
# ==========================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("MACROSTRESS-GAN - PHASE 9")
    print("REPORT AGENT")
    print("=" * 78)

    agent = ReportAgent()

    scenario = {
        "scenario_name": "Geopolitical Crisis",
        "horizon_days": 30,
        "NIFTY50_shock_pct": -15.0,
        "CRUDE_OIL_shock_pct": 30.0,
        "USD_INR_shock_pct": 5.0,
        "INDIA_VIX_shock_pct": 50.0,
        "INDIA_10Y_YIELD_shock_pct": 1.0,
    }

    simulation_result = {
        "num_scenarios": 10,
        "horizon_days": 30,
        "model": "TimeGAN V3.4",
    }

    risk_result = {
        "baseline": {
            "risk_level": "Low",
            "risk_metrics": {
                "VaR_95": 0.010,
                "Expected_Shortfall_95": 0.015,
                "Maximum_Drawdown": 0.060,
                "Expected_Loss": 0.005,
            },
            "worst_scenario": {
                "scenario_index": 2,
                "cumulative_return": -0.040,
                "loss_magnitude": 0.040,
            },
        },

        "stressed": {
            "risk_level": "Low",
            "risk_metrics": {
                "VaR_95": 0.012,
                "Expected_Shortfall_95": 0.018,
                "Maximum_Drawdown": 0.070,
                "Expected_Loss": 0.006,
            },
            "worst_scenario": {
                "scenario_index": 4,
                "cumulative_return": -0.050,
                "loss_magnitude": 0.050,
            },
        },

        "stress_impact": {
            "VaR_95": {
                "baseline": 0.010,
                "stressed": 0.012,
                "delta": 0.002,
                "percentage_change": 20.0,
            },
            "Expected_Shortfall_95": {
                "baseline": 0.015,
                "stressed": 0.018,
                "delta": 0.003,
                "percentage_change": 20.0,
            },
            "Maximum_Drawdown": {
                "baseline": 0.060,
                "stressed": 0.070,
                "delta": 0.010,
                "percentage_change": 16.67,
            },
            "Expected_Loss": {
                "baseline": 0.005,
                "stressed": 0.006,
                "delta": 0.001,
                "percentage_change": 20.0,
            },
            "Portfolio_Mean_Return": {
                "baseline": 0.001,
                "stressed": 0.0005,
                "delta": -0.0005,
                "percentage_change": -50.0,
            },
        },

        "feature_impact": {
            "NIFTY50_Return": {
                "baseline_mean": 0.002,
                "stressed_mean": -0.005,
                "mean_delta": -0.007,
                "baseline_std": 0.02,
                "stressed_std": 0.025,
                "std_delta": 0.005,
            },
            "CRUDE_OIL_Return": {
                "baseline_mean": 0.001,
                "stressed_mean": 0.010,
                "mean_delta": 0.009,
                "baseline_std": 0.03,
                "stressed_std": 0.035,
                "std_delta": 0.005,
            },
            "USD_INR_Return": {
                "baseline_mean": 0.001,
                "stressed_mean": 0.002,
                "mean_delta": 0.001,
                "baseline_std": 0.01,
                "stressed_std": 0.012,
                "std_delta": 0.002,
            },
            "INDIA_VIX_Change": {
                "baseline_mean": -0.005,
                "stressed_mean": 0.020,
                "mean_delta": 0.025,
                "baseline_std": 0.02,
                "stressed_std": 0.03,
                "std_delta": 0.01,
            },
            "INDIA_10Y_YIELD_Change": {
                "baseline_mean": 0.002,
                "stressed_mean": 0.025,
                "mean_delta": 0.023,
                "baseline_std": 0.01,
                "stressed_std": 0.015,
                "std_delta": 0.005,
            },
        },
    }

    validation_result = {
        "overall_status": "PASSED",
        "validation_score": 1.0,
        "passed_checks": 7,
        "total_checks": 7,
    }

    report = agent.run(
        scenario=scenario,
        simulation_result=simulation_result,
        risk_result=risk_result,
        validation_result=validation_result,
    )

    agent.print_report(report)