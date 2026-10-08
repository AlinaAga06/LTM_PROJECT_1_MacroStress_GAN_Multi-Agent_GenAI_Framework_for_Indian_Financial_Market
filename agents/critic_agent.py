"""
MacroStress-GAN
Institutional Daily Risk System

Phase 8 - Critic / Validation Agent
-----------------------------------

Validates outputs produced by:

    Phase 5 - Scenario-Aware Simulation Agent
    Phase 6 - Risk Agent
    Phase 7 - Orchestrator

The Critic Agent does not generate synthetic data and does not
modify the simulation or risk outputs.

It checks:

1. Input structure
2. Shape consistency
3. NaN / Inf values
4. Scaled-data range
5. Scenario effect consistency
6. Risk metric consistency
7. Risk classification consistency
8. Overall validation status
"""

from typing import Dict, Any, Optional

import numpy as np


class CriticAgent:
    """Validates Simulation Agent and Risk Agent outputs."""

    EXPECTED_FEATURE_COUNT = 5
    EXPECTED_SCALED_MIN = 0.0
    EXPECTED_SCALED_MAX = 1.0

    RISK_LEVELS = {
        "Low",
        "Moderate",
        "High",
    }

    def __init__(self):
        print("Initializing Critic / Validation Agent...")
        print("Critic / Validation Agent loaded successfully.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert a value to float safely."""

        try:
            result = float(value)

            if np.isfinite(result):
                return result

        except (TypeError, ValueError):
            pass

        return None

    # ------------------------------------------------------------------
    # 1. Shape Validation
    # ------------------------------------------------------------------

    def validate_shapes(
        self,
        baseline_sequences: np.ndarray,
        stressed_sequences: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate baseline and stressed sequence dimensions."""

        checks = []

        baseline_shape = tuple(baseline_sequences.shape)
        stressed_shape = tuple(stressed_sequences.shape)

        baseline_valid = (
            baseline_sequences.ndim == 3
            and baseline_sequences.shape[-1]
            == self.EXPECTED_FEATURE_COUNT
        )

        stressed_valid = (
            stressed_sequences.ndim == 3
            and stressed_sequences.shape[-1]
            == self.EXPECTED_FEATURE_COUNT
        )

        checks.append(
            {
                "check": "Baseline shape",
                "passed": baseline_valid,
                "details": str(baseline_shape),
            }
        )

        checks.append(
            {
                "check": "Stressed shape",
                "passed": stressed_valid,
                "details": str(stressed_shape),
            }
        )

        same_structure = (
            baseline_sequences.ndim == stressed_sequences.ndim
            and baseline_sequences.shape == stressed_sequences.shape
        )

        checks.append(
            {
                "check": "Baseline/Stressed shape consistency",
                "passed": same_structure,
                "details": (
                    f"baseline={baseline_shape}, "
                    f"stressed={stressed_shape}"
                ),
            }
        )

        passed = all(check["passed"] for check in checks)

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 2. NaN / Inf Validation
    # ------------------------------------------------------------------

    def validate_numeric_values(
        self,
        baseline_sequences: np.ndarray,
        stressed_sequences: np.ndarray,
    ) -> Dict[str, Any]:
        """Check arrays for NaN and infinite values."""

        baseline_nan = int(np.isnan(baseline_sequences).sum())
        baseline_inf = int(np.isinf(baseline_sequences).sum())

        stressed_nan = int(np.isnan(stressed_sequences).sum())
        stressed_inf = int(np.isinf(stressed_sequences).sum())

        passed = (
            baseline_nan == 0
            and baseline_inf == 0
            and stressed_nan == 0
            and stressed_inf == 0
        )

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "baseline_nan": baseline_nan,
            "baseline_inf": baseline_inf,
            "stressed_nan": stressed_nan,
            "stressed_inf": stressed_inf,
        }

    # ------------------------------------------------------------------
    # 3. Scaled Range Validation
    # ------------------------------------------------------------------

    def validate_scaled_range(
        self,
        baseline_sequences: np.ndarray,
        stressed_sequences: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Validate scaled values.

        TimeGAN outputs should normally remain inside [0, 1].
        A very small numerical tolerance is allowed.
        """

        baseline_min = float(np.min(baseline_sequences))
        baseline_max = float(np.max(baseline_sequences))

        stressed_min = float(np.min(stressed_sequences))
        stressed_max = float(np.max(stressed_sequences))

        baseline_passed = (
            baseline_min >= self.EXPECTED_SCALED_MIN - tolerance
            and baseline_max <= self.EXPECTED_SCALED_MAX + tolerance
        )

        stressed_passed = (
            stressed_min >= self.EXPECTED_SCALED_MIN - tolerance
            and stressed_max <= self.EXPECTED_SCALED_MAX + tolerance
        )

        passed = baseline_passed and stressed_passed

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "baseline": {
                "min": baseline_min,
                "max": baseline_max,
                "passed": baseline_passed,
            },
            "stressed": {
                "min": stressed_min,
                "max": stressed_max,
                "passed": stressed_passed,
            },
        }

    # ------------------------------------------------------------------
    # 4. Scenario Effect Validation
    # ------------------------------------------------------------------

    def validate_scenario_effect(
        self,
        scenario: Dict[str, Any],
        feature_impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check whether the observed feature changes are directionally
        consistent with the requested scenario.

        This is a directional validation rather than a strict numerical
        equality check because the GAN generates stochastic paths.
        """

        expected_directions = {
            "NIFTY50_Return": -1,
            "CRUDE_OIL_Return": 1,
            "USD_INR_Return": 1,
            "INDIA_VIX_Change": 1,
            "INDIA_10Y_YIELD_Change": 1,
        }

        scenario_mapping = {
            "NIFTY50_Return": "NIFTY50_shock_pct",
            "CRUDE_OIL_Return": "CRUDE_OIL_shock_pct",
            "USD_INR_Return": "USD_INR_shock_pct",
            "INDIA_VIX_Change": "INDIA_VIX_shock_pct",
            "INDIA_10Y_YIELD_Change": "INDIA_10Y_YIELD_shock_pct",
        }

        checks = []

        for feature, expected_direction in expected_directions.items():

            impact = feature_impact.get(feature, {})

            delta = self._safe_float(
                impact.get("mean_delta")
            )

            scenario_key = scenario_mapping.get(feature)

            target = self._safe_float(
                scenario.get(scenario_key, 0.0)
            )

            if delta is None:
                checks.append(
                    {
                        "feature": feature,
                        "passed": False,
                        "delta": None,
                        "target": target,
                        "reason": "Missing or invalid mean_delta",
                    }
                )
                continue

            # No shock requested means there is no directional
            # requirement.
            if target is None or abs(target) < 1e-12:

                checks.append(
                    {
                        "feature": feature,
                        "passed": True,
                        "delta": delta,
                        "target": target,
                        "reason": "No directional shock required",
                    }
                )

                continue

            actual_direction = (
                1 if delta > 0
                else -1 if delta < 0
                else 0
            )

            passed = actual_direction == expected_direction

            checks.append(
                {
                    "feature": feature,
                    "passed": passed,
                    "delta": delta,
                    "target": target,
                    "expected_direction": expected_direction,
                    "actual_direction": actual_direction,
                }
            )

        passed = all(check["passed"] for check in checks)

        return {
            "status": "passed" if passed else "warning",
            "passed": passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 5. Risk Metric Validation
    # ------------------------------------------------------------------

    def validate_risk_metrics(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate presence and numerical consistency of risk metrics."""

        required_metrics = [
            "VaR_95",
            "Expected_Shortfall_95",
            "Maximum_Drawdown",
            "Expected_Loss",
        ]

        checks = []

        for section_name in ["baseline", "stressed"]:

            section = risk_result.get(section_name, {})

            metrics = section.get("risk_metrics", {})

            for metric in required_metrics:

                value = self._safe_float(
                    metrics.get(metric)
                )

                passed = (
                    value is not None
                    and value >= 0
                )

                checks.append(
                    {
                        "section": section_name,
                        "metric": metric,
                        "value": value,
                        "passed": passed,
                    }
                )

        passed = all(check["passed"] for check in checks)

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 6. Risk Classification Validation
    # ------------------------------------------------------------------

    def validate_risk_classification(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate reported risk classification values."""

        checks = []

        for section_name in ["baseline", "stressed"]:

            section = risk_result.get(section_name, {})

            risk_level = section.get("risk_level")

            passed = risk_level in self.RISK_LEVELS

            checks.append(
                {
                    "section": section_name,
                    "risk_level": risk_level,
                    "passed": passed,
                }
            )

        passed = all(check["passed"] for check in checks)

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 7. Stress Impact Consistency
    # ------------------------------------------------------------------

    def validate_stress_impact(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify that reported deltas equal stressed - baseline."""

        baseline = risk_result.get("baseline", {})
        stressed = risk_result.get("stressed", {})
        impact = risk_result.get("stress_impact", {})

        baseline_metrics = baseline.get("risk_metrics", {})
        stressed_metrics = stressed.get("risk_metrics", {})

        checks = []

        metric_names = [
            "VaR_95",
            "Expected_Shortfall_95",
            "Maximum_Drawdown",
            "Expected_Loss",
        ]

        for metric in metric_names:

            base = self._safe_float(
                baseline_metrics.get(metric)
            )

            stress = self._safe_float(
                stressed_metrics.get(metric)
            )

            reported = self._safe_float(
                impact.get(metric, {}).get("delta")
            )

            if (
                base is None
                or stress is None
                or reported is None
            ):
                checks.append(
                    {
                        "metric": metric,
                        "passed": False,
                        "reason": "Missing metric value",
                    }
                )
                continue

            expected_delta = stress - base

            passed = np.isclose(
                expected_delta,
                reported,
                atol=1e-8,
                rtol=1e-5,
            )

            checks.append(
                {
                    "metric": metric,
                    "baseline": base,
                    "stressed": stress,
                    "reported_delta": reported,
                    "expected_delta": expected_delta,
                    "passed": bool(passed),
                }
            )

        passed = all(check["passed"] for check in checks)

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 8. Full Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        baseline_sequences: np.ndarray,
        stressed_sequences: np.ndarray,
        risk_result: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the complete Phase 8 validation."""

        shape_validation = self.validate_shapes(
            baseline_sequences,
            stressed_sequences,
        )

        numeric_validation = self.validate_numeric_values(
            baseline_sequences,
            stressed_sequences,
        )

        range_validation = self.validate_scaled_range(
            baseline_sequences,
            stressed_sequences,
        )

        feature_impact = risk_result.get(
            "feature_impact",
            {},
        )

        scenario_validation = self.validate_scenario_effect(
            scenario,
            feature_impact,
        )

        metric_validation = self.validate_risk_metrics(
            risk_result,
        )

        classification_validation = (
            self.validate_risk_classification(
                risk_result
            )
        )

        impact_validation = self.validate_stress_impact(
            risk_result
        )

        validations = {
            "shape_validation": shape_validation,
            "numeric_validation": numeric_validation,
            "range_validation": range_validation,
            "scenario_effect_validation": scenario_validation,
            "risk_metric_validation": metric_validation,
            "risk_classification_validation": (
                classification_validation
            ),
            "stress_impact_validation": impact_validation,
        }

        critical_checks = [
            shape_validation["passed"],
            numeric_validation["passed"],
            range_validation["passed"],
            metric_validation["passed"],
            classification_validation["passed"],
            impact_validation["passed"],
        ]

        all_critical_passed = all(critical_checks)

        scenario_effect_passed = (
            scenario_validation["passed"]
        )

        if all_critical_passed and scenario_effect_passed:
            overall_status = "PASSED"
        elif all_critical_passed:
            overall_status = "PASSED_WITH_WARNING"
        else:
            overall_status = "FAILED"

        passed_count = sum(
            1
            for validation in validations.values()
            if validation["passed"]
        )

        total_count = len(validations)

        return {
            "status": "success",
            "agent": "Critic / Validation Agent",
            "phase": "Phase 8",
            "overall_status": overall_status,
            "validation_score": (
                passed_count / total_count
                if total_count > 0
                else 0.0
            ),
            "passed_checks": passed_count,
            "total_checks": total_count,
            "validations": validations,
        }

    # ------------------------------------------------------------------
    # Run Wrapper
    # ------------------------------------------------------------------

    def run(
        self,
        baseline_sequences: np.ndarray,
        stressed_sequences: np.ndarray,
        risk_result: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Public Phase 8 execution method."""

        return self.validate(
            baseline_sequences=baseline_sequences,
            stressed_sequences=stressed_sequences,
            risk_result=risk_result,
            scenario=scenario,
        )


# ==========================================================================
# Standalone Test
# ==========================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("MACROSTRESS-GAN - PHASE 8")
    print("CRITIC / VALIDATION AGENT")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Controlled standalone validation test
    # ------------------------------------------------------------------

    rng = np.random.default_rng(42)

    baseline = rng.uniform(
        0.20,
        0.80,
        size=(10, 30, 5),
    )

    stressed = baseline.copy()

    # Controlled directional stress:
    # NIFTY ↓
    stressed[:, :, 0] -= 0.03

    # Crude ↑
    stressed[:, :, 1] += 0.03

    # USD/INR ↑
    stressed[:, :, 2] += 0.01

    # VIX ↑
    stressed[:, :, 3] += 0.04

    # Yield ↑
    stressed[:, :, 4] += 0.02

    stressed = np.clip(
        stressed,
        0.0,
        1.0,
    )

    scenario = {
        "scenario_name": "Geopolitical Crisis",
        "horizon_days": 30,
        "NIFTY50_shock_pct": -15.0,
        "CRUDE_OIL_shock_pct": 30.0,
        "USD_INR_shock_pct": 5.0,
        "INDIA_VIX_shock_pct": 50.0,
        "INDIA_10Y_YIELD_shock_pct": 1.0,
    }

    # Minimal controlled Risk Agent-style result.
    baseline_metrics = {
        "VaR_95": 0.010,
        "Expected_Shortfall_95": 0.015,
        "Maximum_Drawdown": 0.060,
        "Expected_Loss": 0.005,
    }

    stressed_metrics = {
        "VaR_95": 0.012,
        "Expected_Shortfall_95": 0.018,
        "Maximum_Drawdown": 0.070,
        "Expected_Loss": 0.006,
    }

    risk_result = {
        "baseline": {
            "risk_metrics": baseline_metrics,
            "risk_level": "Low",
        },
        "stressed": {
            "risk_metrics": stressed_metrics,
            "risk_level": "Low",
        },
        "stress_impact": {
            "VaR_95": {
                "delta": 0.002,
            },
            "Expected_Shortfall_95": {
                "delta": 0.003,
            },
            "Maximum_Drawdown": {
                "delta": 0.010,
            },
            "Expected_Loss": {
                "delta": 0.001,
            },
        },
        "feature_impact": {
            "NIFTY50_Return": {
                "mean_delta": -0.03,
            },
            "CRUDE_OIL_Return": {
                "mean_delta": 0.03,
            },
            "USD_INR_Return": {
                "mean_delta": 0.01,
            },
            "INDIA_VIX_Change": {
                "mean_delta": 0.04,
            },
            "INDIA_10Y_YIELD_Change": {
                "mean_delta": 0.02,
            },
        },
    }

    agent = CriticAgent()

    result = agent.run(
        baseline_sequences=baseline,
        stressed_sequences=stressed,
        risk_result=risk_result,
        scenario=scenario,
    )

    print("\n" + "-" * 78)
    print("PHASE 8 VALIDATION RESULT")
    print("-" * 78)

    print(
        f"Overall Status    : "
        f"{result['overall_status']}"
    )

    print(
        f"Validation Score  : "
        f"{result['validation_score']:.2%}"
    )

    print(
        f"Checks Passed     : "
        f"{result['passed_checks']}/"
        f"{result['total_checks']}"
    )

    print("\nValidation Checks:")

    for name, validation in result["validations"].items():

        status = (
            "PASS"
            if validation["passed"]
            else "FAIL"
        )

        print(
            f"  {name:<35} : {status}"
        )

    print("\n" + "=" * 78)
    print("PHASE 8 CRITIC / VALIDATION TEST COMPLETED")
    print("=" * 78)