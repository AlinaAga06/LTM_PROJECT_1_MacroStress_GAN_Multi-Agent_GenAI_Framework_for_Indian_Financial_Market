"""
MacroStress-GAN
Institutional Daily Risk System

Market Analysis Agent
---------------------
Analyzes the latest market state and classifies the current
market into an operational stress regime.

Important:
This first implementation is intentionally transparent and
rule-based. The existing V3.4 regime_detection.py remains the
reference implementation for the trained TimeGAN conditioning.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from agents.data_agent import DataAgent, FEATURES


class MarketAnalysisAgent:
    """Analyzes current market conditions."""

    def __init__(self, project_root: str | None = None):
        self.data_agent = DataAgent(project_root)

    @staticmethod
    def _safe_zscore(value: float, mean: float, std: float) -> float:
        if std == 0 or np.isnan(std):
            return 0.0

        return float((value - mean) / std)

    def calculate_stress_scores(
        self,
        latest: Dict[str, Any],
        statistics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Calculate standardized stress indicators.

        Direction:
        - Negative NIFTY = stress
        - Positive VIX = stress
        - Positive USD/INR = stress
        - Large absolute crude movement = stress
        - Large absolute yield movement = stress
        """

        scores = {}

        # NIFTY: negative return is stressful.
        nifty_z = self._safe_zscore(
            latest["NIFTY50_Return"],
            statistics["NIFTY50_Return"]["mean"],
            statistics["NIFTY50_Return"]["std"],
        )

        scores["NIFTY_Stress"] = max(0.0, -nifty_z)

        # VIX: positive change is stressful.
        vix_z = self._safe_zscore(
            latest["INDIA_VIX_Change"],
            statistics["INDIA_VIX_Change"]["mean"],
            statistics["INDIA_VIX_Change"]["std"],
        )

        scores["VIX_Stress"] = max(0.0, vix_z)

        # USD/INR: positive movement represents INR depreciation.
        usd_z = self._safe_zscore(
            latest["USD_INR_Return"],
            statistics["USD_INR_Return"]["mean"],
            statistics["USD_INR_Return"]["std"],
        )

        scores["USDINR_Stress"] = max(0.0, usd_z)

        # Crude: large movement is treated as a market shock.
        crude_z = self._safe_zscore(
            latest["CRUDE_OIL_Return"],
            statistics["CRUDE_OIL_Return"]["mean"],
            statistics["CRUDE_OIL_Return"]["std"],
        )

        scores["Crude_Stress"] = abs(crude_z)

        # Yield: large movement is treated as rate stress.
        yield_z = self._safe_zscore(
            latest["INDIA_10Y_YIELD_Change"],
            statistics["INDIA_10Y_YIELD_Change"]["mean"],
            statistics["INDIA_10Y_YIELD_Change"]["std"],
        )

        scores["Yield_Stress"] = abs(yield_z)

        return scores

    @staticmethod
    def classify_regime(stress_score: float) -> str:
        """
        Operational regime classification.

        This is a monitoring layer, not a replacement for the
        V3.4 training-time regime detector.
        """

        if stress_score < 1.0:
            return "Normal"

        if stress_score < 2.0:
            return "Elevated Volatility"

        if stress_score < 3.0:
            return "Market Stress"

        return "Crisis"

    def run(self) -> Dict[str, Any]:
        """Execute market analysis."""

        data_result = self.data_agent.run()

        latest = data_result["latest_market_data"]
        statistics = data_result["recent_statistics"]

        stress_scores = self.calculate_stress_scores(
            latest,
            statistics,
        )

        # Average stress across the five market dimensions.
        raw_score = float(np.mean(list(stress_scores.values())))

        regime = self.classify_regime(raw_score)

        return {
            "agent": "MarketAnalysisAgent",
            "status": "success",
            "latest_market_data": latest,
            "stress_scores": stress_scores,
            "overall_stress_score": raw_score,
            "regime": regime,
        }


if __name__ == "__main__":
    agent = MarketAnalysisAgent()

    result = agent.run()

    print("=" * 70)
    print("MACROSTRESS-GAN MARKET ANALYSIS AGENT")
    print("=" * 70)

    print("\nStatus:", result["status"])
    print("Regime:", result["regime"])
    print(
        "Overall Stress Score:",
        round(result["overall_stress_score"], 4),
    )

    print("\nStress Scores:")

    for key, value in result["stress_scores"].items():
        print(f"{key}: {value:.4f}")