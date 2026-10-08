"""
MacroStress-GAN
Institutional Daily Risk System

Data Agent
----------
Responsible for:
- Loading the processed five-factor financial dataset
- Validating required columns
- Extracting the latest market observation
- Calculating recent market statistics
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


class DataAgent:
    """Loads and validates daily financial market data."""

    def __init__(self, project_root: str | None = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)

        self.data_path = (
            self.project_root
            / "data"
            / "processed"
            / "macro_stress_5vars_processed.csv"
        )

    def load_data(self) -> pd.DataFrame:
        """Load the processed five-variable dataset."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found:\n{self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        missing = [col for col in FEATURES if col not in df.columns]

        if missing:
            raise ValueError(
                f"Required market features are missing: {missing}"
            )

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset quality."""

        missing_values = df[FEATURES].isna().sum().to_dict()

        numeric_status = {
            col: bool(pd.api.types.is_numeric_dtype(df[col]))
            for col in FEATURES
        }

        return {
            "rows": int(len(df)),
            "features": FEATURES,
            "missing_values": missing_values,
            "total_missing": int(sum(missing_values.values())),
            "numeric_columns": numeric_status,
            "valid": (
                len(df) > 0
                and sum(missing_values.values()) == 0
                and all(numeric_status.values())
            ),
        }

    def get_latest_observation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return the latest available market observation."""

        latest = df.iloc[-1]

        result = {}

        for feature in FEATURES:
            value = latest[feature]

            result[feature] = (
                None if pd.isna(value) else float(value)
            )

        # Try to identify a date column if one exists.
        date_column = None

        for candidate in ["Date", "date", "Datetime", "datetime"]:
            if candidate in df.columns:
                date_column = candidate
                break

        if date_column:
            result["date"] = str(latest[date_column])
        else:
            result["date"] = None

        return result

    def get_recent_window(
        self,
        df: pd.DataFrame,
        window: int = 30,
    ) -> pd.DataFrame:
        """Return the latest rolling window."""

        if window <= 0:
            raise ValueError("window must be greater than zero")

        return df.tail(window).copy()

    def calculate_recent_statistics(
        self,
        df: pd.DataFrame,
        window: int = 30,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate recent mean, std, min and max."""

        recent = self.get_recent_window(df, window)

        statistics = {}

        for feature in FEATURES:
            series = pd.to_numeric(
                recent[feature],
                errors="coerce",
            ).dropna()

            statistics[feature] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        return statistics

    def run(self) -> Dict[str, Any]:
        """Execute the complete Data Agent workflow."""

        df = self.load_data()

        validation = self.validate_data(df)

        if not validation["valid"]:
            raise ValueError(
                f"Dataset validation failed: {validation}"
            )

        latest = self.get_latest_observation(df)

        statistics = self.calculate_recent_statistics(
            df,
            window=30,
        )

        return {
            "agent": "DataAgent",
            "status": "success",
            "dataset": str(self.data_path),
            "validation": validation,
            "latest_market_data": latest,
            "recent_statistics": statistics,
        }


if __name__ == "__main__":
    agent = DataAgent()

    result = agent.run()

    print("=" * 70)
    print("MACROSTRESS-GAN DATA AGENT")
    print("=" * 70)

    print("\nStatus:", result["status"])
    print("Rows:", result["validation"]["rows"])
    print("Missing:", result["validation"]["total_missing"])

    print("\nLatest Market Data:")

    for key, value in result["latest_market_data"].items():
        print(f"{key}: {value}")