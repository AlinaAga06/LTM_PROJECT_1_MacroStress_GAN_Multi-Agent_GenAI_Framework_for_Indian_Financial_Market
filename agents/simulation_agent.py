"""
MacroStress-GAN
Stock Market TimeGAN V2 Simulation Agent

Compatible with:

    from agents.simulation_agent import SimulationAgent

Main implementation:

    StockTimeGANV2SimulationAgent

Pipeline:

    Scenario -> Daily Feature Shifts -> TimeGAN V2 -> Raw Scenario Injection
    -> Scenario Propagation Validation -> Stressed Synthetic Paths
    -> (CSV export for the dashboard) -> Downstream Risk Analysis
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "timegan" / "stock" / "timegan_stock_v2.pt"

SCALER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "institutions" / "stock" / "stock_timegan_scaler.pkl"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# CSV read by the Streamlit dashboard
SCENARIO_CSV_PATH = OUTPUT_DIR / "synthetic" / "stock" / "stock_v2_synthetic_financial_data.csv"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_VERSION = "TimeGAN_Stock_V2"

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

FEATURE_DIM = 5
SEQ_LEN = 30
HIDDEN_DIM = 24
NUM_LAYERS = 2


# =============================================================================
# DEFAULT SCENARIO
# =============================================================================

DEFAULT_SCENARIO = {
    "name": "Geopolitical Crisis",
    "scenario_name": "Geopolitical Crisis",
    "horizon_days": 30,
    "NIFTY50_shock_pct": -0.15,
    "CRUDE_OIL_shock_pct": 0.30,
    "USD_INR_shock_pct": 0.05,
    "INDIA_VIX_shock_pct": 0.50,
    # +1 percentage point
    "INDIA_10Y_YIELD_shock_pct": 0.01,
    "market_shocks": {
        "NIFTY50": -15.0,
        "CRUDE_OIL": 30.0,
        "USD_INR": 5.0,
        "INDIA_VIX": 50.0,
        "INDIA_10Y_YIELD": 1.0,
    },
}


# =============================================================================
# DEFAULT ASSETS
# =============================================================================

DEFAULT_ASSETS = [
    {
        "name": "Asset_A",
        "weight": 0.40,
        "allocation": 0.40,
        "portfolio_weight": 0.40,
        "sensitivities": {"NIFTY50": 0.80, "USD_INR": -0.10, "INDIA_VIX": -0.10},
        "exposures": {"NIFTY50": 0.80, "USD_INR": -0.10, "INDIA_VIX": -0.10},
    },
    {
        "name": "Asset_B",
        "weight": 0.35,
        "allocation": 0.35,
        "portfolio_weight": 0.35,
        "sensitivities": {"NIFTY50": 0.60, "CRUDE_OIL": -0.20, "USD_INR": -0.20},
        "exposures": {"NIFTY50": 0.60, "CRUDE_OIL": -0.20, "USD_INR": -0.20},
    },
    {
        "name": "Asset_C",
        "weight": 0.25,
        "allocation": 0.25,
        "portfolio_weight": 0.25,
        "sensitivities": {"NIFTY50": 0.40, "INDIA_10Y_YIELD": -0.60},
        "exposures": {"NIFTY50": 0.40, "INDIA_10Y_YIELD": -0.60},
    },
]


# =============================================================================
# DEFAULT CONSTRAINTS
# =============================================================================

DEFAULT_CONSTRAINTS = {
    "min_weight": 0.05,
    "max_weight": 0.60,
    "minimum_weight": 0.05,
    "maximum_weight": 0.60,
}


# =============================================================================
# TIMEGAN NETWORKS
# =============================================================================

class Generator(nn.Module):
    """TimeGAN Generator: [batch, seq, feature_dim] -> [batch, seq, hidden_dim]."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.activation(self.fc(hidden))


class Supervisor(nn.Module):
    """TimeGAN Supervisor."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.activation(self.fc(hidden))


class Recovery(nn.Module):
    """TimeGAN Recovery network: hidden space -> feature space."""

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.activation(self.fc(hidden))


# =============================================================================
# STOCK TIMEGAN V2 SIMULATION AGENT
# =============================================================================

class StockTimeGANV2SimulationAgent:
    """Stock Market TimeGAN V2 Simulation Agent."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        scaler_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path is not None else SCALER_PATH

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = None
        self.generator = None
        self.supervisor = None
        self.recovery = None
        self.model_loaded = False

        self._load_scaler()
        self._load_model()

    # -------------------------------------------------------------------------
    # LOAD SCALER
    # -------------------------------------------------------------------------

    def _load_scaler(self) -> None:
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found:\n{self.scaler_path}")

        with open(self.scaler_path, "rb") as file:
            self.scaler = pickle.load(file)

        print(f"Scaler loaded: {self.scaler_path}")

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"TimeGAN model not found:\n{self.model_path}")

        self.generator = Generator(
            input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        ).to(self.device)

        self.supervisor = Supervisor(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        ).to(self.device)

        self.recovery = Recovery(
            hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=NUM_LAYERS
        ).to(self.device)

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        self._load_checkpoint(checkpoint)

        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        self.model_loaded = True

        print(f"Model loaded: {self.model_path}")
        print(f"Model version: {MODEL_VERSION}")
        print(f"Device: {self.device}")

    # -------------------------------------------------------------------------
    # CHECKPOINT LOADER
    # -------------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Unsupported TimeGAN checkpoint format.")

        def first_present(keys):
            for key in keys:
                if key in checkpoint:
                    return checkpoint[key]
            return None

        generator_state = first_present(
            ["generator", "generator_state_dict", "G", "G_state_dict"])
        supervisor_state = first_present(
            ["supervisor", "supervisor_state_dict", "S", "S_state_dict"])
        recovery_state = first_present(
            ["recovery", "recovery_state_dict", "R", "R_state_dict"])

        if generator_state is None:
            raise RuntimeError("Generator weights not found in TimeGAN checkpoint.")
        if supervisor_state is None:
            raise RuntimeError("Supervisor weights not found in TimeGAN checkpoint.")
        if recovery_state is None:
            raise RuntimeError("Recovery weights not found in TimeGAN checkpoint.")

        self.generator.load_state_dict(generator_state)
        self.supervisor.load_state_dict(supervisor_state)
        self.recovery.load_state_dict(recovery_state)

    # -------------------------------------------------------------------------
    # GENERATE SCALED DATA
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate_scaled(
        self,
        num_scenarios: int = 1000,
        seq_len: int = SEQ_LEN,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if not self.model_loaded:
            raise RuntimeError("TimeGAN model has not been loaded.")

        noise = torch.rand(num_scenarios, seq_len, FEATURE_DIM, device=self.device)

        generated = self.generator(noise)
        generated = self.supervisor(generated)
        generated = self.recovery(generated)
        generated = torch.clamp(generated, 0.0, 1.0)

        return generated.cpu().numpy()

    # -------------------------------------------------------------------------
    # INVERSE TRANSFORM / TRANSFORM
    # -------------------------------------------------------------------------

    def _inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        original_shape = scaled_data.shape
        flat_data = scaled_data.reshape(-1, FEATURE_DIM)
        raw_data = self.scaler.inverse_transform(flat_data)
        return raw_data.reshape(original_shape)

    def _transform(self, raw_data: np.ndarray) -> np.ndarray:
        original_shape = raw_data.shape
        flat_data = raw_data.reshape(-1, FEATURE_DIM)
        scaled_data = self.scaler.transform(flat_data)
        return scaled_data.reshape(original_shape)

    # -------------------------------------------------------------------------
    # EXTRACT SHOCKS
    # -------------------------------------------------------------------------

    def _extract_shocks(
        self,
        scenario: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Resolve market shocks from different ScenarioAgent formats.

        Internal representation:

            NIFTY50            -0.15
            CRUDE_OIL          +0.30
            USD_INR            +0.05
            INDIA_VIX          +0.50
            INDIA_10Y_YIELD    +0.01

        Examples:
            NIFTY50 = -15.0         -> -0.15
            USD_INR = 5.0           -> +0.05
            INDIA_VIX = 50.0        -> +0.50
            INDIA_10Y_YIELD = 1.0   -> +0.01
            INDIA_10Y_YIELD = 0.01  -> +0.01
        """

        result = {
            "NIFTY50": 0.0,
            "CRUDE_OIL": 0.0,
            "USD_INR": 0.0,
            "INDIA_VIX": 0.0,
            "INDIA_10Y_YIELD": 0.0,
        }

        aliases = {
            "NIFTY50": ["NIFTY50", "NIFTY", "NIFTY_50", "NIFTY50_SHOCK", "NIFTY50_SHOCK_PCT"],
            "CRUDE_OIL": ["CRUDE_OIL", "CRUDE", "BRENT", "CRUDE_OIL_SHOCK", "CRUDE_OIL_SHOCK_PCT"],
            "USD_INR": ["USD_INR", "USDINR", "USD/INR", "USD_INR_SHOCK", "USD_INR_SHOCK_PCT"],
            "INDIA_VIX": ["INDIA_VIX", "VIX", "INDIA_VIX_SHOCK", "INDIA_VIX_SHOCK_PCT"],
            "INDIA_10Y_YIELD": [
                "INDIA_10Y_YIELD", "INDIA_10Y", "10Y_YIELD",
                "INDIA_10Y_GOVERNMENT_BOND_YIELD",
                "INDIA_10Y_YIELD_SHOCK", "INDIA_10Y_YIELD_SHOCK_PCT",
            ],
        }

        def normalize_value(variable: str, value: Any) -> float:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return 0.0

            # INDIA 10Y YIELD: 1.0 -> 0.01, 0.5 -> 0.005, 0.01 -> 0.01
            if variable == "INDIA_10Y_YIELD":
                if abs(numeric_value) >= 0.1:
                    return numeric_value / 100.0
                return numeric_value

            # Other variables
            if abs(numeric_value) > 1.0:
                return numeric_value / 100.0

            return numeric_value

        def recursive_search(obj: Any) -> None:
            if not isinstance(obj, dict):
                return

            for key, value in obj.items():
                key_upper = str(key).strip().upper()
                matched_variable = None

                for variable, variable_aliases in aliases.items():
                    alias_set = {str(alias).strip().upper() for alias in variable_aliases}
                    if key_upper in alias_set:
                        matched_variable = variable
                        break

                if matched_variable is not None:
                    result[matched_variable] = normalize_value(matched_variable, value)

                if isinstance(value, dict):
                    recursive_search(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            recursive_search(item)

        if scenario is not None:
            recursive_search(scenario)

        if kwargs:
            recursive_search(kwargs)

        return result

    # -------------------------------------------------------------------------
    # EQUIVALENT DAILY RETURN
    # -------------------------------------------------------------------------

    @staticmethod
    def _equivalent_daily_return(total_return: float, horizon_days: int) -> float:
        if horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")

        if total_return <= -1.0:
            raise ValueError("total_return must be greater than -100%.")

        daily_return = np.expm1(np.log1p(total_return) / horizon_days)

        return float(daily_return)

    # -------------------------------------------------------------------------
    # BUILD DAILY FEATURE SHIFTS
    # -------------------------------------------------------------------------

    def _build_daily_feature_shifts(
        self,
        shocks: Dict[str, float],
        horizon_days: int,
    ) -> np.ndarray:
        if horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")

        daily = np.zeros(FEATURE_DIM, dtype=np.float64)

        # NIFTY50 / CRUDE OIL / USD_INR returns (compounding-equivalent daily return)
        daily[0] = self._equivalent_daily_return(shocks["NIFTY50"], horizon_days)
        daily[1] = self._equivalent_daily_return(shocks["CRUDE_OIL"], horizon_days)
        daily[2] = self._equivalent_daily_return(shocks["USD_INR"], horizon_days)

        # INDIA VIX
        # Reference VIX = 20; +50% shock -> 20 * 0.50 = +10 VIX points
        # +10 / 30 = +0.3333333 daily
        vix_reference_level = 20.0
        vix_total_point_change = vix_reference_level * shocks["INDIA_VIX"]
        daily[3] = vix_total_point_change / horizon_days

        # INDIA 10Y GOVERNMENT BOND YIELD
        # +1 percentage point = 0.01; 0.01 / 30 = 0.0003333333
        daily[4] = shocks["INDIA_10Y_YIELD"] / horizon_days

        return daily

    # -------------------------------------------------------------------------
    # APPLY RAW SCENARIO
    # -------------------------------------------------------------------------

    def _apply_raw_scenario(
        self,
        baseline_raw: np.ndarray,
        shocks: Dict[str, float],
        horizon_days: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        daily_shifts = self._build_daily_feature_shifts(shocks, horizon_days)

        shift_tensor = daily_shifts.reshape(1, 1, FEATURE_DIM)

        stressed_raw = baseline_raw + shift_tensor

        # IMPORTANT: DO NOT CLIP RAW DATA HERE.
        # Clipping before validation would destroy the intended shock,
        # especially for INDIA_VIX_Change.
        stressed_raw = np.nan_to_num(stressed_raw, nan=0.0, posinf=0.0, neginf=0.0)

        total_feature_shifts = daily_shifts * horizon_days

        diagnostics = {
            "daily_feature_shifts": {
                feature: float(daily_shifts[index]) for index, feature in enumerate(FEATURES)
            },
            "total_feature_shifts": {
                feature: float(total_feature_shifts[index]) for index, feature in enumerate(FEATURES)
            },
            "vix_reference_level": 20.0,
            "vix_total_point_change": float(20.0 * shocks["INDIA_VIX"]),
            "raw_feature_clipping_applied": False,
            "exact_scenario_propagation": True,
        }

        return stressed_raw, diagnostics

    # -------------------------------------------------------------------------
    # VALIDATE SCENARIO PROPAGATION
    # -------------------------------------------------------------------------

    def _validate_scenario_propagation(
        self,
        baseline_raw: np.ndarray,
        stressed_raw: np.ndarray,
        daily_shifts: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        baseline_mean = baseline_raw.mean(axis=(0, 1))
        stressed_mean = stressed_raw.mean(axis=(0, 1))

        actual_delta = stressed_mean - baseline_mean
        expected_delta = daily_shifts

        feature_results = {}
        validation_passed = True

        for index, feature in enumerate(FEATURES):
            actual = float(actual_delta[index])
            expected = float(expected_delta[index])
            absolute_error = abs(actual - expected)
            passed = absolute_error <= tolerance

            if not passed:
                validation_passed = False

            feature_results[feature] = {
                "baseline_mean": float(baseline_mean[index]),
                "stressed_mean": float(stressed_mean[index]),
                "actual_delta": actual,
                "expected_delta": expected,
                "absolute_error": float(absolute_error),
                "tolerance": float(tolerance),
                "passed": bool(passed),
            }

        return {
            "passed": bool(validation_passed),
            "tolerance": float(tolerance),
            "features": feature_results,
        }

    # -------------------------------------------------------------------------
    # SIMULATE
    # -------------------------------------------------------------------------

    def simulate(
        self,
        scenario: Optional[Dict[str, Any]] = None,
        num_scenarios: int = 1000,
        horizon_days: int = 30,
        seed: Optional[int] = 42,
        portfolio_amount: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        # RESOLVE SCENARIO
        if scenario is None:
            scenario = DEFAULT_SCENARIO.copy()

        # RESOLVE HORIZON
        scenario_horizon = scenario.get("horizon_days", horizon_days)

        try:
            horizon_days = int(scenario_horizon)
        except (TypeError, ValueError):
            horizon_days = 30

        if horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")

        # EXTRACT SHOCKS
        shocks = self._extract_shocks(scenario=scenario, **kwargs)

        # SCENARIO NAME (separate variable avoids malformed multiline f-strings)
        scenario_name = scenario.get("name", scenario.get("scenario_name", "Unknown"))

        # HEADER
        print()
        print("=" * 78)
        print("STOCK TIMEGAN V2 SCENARIO")
        print("=" * 78)
        print(f"Scenario : {scenario_name}")
        print(f"Horizon  : {horizon_days} days")

        # TOTAL MARKET SHOCKS
        print()
        print("SCENARIO TOTAL MARKET SHOCKS")
        print(f"NIFTY50           : {shocks['NIFTY50'] * 100:+.2f}%")
        print(f"CRUDE_OIL         : {shocks['CRUDE_OIL'] * 100:+.2f}%")
        print(f"USD_INR           : {shocks['USD_INR'] * 100:+.2f}%")
        print(f"INDIA_VIX         : {shocks['INDIA_VIX'] * 100:+.2f}%")
        print(f"INDIA_10Y_YIELD   : {shocks['INDIA_10Y_YIELD'] * 100:+.2f} percentage points")

        # RESOLVED SCENARIO VALUES
        print()
        print("RESOLVED SCENARIO VALUES")
        for key in ("NIFTY50", "CRUDE_OIL", "USD_INR", "INDIA_VIX", "INDIA_10Y_YIELD"):
            print(f"{key:<20}: {shocks[key]:+.8f}")

        # DAILY SHIFTS
        daily_shifts = self._build_daily_feature_shifts(shocks, horizon_days)
        total_shifts = daily_shifts * horizon_days

        print()
        print("DAILY TIMEGAN FEATURE SHIFTS")
        for index, feature in enumerate(FEATURES):
            print(f"{feature:<24}: {daily_shifts[index]:+.8f}")

        print()
        print("TOTAL TIMEGAN FEATURE SHIFTS")
        for index, feature in enumerate(FEATURES):
            print(f"{feature:<24}: {total_shifts[index]:+.8f}")

        # GENERATE BASELINE
        print()
        print("=" * 78)
        print("GENERATING TIMEGAN V2 BASELINE")
        print("=" * 78)

        baseline_scaled = self.generate_scaled(
            num_scenarios=num_scenarios, seq_len=horizon_days, seed=seed
        )

        # INVERSE SCALE
        baseline_raw = self._inverse_transform(baseline_scaled)

        # APPLY SCENARIO
        stressed_raw, diagnostics = self._apply_raw_scenario(
            baseline_raw, shocks, horizon_days
        )

        # VALIDATION
        validation = self._validate_scenario_propagation(
            baseline_raw=baseline_raw,
            stressed_raw=stressed_raw,
            daily_shifts=daily_shifts,
            tolerance=1e-6,
        )

        # RAW DIFFERENCE
        difference = stressed_raw - baseline_raw

        print()
        print("RAW SCENARIO PROPAGATION")
        print(f"Changed raw values : {np.count_nonzero(difference):,} / {difference.size:,}")
        print(f"Mean absolute shift: {np.mean(np.abs(difference)):.8f}")
        print(f"Maximum raw shift  : {np.max(np.abs(difference)):.8f}")

        # FEATURE VALIDATION OUTPUT
        print()
        print("FEATURE-LEVEL BASELINE VS STRESSED")
        for feature in FEATURES:
            row = validation["features"][feature]
            print(
                f"{feature:<24} "
                f"baseline={row['baseline_mean']:+.8f} "
                f"stressed={row['stressed_mean']:+.8f} "
                f"delta={row['actual_delta']:+.8f} "
                f"expected={row['expected_delta']:+.8f}"
            )

        # VALIDATION FAILURE
        if not validation["passed"]:
            print()
            print("=" * 78)
            print("SCENARIO PROPAGATION VALIDATION FAILED")
            print("=" * 78)

            for feature in FEATURES:
                row = validation["features"][feature]
                if not row["passed"]:
                    print(
                        f"{feature}: "
                        f"actual={row['actual_delta']:.10f}, "
                        f"expected={row['expected_delta']:.10f}, "
                        f"error={row['absolute_error']:.10f}, "
                        f"tolerance={row['tolerance']:.10f}"
                    )

            raise RuntimeError("Scenario propagation validation failed.")

        # VALIDATION SUCCESS
        print()
        print("=" * 78)
        print("SCENARIO PROPAGATION VALIDATION PASSED")
        print("=" * 78)

        # SCALE STRESSED DATA
        stressed_scaled = self._transform(stressed_raw)

        # NUMERICAL CLEANUP
        baseline_scaled = np.nan_to_num(baseline_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        stressed_scaled = np.nan_to_num(stressed_scaled, nan=0.0, posinf=1.0, neginf=0.0)

        # CLIP ONLY SCALED DATA
        baseline_scaled = np.clip(baseline_scaled, 0.0, 1.0)
        stressed_scaled = np.clip(stressed_scaled, 0.0, 1.0)

        # RESULT
        result = {
            "model_version": MODEL_VERSION,
            "simulation_type": "stock_market",
            "feature_names": FEATURES,
            "num_scenarios": int(num_scenarios),
            "horizon_days": int(horizon_days),
            "sequence_length": int(horizon_days),
            "feature_dimension": FEATURE_DIM,
            "seed": seed,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "scenario": {
                "name": scenario_name,
                "scenario_name": scenario_name,
                "horizon_days": int(horizon_days),
                "market_shocks": {key: float(value) for key, value in shocks.items()},
            },
            "daily_feature_shifts": {
                feature: float(daily_shifts[index]) for index, feature in enumerate(FEATURES)
            },
            "total_feature_shifts": {
                feature: float(total_shifts[index]) for index, feature in enumerate(FEATURES)
            },
            "diagnostics": diagnostics,
            "validation": validation,
            "baseline_scaled": baseline_scaled,
            "stressed_scaled": stressed_scaled,
            "baseline_raw": baseline_raw,
            "stressed_raw": stressed_raw,
            "assets": DEFAULT_ASSETS,
            "constraints": DEFAULT_CONSTRAINTS,
            "portfolio_amount": (
                float(portfolio_amount) if portfolio_amount is not None else None
            ),
        }

        return result

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    def run(
        self,
        scenario: Optional[Dict[str, Any]] = None,
        num_scenarios: int = 1000,
        horizon_days: int = 30,
        seed: Optional[int] = 42,
        portfolio_amount: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.simulate(
            scenario=scenario,
            num_scenarios=num_scenarios,
            horizon_days=horizon_days,
            seed=seed,
            portfolio_amount=portfolio_amount,
            **kwargs,
        )


# =============================================================================
# ORCHESTRATOR COMPATIBILITY CLASS
# =============================================================================

class SimulationAgent(StockTimeGANV2SimulationAgent):
    """
    Compatibility wrapper for the existing orchestrator.

    Existing orchestrator import:

        from agents.simulation_agent import SimulationAgent
    """

    def __init__(self, model_path=None, scaler_path=None, device=None, **kwargs):
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            device=device,
            **kwargs,
        )

    def run(
        self,
        scenario=None,
        num_scenarios=1000,
        horizon_days=30,
        seed=42,
        portfolio_amount=None,
        **kwargs,
    ):
        return self.simulate(
            scenario=scenario,
            num_scenarios=num_scenarios,
            horizon_days=horizon_days,
            seed=seed,
            portfolio_amount=portfolio_amount,
            **kwargs,
        )


# =============================================================================
# JSON SAFE CONVERSION
# =============================================================================

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]

    return obj


# =============================================================================
# SCENARIO CSV EXPORT (read by the Streamlit dashboard)
# =============================================================================

def save_scenarios_csv(result: Dict[str, Any], path: Path) -> Path:
    """
    Write the stressed TimeGAN paths as a flat CSV:

        scenario, day, NIFTY50_Return, CRUDE_OIL_Return, USD_INR_Return,
        INDIA_VIX_Change, INDIA_10Y_YIELD_Change
    """
    path = Path(path)

    arr = np.asarray(result["stressed_raw"], dtype=np.float64)   # (scenarios, days, 5)
    s, d, f = arr.shape

    df = pd.DataFrame(arr.reshape(-1, f), columns=FEATURES)
    df.insert(0, "scenario", np.repeat(np.arange(s), d))
    df.insert(1, "day", np.tile(np.arange(d), s))

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return path


# =============================================================================
# STANDALONE TEST
# =============================================================================

def main() -> None:
    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK TIMEGAN V2 SIMULATION AGENT TEST")
    print("=" * 78)
    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Scaler       : {SCALER_PATH}")

    # TEST SCENARIO
    test_scenario = {
        "scenario_name": "Geopolitical Crisis",
        "horizon_days": 30,
        "market_shocks": {
            "NIFTY50": -15.0,
            "CRUDE_OIL": 30.0,
            "USD_INR": 5.0,
            "INDIA_VIX": 50.0,
            "INDIA_10Y_YIELD": 1.0,
        },
    }

    # INITIALIZE
    agent = SimulationAgent()

    # RUN
    result = agent.run(
        scenario=test_scenario,
        num_scenarios=1000,
        horizon_days=30,
        seed=42,
    )

    # CREATE OUTPUT DIRECTORY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # METADATA
    metadata = {
        key: result[key]
        for key in (
            "model_version", "simulation_type", "feature_names", "num_scenarios",
            "horizon_days", "sequence_length", "feature_dimension", "seed", "device",
            "model_path", "scaler_path", "scenario", "daily_feature_shifts",
            "total_feature_shifts", "diagnostics", "validation",
        )
    }

    metadata_path = OUTPUT_DIR / "stock_simulation_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(metadata), file, indent=2)

    # SCENARIO CSV FOR THE DASHBOARD
    csv_path = save_scenarios_csv(result, SCENARIO_CSV_PATH)

    # FINAL OUTPUT
    print()
    print("=" * 78)
    print("SIMULATION COMPLETE")
    print("=" * 78)
    print()
    print("Metadata saved to:")
    print(metadata_path)
    print()
    print("Scenario CSV saved to:")
    print(csv_path)
    print()
    print("FINAL VALIDATION:")
    print(f"  Passed : {result['validation']['passed']}")
    print()
    print("MODEL:")
    print(f"  Version   : {result['model_version']}")
    print(f"  Scenarios : {result['num_scenarios']}")
    print(f"  Horizon   : {result['horizon_days']} days")
    print(f"  Features  : {result['feature_dimension']}")
    print()
    print("TOTAL FEATURE SHIFTS:")

    for feature in FEATURES:
        value = result["total_feature_shifts"][feature]
        print(f"  {feature:<24}: {value:+.8f}")

    print()
    print("=" * 78)
    print("STOCK TIMEGAN V2 TEST PASSED")
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()