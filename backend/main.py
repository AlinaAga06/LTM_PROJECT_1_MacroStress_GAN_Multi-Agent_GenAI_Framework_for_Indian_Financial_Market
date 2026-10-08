from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PHASE 9 ORCHESTRATOR
# ============================================================================

from agents.orchestrator_agent import OrchestratorAgent


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="MacroStress-GAN API",
    description="Phase 10 API for Indian Financial Market Stress Testing",
    version="2.2.0"
)


# ============================================================================
# REQUEST MODEL
# ============================================================================

class StressTestRequest(BaseModel):

    scenario_name: str = Field(
        default="Geopolitical Crisis",
        min_length=1,
        max_length=200,
        description="Name of the stress scenario"
    )

    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Stress-test horizon in days"
    )

    num_scenarios: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of simulation scenarios"
    )

    regime: str = Field(
        default="Market Stress",
        min_length=1,
        max_length=100,
        description="Market regime"
    )

    # ------------------------------------------------------------------------
    # MARKET SHOCKS
    # ------------------------------------------------------------------------

    nifty_shock_pct: float = Field(
        default=-15.0,
        ge=-100.0,
        le=100.0,
        description="NIFTY50 percentage shock"
    )

    crude_shock_pct: float = Field(
        default=30.0,
        ge=-100.0,
        le=100.0,
        description="Crude oil percentage shock"
    )

    usdinr_shock_pct: float = Field(
        default=5.0,
        ge=-100.0,
        le=100.0,
        description="USD/INR percentage shock"
    )

    vix_shock_pct: float = Field(
        default=50.0,
        ge=-100.0,
        le=100.0,
        description="India VIX percentage shock"
    )

    yield_shock_pct: float = Field(
        default=1.0,
        ge=-100.0,
        le=100.0,
        description="India 10Y yield shock"
    )

    # ------------------------------------------------------------------------
    # INSTITUTIONAL PORTFOLIO
    # ------------------------------------------------------------------------

    portfolio_amount: float = Field(
        default=100000000.0,
        gt=0,
        description="Institutional portfolio amount in INR"
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=10,
        description="Portfolio currency"
    )


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert NumPy, Pandas and other non-JSON-native
    objects into standard Python objects.
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())

    if isinstance(obj, pd.DataFrame):
        return make_json_safe(
            obj.to_dict(orient="records")
        )

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.tolist())

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in obj
        ]

    if hasattr(obj, "__dict__"):
        return make_json_safe(vars(obj))

    return str(obj)


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
def root():

    return {
        "message": "MacroStress-GAN API is running",
        "status": "connected",
        "version": "2.2.0",
        "phase": "Phase 10",
        "portfolio_monetary_risk": True
    }


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
def health_check():

    return {
        "status": "connected",
        "service": "MacroStress-GAN Backend",
        "version": "2.2.0",
        "phase": "Phase 10",
        "portfolio_monetary_risk": True,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# API INFORMATION
# ============================================================================

@app.get("/api/info")
def api_info():

    return {
        "service": "MacroStress-GAN",
        "phase": "Phase 10",
        "version": "2.2.0",

        "pipeline": [
            "Scenario Agent",
            "Simulation Agent",
            "Risk Agent",
            "Monetary Risk Calculator",
            "Critic / Validation Agent",
            "Report Agent"
        ],

        "scenario_inputs": [
            "scenario_name",
            "horizon_days",
            "num_scenarios",
            "regime",
            "nifty_shock_pct",
            "crude_shock_pct",
            "usdinr_shock_pct",
            "vix_shock_pct",
            "yield_shock_pct"
        ],

        "institutional_inputs": [
            "portfolio_amount",
            "currency"
        ],

        "endpoints": {
            "root": "/",
            "health": "/health",
            "info": "/api/info",
            "status": "/api/status",
            "stress_test": "/api/stress-test"
        }
    }


# ============================================================================
# STRESS TEST
# ============================================================================

@app.post("/api/stress-test")
def run_stress_test(request: StressTestRequest):

    print()
    print("=" * 80)
    print("FASTAPI - PHASE 10 INSTITUTIONAL STRESS TEST")
    print("=" * 80)

    print(f"Scenario          : {request.scenario_name}")
    print(f"Horizon           : {request.horizon_days} days")
    print(f"Scenarios         : {request.num_scenarios}")
    print(f"Regime            : {request.regime}")

    print()
    print("Portfolio:")
    print(f"  Amount          : {request.portfolio_amount:,.2f}")
    print(f"  Currency        : {request.currency}")

    print()
    print("Market Shocks:")
    print(f"  NIFTY50         : {request.nifty_shock_pct}%")
    print(f"  CRUDE OIL       : {request.crude_shock_pct}%")
    print(f"  USD/INR         : {request.usdinr_shock_pct}%")
    print(f"  INDIA VIX       : {request.vix_shock_pct}%")
    print(f"  10Y YIELD       : {request.yield_shock_pct}%")

    print("=" * 80)

    try:

        # ====================================================================
        # CREATE ORCHESTRATOR
        # ====================================================================

        orchestrator = OrchestratorAgent()

        # ====================================================================
        # RUN COMPLETE PHASE 9 PIPELINE
        # ====================================================================

        result = orchestrator.run(

            # Simulation configuration
            horizon_days=request.horizon_days,
            num_scenarios=request.num_scenarios,
            regime=request.regime,

            # Scenario configuration
            scenario_name=request.scenario_name,

            nifty_shock_pct=request.nifty_shock_pct,
            crude_shock_pct=request.crude_shock_pct,
            usdinr_shock_pct=request.usdinr_shock_pct,
            vix_shock_pct=request.vix_shock_pct,
            yield_shock_pct=request.yield_shock_pct,

            # =================================================================
            # INSTITUTIONAL PORTFOLIO
            # =================================================================

            portfolio_amount=request.portfolio_amount,
            currency=request.currency
        )

        # ====================================================================
        # JSON SAFE CONVERSION
        # ====================================================================

        safe_result = make_json_safe(result)

        # ====================================================================
        # COMPLETION LOG
        # ====================================================================

        print()
        print("=" * 80)
        print("FASTAPI - STRESS TEST COMPLETED")
        print("=" * 80)

        print("Custom scenario successfully passed through:")
        print()
        print("  Scenario Agent")
        print("       ↓")
        print("  Simulation Agent / TimeGAN")
        print("       ↓")
        print("  Risk Agent")
        print("       ↓")
        print("  Monetary Risk Calculator")
        print("       ↓")
        print("  Critic / Validation Agent")
        print("       ↓")
        print("  Report Agent")

        print()
        print("Portfolio monetary risk calculated successfully.")
        print("Result converted to JSON-safe format successfully.")

        print("=" * 80)

        return {

            "success": True,

            "message": "Institutional stress test completed successfully",

            "timestamp": datetime.now().isoformat(),

            # ----------------------------------------------------------------
            # REQUEST
            # ----------------------------------------------------------------

            "request": {

                "scenario_name": request.scenario_name,

                "horizon_days": request.horizon_days,

                "num_scenarios": request.num_scenarios,

                "regime": request.regime,

                "nifty_shock_pct":
                    request.nifty_shock_pct,

                "crude_shock_pct":
                    request.crude_shock_pct,

                "usdinr_shock_pct":
                    request.usdinr_shock_pct,

                "vix_shock_pct":
                    request.vix_shock_pct,

                "yield_shock_pct":
                    request.yield_shock_pct,

                # Portfolio
                "portfolio_amount":
                    request.portfolio_amount,

                "currency":
                    request.currency
            },

            # ----------------------------------------------------------------
            # PIPELINE RESULT
            # ----------------------------------------------------------------

            "result": safe_result
        }

    except Exception as exc:

        print()
        print("=" * 80)
        print("FASTAPI - STRESS TEST FAILED")
        print("=" * 80)

        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")

        print("=" * 80)

        raise HTTPException(

            status_code=500,

            detail={

                "message":
                    "Stress test execution failed",

                "error":
                    str(exc),

                "error_type":
                    type(exc).__name__
            }
        )


# ============================================================================
# STATUS
# ============================================================================

@app.get("/api/status")
def api_status():

    return {

        "status": "ready",

        "service": "MacroStress-GAN",

        "phase": "Phase 10",

        "version": "2.2.0",

        "orchestrator": "Phase 9",

        "custom_scenario": True,

        "portfolio_input": True,

        "monetary_risk": True,

        "timegan": True,

        "risk_analysis": True,

        "timestamp": datetime.now().isoformat()
    }