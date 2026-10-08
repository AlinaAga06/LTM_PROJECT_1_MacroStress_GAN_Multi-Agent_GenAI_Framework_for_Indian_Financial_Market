"""
MacroStress-GAN
Institutional Daily Risk System

SCENARIO AGENT
==============

Creates, validates, standardizes and loads stress-testing scenarios.

Supported scenario sources:

1. Direct Python input
2. Dictionary configuration
3. JSON files
4. CSV files

Core market variables:

    NIFTY50
    CRUDE_OIL
    USD_INR
    INDIA_VIX
    INDIA_10Y_YIELD

Institution-specific variables can be added separately:

    Credit_Spread
    Deposit_Outflow
    NPA_Ratio
    Loan_Default_Rate
    Liquidity_Ratio
    Corporate_Bond_Yield
    etc.

IMPORTANT
---------
Institution-specific variables are NOT added to TimeGAN V3.4.

The current TimeGAN V3.4 checkpoint expects:

    FEATURE_DIM = 5

Therefore the five market variables remain the TimeGAN variables,
while institution-specific variables are carried as structured
scenario information for downstream institutional analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json


class ScenarioAgent:
    """
    Scenario creation and validation agent.

    This agent does NOT generate synthetic data.

    Responsibilities:

        Input
          ↓
        Validate
          ↓
        Standardize
          ↓
        Return structured scenario
    """

    # ================================================================
    # CORE MARKET VARIABLES
    # ================================================================

    CORE_MARKET_VARIABLES = [
        "NIFTY50",
        "CRUDE_OIL",
        "USD_INR",
        "INDIA_VIX",
        "INDIA_10Y_YIELD",
    ]

    # ================================================================
    # TIMEGAN FEATURES
    # ================================================================

    TIMEGAN_FEATURES = [
        "NIFTY50_Return",
        "CRUDE_OIL_Return",
        "USD_INR_Return",
        "INDIA_VIX_Change",
        "INDIA_10Y_YIELD_Change",
    ]

    # ================================================================
    # ALLOWED SHOCK TYPES
    # ================================================================

    ALLOWED_SHOCK_TYPES = {
        "cumulative_pct",
        "percentage_points",
        "bps",
        "absolute",
        "daily_pct",
    }

    # ================================================================
    # DEFAULTS
    # ================================================================

    DEFAULTS = {
        "scenario_name": "Institutional Stress Test",
        "horizon_days": 30,

        "nifty_shock_pct": 0.0,
        "crude_shock_pct": 0.0,
        "usdinr_shock_pct": 0.0,
        "vix_shock_pct": 0.0,
        "yield_shock_pct": 0.0,
    }

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(self):

        self.agent_name = "ScenarioAgent"

        self.version = "Phase9-Institutional-Scenario"

    # ================================================================
    # NORMALIZE SHOCK TYPE
    # ================================================================

    @classmethod
    def _normalize_shock_type(
        cls,
        shock_type: Any,
    ) -> str:

        if shock_type is None:
            return "absolute"

        value = str(
            shock_type
        ).strip().lower()

        aliases = {
            "pct": "cumulative_pct",
            "%": "cumulative_pct",
            "percent": "cumulative_pct",
            "percentage": "cumulative_pct",

            "percentage_point": "percentage_points",
            "percentage-points": "percentage_points",
            "pp": "percentage_points",

            "basis_points": "bps",
            "basis-point": "bps",
            "basis-points": "bps",

            "daily_percent": "daily_pct",
            "daily_percentage": "daily_pct",
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in cls.ALLOWED_SHOCK_TYPES:

            raise ValueError(
                f"Unsupported shock_type '{shock_type}'. "
                f"Allowed values: "
                f"{sorted(cls.ALLOWED_SHOCK_TYPES)}"
            )

        return value

    # ================================================================
    # NORMALIZE INSTITUTION VARIABLE
    # ================================================================

    @classmethod
    def _normalize_institution_variable(
        cls,
        variable: Dict[str, Any],
        index: int = 0,
    ) -> Dict[str, Any]:
        """
        Normalize one institution-specific variable.
        """

        if not isinstance(
            variable,
            dict,
        ):

            raise TypeError(
                f"Institution variable #{index + 1} "
                "must be a dictionary."
            )

        name = str(
            variable.get(
                "name",
                "",
            )
        ).strip()

        if not name:

            raise ValueError(
                f"Institution variable #{index + 1} "
                "is missing 'name'."
            )

        # ------------------------------------------------------------
        # Prevent core market variables from being duplicated here.
        # ------------------------------------------------------------

        normalized_core = {
            item.upper()
            for item in cls.CORE_MARKET_VARIABLES
        }

        if name.upper() in normalized_core:

            raise ValueError(
                f"'{name}' is a core market variable. "
                "Place it inside market_shocks."
            )

        shock = variable.get(
            "shock",
            0.0,
        )

        try:
            shock = float(shock)

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"Invalid shock value for "
                f"institution variable '{name}': "
                f"{shock}"
            )

        shock_type = cls._normalize_shock_type(
            variable.get(
                "shock_type",
                variable.get(
                    "type",
                    "absolute",
                ),
            )
        )

        result = {
            "name": name,
            "shock": shock,
            "shock_type": shock_type,
            "unit": str(
                variable.get(
                    "unit",
                    "",
                )
            ),
            "direction": str(
                variable.get(
                    "direction",
                    "",
                )
            ),
            "description": str(
                variable.get(
                    "description",
                    "",
                )
            ),
        }

        # ------------------------------------------------------------
        # Preserve optional institution-specific fields.
        # ------------------------------------------------------------

        optional_fields = [
            "baseline",
            "target",
            "min",
            "max",
            "exposure",
            "impact_model",
            "source",
        ]

        for field in optional_fields:

            if field in variable:

                result[field] = variable[field]

        return result

    # ================================================================
    # NORMALIZE INSTITUTION VARIABLES
    # ================================================================

    @classmethod
    def _normalize_institution_variables(
        cls,
        institution_variables: Optional[
            List[Dict[str, Any]]
        ],
    ) -> List[Dict[str, Any]]:

        if institution_variables is None:
            return []

        if not isinstance(
            institution_variables,
            list,
        ):

            raise TypeError(
                "institution_variables must be "
                "a list of dictionaries."
            )

        normalized = []

        names_seen = set()

        for index, variable in enumerate(
            institution_variables
        ):

            item = cls._normalize_institution_variable(
                variable,
                index=index,
            )

            name_key = item["name"].lower()

            if name_key in names_seen:

                raise ValueError(
                    f"Duplicate institution variable: "
                    f"{item['name']}"
                )

            names_seen.add(
                name_key
            )

            normalized.append(
                item
            )

        return normalized

    # ================================================================
    # NORMALIZE MARKET SHOCKS
    # ================================================================

    @classmethod
    def _normalize_market_shocks(
        cls,
        market_shocks: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[str, float]:

        if market_shocks is None:

            return {
                "NIFTY50": 0.0,
                "CRUDE_OIL": 0.0,
                "USD_INR": 0.0,
                "INDIA_VIX": 0.0,
                "INDIA_10Y_YIELD": 0.0,
            }

        if not isinstance(
            market_shocks,
            dict,
        ):

            raise TypeError(
                "market_shocks must be a dictionary."
            )

        normalized = {
            "NIFTY50": 0.0,
            "CRUDE_OIL": 0.0,
            "USD_INR": 0.0,
            "INDIA_VIX": 0.0,
            "INDIA_10Y_YIELD": 0.0,
        }

        # Case-insensitive lookup.
        key_map = {
            key.upper(): key
            for key in market_shocks.keys()
        }

        for core_name in cls.CORE_MARKET_VARIABLES:

            source_key = key_map.get(
                core_name.upper()
            )

            if source_key is None:
                continue

            value = market_shocks[
                source_key
            ]

            try:

                normalized[
                    core_name
                ] = float(value)

            except (
                TypeError,
                ValueError,
            ):

                raise ValueError(
                    f"Invalid market shock for "
                    f"{core_name}: {value}"
                )

        return normalized

    # ================================================================
    # BUILD SCENARIO
    # ================================================================

    @classmethod
    def _build_scenario(
        cls,
        scenario_name: str,
        horizon_days: int,
        market_shocks: Dict[str, Any],
        institution_variables: Optional[
            List[Dict[str, Any]]
        ] = None,
        source: str = "direct",
    ) -> Dict[str, Any]:
        """
        Build final standardized scenario.
        """

        # ------------------------------------------------------------
        # Validate horizon.
        # ------------------------------------------------------------

        try:

            horizon_days = int(
                horizon_days
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "horizon_days must be an integer."
            )

        if horizon_days <= 0:

            raise ValueError(
                "horizon_days must be greater than zero."
            )

        # ------------------------------------------------------------
        # Scenario name.
        # ------------------------------------------------------------

        scenario_name = str(
            scenario_name
        ).strip()

        if not scenario_name:

            scenario_name = (
                "Institutional Stress Test"
            )

        # ------------------------------------------------------------
        # Market shocks.
        # ------------------------------------------------------------

        normalized_market = (
            cls._normalize_market_shocks(
                market_shocks
            )
        )

        # ------------------------------------------------------------
        # Institution variables.
        # ------------------------------------------------------------

        normalized_institution = (
            cls._normalize_institution_variables(
                institution_variables
            )
        )

        # ------------------------------------------------------------
        # Final standardized scenario.
        # ------------------------------------------------------------

        scenario = {
            "scenario_name": scenario_name,

            "horizon_days": horizon_days,

            "market_shocks": normalized_market,

            "institution_variables": (
                normalized_institution
            ),

            "institution_variable_count": len(
                normalized_institution
            ),

            "timegan_features": (
                cls.TIMEGAN_FEATURES.copy()
            ),

            "timegan_feature_count": 5,

            "scenario_source": source,

            "scenario_engine": (
                "TimeGAN V3.4 + "
                "Institution Custom Scenario Layer"
            ),

            # --------------------------------------------------------
            # Backward-compatible keys.
            # --------------------------------------------------------

            "NIFTY50_shock_pct": normalized_market[
                "NIFTY50"
            ],

            "CRUDE_OIL_shock_pct": normalized_market[
                "CRUDE_OIL"
            ],

            "USD_INR_shock_pct": normalized_market[
                "USD_INR"
            ],

            "INDIA_VIX_shock_pct": normalized_market[
                "INDIA_VIX"
            ],

            "INDIA_10Y_YIELD_shock_pct": normalized_market[
                "INDIA_10Y_YIELD"
            ],
        }

        return scenario

    # ================================================================
    # CREATE SCENARIO
    # ================================================================

    def create_scenario(
        self,
        scenario_name: str = "Institutional Stress Test",
        horizon_days: int = 30,

        nifty_shock_pct: float = 0.0,
        crude_shock_pct: float = 0.0,
        usdinr_shock_pct: float = 0.0,
        vix_shock_pct: float = 0.0,
        yield_shock_pct: float = 0.0,

        institution_variables: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Create a scenario directly.

        Example:

            agent.create_scenario(
                scenario_name="Banking Stress",
                horizon_days=30,
                nifty_shock_pct=-15,
                crude_shock_pct=30,
                usdinr_shock_pct=5,
                vix_shock_pct=50,
                yield_shock_pct=1,
                institution_variables=[
                    {
                        "name": "Credit_Spread",
                        "shock": 150,
                        "shock_type": "bps",
                        "unit": "bps"
                    }
                ]
            )
        """

        market_shocks = {
            "NIFTY50": nifty_shock_pct,
            "CRUDE_OIL": crude_shock_pct,
            "USD_INR": usdinr_shock_pct,
            "INDIA_VIX": vix_shock_pct,
            "INDIA_10Y_YIELD": yield_shock_pct,
        }

        scenario = self._build_scenario(
            scenario_name=scenario_name,
            horizon_days=horizon_days,
            market_shocks=market_shocks,
            institution_variables=(
                institution_variables
            ),
            source="direct",
        )

        return {
            "agent": self.agent_name,
            "status": "success",
            "scenario": scenario,
        }

    # ================================================================
    # CREATE FROM DICTIONARY
    # ================================================================

    def create_from_dict(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create scenario from dictionary configuration.

        Supported structure:

        {
            "scenario_name": "...",
            "horizon_days": 30,

            "market_shocks": {
                "NIFTY50": -15,
                "CRUDE_OIL": 30,
                "USD_INR": 5,
                "INDIA_VIX": 50,
                "INDIA_10Y_YIELD": 1
            },

            "institution_variables": [
                {
                    "name": "Credit_Spread",
                    "shock": 150,
                    "shock_type": "bps",
                    "unit": "bps"
                }
            ]
        }
        """

        if not isinstance(
            config,
            dict,
        ):

            raise TypeError(
                "Scenario configuration must be a dictionary."
            )

        scenario_name = config.get(
            "scenario_name",
            "Institutional Stress Test",
        )

        horizon_days = config.get(
            "horizon_days",
            30,
        )

        # ------------------------------------------------------------
        # Preferred market_shocks format.
        # ------------------------------------------------------------

        market_shocks = config.get(
            "market_shocks"
        )

        if market_shocks is None:

            # --------------------------------------------------------
            # Backward-compatible format.
            # --------------------------------------------------------

            market_shocks = {
                "NIFTY50": config.get(
                    "NIFTY50_shock_pct",
                    config.get(
                        "nifty_shock_pct",
                        0.0,
                    ),
                ),

                "CRUDE_OIL": config.get(
                    "CRUDE_OIL_shock_pct",
                    config.get(
                        "crude_shock_pct",
                        0.0,
                    ),
                ),

                "USD_INR": config.get(
                    "USD_INR_shock_pct",
                    config.get(
                        "usdinr_shock_pct",
                        0.0,
                    ),
                ),

                "INDIA_VIX": config.get(
                    "INDIA_VIX_shock_pct",
                    config.get(
                        "vix_shock_pct",
                        0.0,
                    ),
                ),

                "INDIA_10Y_YIELD": config.get(
                    "INDIA_10Y_YIELD_shock_pct",
                    config.get(
                        "yield_shock_pct",
                        0.0,
                    ),
                ),
            }

        institution_variables = config.get(
            "institution_variables",
            [],
        )

        scenario = self._build_scenario(
            scenario_name=scenario_name,
            horizon_days=horizon_days,
            market_shocks=market_shocks,
            institution_variables=(
                institution_variables
            ),
            source="dictionary",
        )

        return {
            "agent": self.agent_name,
            "status": "success",
            "scenario": scenario,
        }

    # ================================================================
    # LOAD JSON
    # ================================================================

    def load_json(
        self,
        file_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Load scenario from JSON.
        """

        path = Path(
            file_path
        ).expanduser()

        if not path.exists():

            raise FileNotFoundError(
                f"JSON scenario file not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Scenario path is not a file: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(
                file
            )

        if not isinstance(
            config,
            dict,
        ):

            raise ValueError(
                "JSON scenario must contain "
                "a JSON object."
            )

        result = self.create_from_dict(
            config
        )

        result[
            "scenario"
        ][
            "scenario_source"
        ] = "json"

        result[
            "scenario"
        ][
            "scenario_file"
        ] = str(
            path
        )

        return result

    # ================================================================
    # LOAD CSV
    # ================================================================

    def load_csv(
        self,
        file_path: str | Path,
        scenario_name: str = "Institutional Stress Test",
        horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Load scenario from CSV.

        Expected columns:

            variable,shock,type,unit,description

        Example:

            NIFTY50,-15,cumulative_pct,%,NIFTY stress
            CRUDE_OIL,30,cumulative_pct,%,Crude stress
            USD_INR,5,cumulative_pct,%,INR depreciation
            INDIA_VIX,50,cumulative_pct,%,VIX shock
            INDIA_10Y_YIELD,1,percentage_points,percentage points,Yield shock
            Credit_Spread,150,bps,bps,Credit spread widening
            Deposit_Outflow,8,cumulative_pct,%,Deposit outflow
            NPA_Ratio,2,percentage_points,percentage points,NPA increase
        """

        path = Path(
            file_path
        ).expanduser()

        if not path.exists():

            raise FileNotFoundError(
                f"CSV scenario file not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Scenario path is not a file: {path}"
            )

        market_shocks = {}

        institution_variables = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:

                raise ValueError(
                    "CSV scenario file has no header."
                )

            # --------------------------------------------------------
            # Normalize headers.
            # --------------------------------------------------------

            header_map = {}

            for header in reader.fieldnames:

                if header is None:
                    continue

                normalized_header = (
                    str(header)
                    .strip()
                    .lower()
                )

                header_map[
                    normalized_header
                ] = header

            # --------------------------------------------------------
            # Require variable + shock.
            # --------------------------------------------------------

            if "variable" not in header_map:

                raise ValueError(
                    "CSV must contain a 'variable' column."
                )

            if "shock" not in header_map:

                raise ValueError(
                    "CSV must contain a 'shock' column."
                )

            variable_column = header_map[
                "variable"
            ]

            shock_column = header_map[
                "shock"
            ]

            type_column = header_map.get(
                "type",
                header_map.get(
                    "shock_type"
                ),
            )

            unit_column = header_map.get(
                "unit"
            )

            description_column = header_map.get(
                "description"
            )

            direction_column = header_map.get(
                "direction"
            )

            # --------------------------------------------------------
            # Read rows.
            # --------------------------------------------------------

            for row_number, row in enumerate(
                reader,
                start=2,
            ):

                raw_name = row.get(
                    variable_column,
                    "",
                )

                name = str(
                    raw_name
                ).strip()

                if not name:
                    continue

                raw_shock = row.get(
                    shock_column,
                    "",
                )

                try:

                    shock = float(
                        raw_shock
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    raise ValueError(
                        f"Invalid shock '{raw_shock}' "
                        f"at CSV row {row_number}."
                    )

                # ----------------------------------------------------
                # Normalize core-variable name.
                # ----------------------------------------------------

                core_match = None

                for core in (
                    self.CORE_MARKET_VARIABLES
                ):

                    if name.upper() == core.upper():

                        core_match = core
                        break

                if core_match is not None:

                    market_shocks[
                        core_match
                    ] = shock

                    continue

                # ----------------------------------------------------
                # Institution variable.
                # ----------------------------------------------------

                shock_type = "absolute"

                if type_column is not None:

                    value = row.get(
                        type_column,
                        "",
                    )

                    if value:
                        shock_type = value

                variable = {
                    "name": name,
                    "shock": shock,
                    "shock_type": shock_type,
                    "unit": (
                        row.get(
                            unit_column,
                            "",
                        )
                        if unit_column
                        else ""
                    ),
                    "direction": (
                        row.get(
                            direction_column,
                            "",
                        )
                        if direction_column
                        else ""
                    ),
                    "description": (
                        row.get(
                            description_column,
                            "",
                        )
                        if description_column
                        else ""
                    ),
                }

                institution_variables.append(
                    variable
                )

        scenario = self._build_scenario(
            scenario_name=scenario_name,
            horizon_days=horizon_days,
            market_shocks=market_shocks,
            institution_variables=(
                institution_variables
            ),
            source="csv",
        )

        scenario[
            "scenario_file"
        ] = str(
            path
        )

        return {
            "agent": self.agent_name,
            "status": "success",
            "scenario": scenario,
        }

    # ================================================================
    # GENERIC SCENARIO FILE LOADER
    # ================================================================

    def load_scenario_file(
        self,
        file_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Automatically select JSON or CSV loader.
        """

        path = Path(
            file_path
        ).expanduser()

        if not path.exists():

            raise FileNotFoundError(
                f"Scenario file not found: {path}"
            )

        extension = path.suffix.lower()

        if extension == ".json":

            return self.load_json(
                path
            )

        if extension == ".csv":

            return self.load_csv(
                path
            )

        raise ValueError(
            f"Unsupported scenario file extension "
            f"'{extension}'. Use .json or .csv."
        )

    # ================================================================
    # GEOPOLITICAL CRISIS
    # ================================================================

    def geopolitical_crisis(
        self,
        horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Existing Geopolitical Crisis scenario.

        Preserved values:

            NIFTY50       -15%
            CRUDE_OIL     +30%
            USD_INR        +5%
            INDIA_VIX     +50%
            INDIA_10Y_YIELD +1 percentage point
        """

        return self.create_scenario(
            scenario_name="Geopolitical Crisis",

            horizon_days=horizon_days,

            nifty_shock_pct=-15.0,

            crude_shock_pct=30.0,

            usdinr_shock_pct=5.0,

            vix_shock_pct=50.0,

            yield_shock_pct=1.0,

            institution_variables=[],
        )

    # ================================================================
    # PRINT SCENARIO
    # ================================================================

    @staticmethod
    def print_scenario(
        result: Dict[str, Any],
    ) -> None:
        """
        Print scenario in a readable format.
        """

        if "scenario" in result:

            scenario = result[
                "scenario"
            ]

        else:

            scenario = result

        print()
        print("=" * 80)
        print("SCENARIO")
        print("=" * 80)

        print(
            f"Scenario Name : "
            f"{scenario.get('scenario_name', 'N/A')}"
        )

        print(
            f"Horizon       : "
            f"{scenario.get('horizon_days', 'N/A')} days"
        )

        # ------------------------------------------------------------
        # Market shocks
        # ------------------------------------------------------------

        print()
        print("Market Shocks")
        print("-" * 80)

        market_shocks = scenario.get(
            "market_shocks",
            {},
        )

        for name in (
            ScenarioAgent.CORE_MARKET_VARIABLES
        ):

            print(
                f"  {name:<25} : "
                f"{market_shocks.get(name, 0.0)}"
            )

        # ------------------------------------------------------------
        # Institution variables
        # ------------------------------------------------------------

        print()
        print("Institution Variables")
        print("-" * 80)

        institution_variables = scenario.get(
            "institution_variables",
            [],
        )

        if not institution_variables:

            print(
                "  None"
            )

        else:

            for item in institution_variables:

                name = item.get(
                    "name",
                    "Unknown",
                )

                shock = item.get(
                    "shock",
                    0.0,
                )

                shock_type = item.get(
                    "shock_type",
                    "absolute",
                )

                unit = item.get(
                    "unit",
                    "",
                )

                print(
                    f"  {name:<25} : "
                    f"{shock} "
                    f"{unit} "
                    f"[{shock_type}]"
                )

        # ------------------------------------------------------------
        # TimeGAN
        # ------------------------------------------------------------

        print()
        print("TimeGAN")
        print("-" * 80)

        print(
            "  Model       : TimeGAN V3.4"
        )

        print(
            "  Feature Dim : 5"
        )

        print(
            "  Institution variables "
            "added to TimeGAN : NO"
        )

        print()
        print("=" * 80)

    # ================================================================
    # STANDALONE TEST
    # ================================================================

    def test_custom_scenario(self) -> Dict[str, Any]:
        """
        Internal test for institution-specific variables.
        """

        institution_variables = [
            {
                "name": "Credit_Spread",
                "shock": 150,
                "shock_type": "bps",
                "unit": "bps",
                "description": (
                    "Corporate credit spread widening"
                ),
            },

            {
                "name": "Deposit_Outflow",
                "shock": 8,
                "shock_type": "cumulative_pct",
                "unit": "%",
                "description": (
                    "Cumulative deposit outflow"
                ),
            },

            {
                "name": "NPA_Ratio",
                "shock": 2,
                "shock_type": "percentage_points",
                "unit": "percentage points",
                "description": (
                    "Increase in NPA ratio"
                ),
            },

            {
                "name": "Loan_Default_Rate",
                "shock": 3,
                "shock_type": "percentage_points",
                "unit": "percentage points",
                "description": (
                    "Increase in loan default rate"
                ),
            },
        ]

        return self.create_scenario(
            scenario_name="Banking Liquidity Stress",
            horizon_days=30,

            nifty_shock_pct=-15.0,
            crude_shock_pct=30.0,
            usdinr_shock_pct=5.0,
            vix_shock_pct=50.0,
            yield_shock_pct=1.0,

            institution_variables=(
                institution_variables
            ),
        )


# =====================================================================
# DIRECT EXECUTION
# =====================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("MacroStress-GAN - Scenario Agent")
    print("=" * 80)

    agent = ScenarioAgent()

    # ================================================================
    # TEST 1 - EXISTING GEOPOLITICAL CRISIS
    # ================================================================

    print()
    print("TEST 1 - Geopolitical Crisis")

    result = agent.geopolitical_crisis(
        horizon_days=30
    )

    agent.print_scenario(
        result
    )

    # ================================================================
    # TEST 2 - CUSTOM INSTITUTION VARIABLES
    # ================================================================

    print()
    print("TEST 2 - Custom Institution Scenario")

    custom_result = (
        agent.test_custom_scenario()
    )

    agent.print_scenario(
        custom_result
    )

    # ================================================================
    # TEST 3 - DICTIONARY CONFIGURATION
    # ================================================================

    print()
    print("TEST 3 - Dictionary Configuration")

    dictionary_config = {
        "scenario_name": "Banking Liquidity Stress",
        "horizon_days": 30,

        "market_shocks": {
            "NIFTY50": -15,
            "CRUDE_OIL": 30,
            "USD_INR": 5,
            "INDIA_VIX": 50,
            "INDIA_10Y_YIELD": 1,
        },

        "institution_variables": [
            {
                "name": "Credit_Spread",
                "shock": 150,
                "shock_type": "bps",
                "unit": "bps",
            },

            {
                "name": "Deposit_Outflow",
                "shock": 8,
                "shock_type": "cumulative_pct",
                "unit": "%",
            },

            {
                "name": "NPA_Ratio",
                "shock": 2,
                "shock_type": "percentage_points",
                "unit": "percentage points",
            },
        ],
    }

    dictionary_result = (
        agent.create_from_dict(
            dictionary_config
        )
    )

    agent.print_scenario(
        dictionary_result
    )

    print()
    print("=" * 80)
    print("SCENARIO AGENT TESTS COMPLETED")
    print("=" * 80)