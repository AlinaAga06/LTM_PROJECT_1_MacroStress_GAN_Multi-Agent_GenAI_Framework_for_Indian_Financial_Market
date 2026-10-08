"""
MACROSTRESS-GAN
Stock Script Analyzer V2
========================

Purpose
-------
Safely parse an institute-provided Python configuration script
without executing the uploaded Python code.

The analyzer extracts:

    - institution type
    - portfolio amount
    - currency
    - requested macro variables
    - actual portfolio assets
    - initial portfolio weights
    - macro risk-driver sensitivities
    - portfolio constraints

The five macro variables are treated as RISK DRIVERS:

    NIFTY50
    CRUDE_OIL
    USD_INR
    INDIA_VIX
    INDIA_10Y_YIELD

They are NOT treated as portfolio assets.

The actual portfolio assets are supplied by the institute.

Security
--------
The uploaded script is NEVER executed.

Python AST parsing is used to extract the configuration.

Pipeline
--------
Institute Script
       |
       v
StockScriptAnalyzer V2
       |
       +--> Portfolio Assets
       |
       +--> Initial Weights
       |
       +--> Risk-Driver Sensitivities
       |
       +--> Constraints
       |
       v
Stock Data Discovery Agent
       |
       v
TimeGAN Stock V2
       |
       v
Risk / Monetary Risk
       |
       v
Portfolio Optimization V2
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StockScriptAnalyzer:
    """
    Secure AST-based analyzer for institute stock-market scripts.
    """

    VERSION = "StockScriptAnalyzer-V2"

    # ------------------------------------------------------------------
    # Supported institution type
    # ------------------------------------------------------------------

    SUPPORTED_INSTITUTION_TYPE = (
        "stock_market"
    )

    # ------------------------------------------------------------------
    # Supported macro risk drivers
    # ------------------------------------------------------------------

    SUPPORTED_VARIABLES = [
        "NIFTY50",
        "CRUDE_OIL",
        "USD_INR",
        "INDIA_VIX",
        "INDIA_10Y_YIELD",
    ]

    # ------------------------------------------------------------------
    # Macro variable -> TimeGAN feature
    # ------------------------------------------------------------------

    VARIABLE_TO_TIMEGAN_FEATURE = {
        "NIFTY50": "NIFTY50_Return",
        "CRUDE_OIL": "CRUDE_OIL_Return",
        "USD_INR": "USD_INR_Return",
        "INDIA_VIX": "INDIA_VIX_Change",
        "INDIA_10Y_YIELD": (
            "INDIA_10Y_YIELD_Change"
        ),
    }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    OUTPUT_PATH = Path(
        "outputs/stock_script_analysis.json"
    )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        source_file: Optional[str] = None,
    ):

        self.source_file = (
            str(source_file)
            if source_file
            else None
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def analyze_file(
        self,
        script_path: str,
        save_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Read and analyze an institute Python script.

        The script is parsed but NEVER executed.
        """

        path = Path(
            script_path
        )

        if not path.exists():

            return self._failure(
                (
                    "Script file does not exist: "
                    f"{path}"
                ),
                save_output=save_output,
            )

        if not path.is_file():

            return self._failure(
                (
                    "Script path is not a file: "
                    f"{path}"
                ),
                save_output=save_output,
            )

        try:

            source = path.read_text(
                encoding="utf-8"
            )

        except Exception as exc:

            return self._failure(
                (
                    "Unable to read script: "
                    f"{exc}"
                ),
                save_output=save_output,
            )

        return self.analyze_source(
            source=source,
            source_file=str(
                path.resolve()
            ),
            save_output=save_output,
        )

    # ==================================================================
    # ANALYZE SOURCE
    # ==================================================================

    def analyze_source(
        self,
        source: str,
        source_file: Optional[str] = None,
        save_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze Python source code using AST.

        IMPORTANT:
        The source is never executed.
        """

        if not isinstance(
            source,
            str,
        ):

            return self._failure(
                "source must be a string.",
                save_output=save_output,
            )

        if not source.strip():

            return self._failure(
                "The supplied Python script is empty.",
                save_output=save_output,
            )

        self.source_file = source_file

        try:

            tree = ast.parse(
                source
            )

        except SyntaxError as exc:

            return self._failure(
                (
                    "Invalid Python syntax: "
                    f"{exc}"
                ),
                save_output=save_output,
            )

        try:

            institution_node = (
                self._find_institution_assignment(
                    tree
                )
            )

            if institution_node is None:

                raise ValueError(
                    "Could not find "
                    "'institution = {...}' "
                    "configuration."
                )

            institution = (
                self._evaluate_literal(
                    institution_node
                )
            )

            if not isinstance(
                institution,
                dict,
            ):

                raise ValueError(
                    "'institution' must contain "
                    "a Python dictionary."
                )

            # ----------------------------------------------------------
            # Validate basic institution fields
            # ----------------------------------------------------------

            institution = (
                self._normalize_institution(
                    institution
                )
            )

            # ----------------------------------------------------------
            # Determine portfolio format
            # ----------------------------------------------------------

            if "portfolio_assets" in (
                institution
            ):

                parsed = (
                    self._parse_v2_portfolio(
                        institution
                    )
                )

                portfolio_format = (
                    "portfolio_assets_v2"
                )

            elif "portfolio_weights" in (
                institution
            ):

                parsed = (
                    self._parse_legacy_portfolio(
                        institution
                    )
                )

                portfolio_format = (
                    "legacy_macro_weights"
                )

            else:

                raise ValueError(
                    "Institution configuration must "
                    "contain 'portfolio_assets'."
                )

            # ----------------------------------------------------------
            # Build result
            # ----------------------------------------------------------

            result = {

                "status": "validated",

                "institution": (
                    parsed[
                        "institution"
                    ]
                ),

                "timegan_mapping": (
                    parsed[
                        "timegan_mapping"
                    ]
                ),

                "timegan_features": (
                    parsed[
                        "timegan_features"
                    ]
                ),

                "portfolio_assets": (
                    parsed[
                        "portfolio_assets"
                    ]
                ),

                "portfolio_allocations": (
                    parsed[
                        "portfolio_allocations"
                    ]
                ),

                "risk_driver_sensitivities": (
                    parsed[
                        "risk_driver_sensitivities"
                    ]
                ),

                "constraints": (
                    parsed[
                        "constraints"
                    ]
                ),

                "validation": (
                    parsed[
                        "validation"
                    ]
                ),

                "analyzer": {

                    "name": (
                        "StockScriptAnalyzer"
                    ),

                    "version": (
                        self.VERSION
                    ),

                    "source_file": (
                        self.source_file
                    ),

                    "execution_performed": (
                        False
                    ),

                    "ast_parsing": True,

                    "portfolio_format": (
                        portfolio_format
                    ),

                    "safe_mode": True,
                },
            }

            if save_output:

                self._save_result(
                    result
                )

            return result

        except Exception as exc:

            return self._failure(
                str(exc),
                save_output=save_output,
            )

    # ==================================================================
    # FIND institution = {...}
    # ==================================================================

    def _find_institution_assignment(
        self,
        tree: ast.AST,
    ) -> Optional[ast.AST]:
        """
        Find:

            institution = {...}

        without executing the script.
        """

        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.Assign,
            ):

                for target in (
                    node.targets
                ):

                    if (
                        isinstance(
                            target,
                            ast.Name,
                        )
                        and target.id
                        == "institution"
                    ):

                        return node.value

            elif isinstance(
                node,
                ast.AnnAssign,
            ):

                target = (
                    node.target
                )

                if (
                    isinstance(
                        target,
                        ast.Name,
                    )
                    and target.id
                    == "institution"
                ):

                    return node.value

        return None

    # ==================================================================
    # SAFE AST LITERAL EVALUATION
    # ==================================================================

    def _evaluate_literal(
        self,
        node: ast.AST,
    ) -> Any:
        """
        Safely convert AST literals to Python values.

        Supported:
            dict
            list
            tuple
            string
            integer
            float
            boolean
            None
        """

        try:

            return ast.literal_eval(
                node
            )

        except Exception as exc:

            raise ValueError(
                "Institution configuration "
                "must contain only literal "
                "Python values. "
                f"Unsafe or unsupported expression: "
                f"{exc}"
            )

    # ==================================================================
    # NORMALIZE BASIC INSTITUTION CONFIGURATION
    # ==================================================================

    def _normalize_institution(
        self,
        institution: Dict[str, Any],
    ) -> Dict[str, Any]:

        normalized = dict(
            institution
        )

        # --------------------------------------------------------------
        # Type
        # --------------------------------------------------------------

        institution_type = str(
            normalized.get(
                "type",
                "",
            )
        ).strip()

        if (
            institution_type
            != self.SUPPORTED_INSTITUTION_TYPE
        ):

            raise ValueError(
                "Unsupported institution type: "
                f"{institution_type}. "
                "Expected 'stock_market'."
            )

        normalized[
            "type"
        ] = institution_type

        # --------------------------------------------------------------
        # Portfolio amount
        # --------------------------------------------------------------

        if (
            "portfolio_amount"
            not in normalized
        ):

            raise ValueError(
                "Missing 'portfolio_amount'."
            )

        try:

            portfolio_amount = float(
                normalized[
                    "portfolio_amount"
                ]
            )

        except Exception:

            raise ValueError(
                "portfolio_amount must be numeric."
            )

        if portfolio_amount <= 0:

            raise ValueError(
                "portfolio_amount must "
                "be greater than zero."
            )

        normalized[
            "portfolio_amount"
        ] = portfolio_amount

        # --------------------------------------------------------------
        # Currency
        # --------------------------------------------------------------

        currency = str(
            normalized.get(
                "currency",
                "",
            )
        ).strip().upper()

        if not currency:

            raise ValueError(
                "Missing 'currency'."
            )

        normalized[
            "currency"
        ] = currency

        # --------------------------------------------------------------
        # Variables
        # --------------------------------------------------------------

        variables = normalized.get(
            "variables"
        )

        if not isinstance(
            variables,
            list,
        ):

            raise ValueError(
                "'variables' must be a list."
            )

        if not variables:

            raise ValueError(
                "'variables' cannot be empty."
            )

        variables = [
            str(
                variable
            ).strip().upper()
            for variable in variables
        ]

        unsupported = [
            variable
            for variable in variables
            if variable
            not in self.SUPPORTED_VARIABLES
        ]

        if unsupported:

            raise ValueError(
                "Unsupported variables: "
                f"{unsupported}. "
                f"Supported variables: "
                f"{self.SUPPORTED_VARIABLES}"
            )

        # Remove duplicates while preserving order.
        variables = list(
            dict.fromkeys(
                variables
            )
        )

        normalized[
            "variables"
        ] = variables

        return normalized

    # ==================================================================
    # PARSE V2 PORTFOLIO
    # ==================================================================

    def _parse_v2_portfolio(
        self,
        institution: Dict[str, Any],
    ) -> Dict[str, Any]:

        assets = institution[
            "portfolio_assets"
        ]

        if not isinstance(
            assets,
            dict,
        ):

            raise ValueError(
                "'portfolio_assets' must "
                "be a dictionary."
            )

        if len(assets) < 2:

            raise ValueError(
                "At least two actual portfolio "
                "assets are required."
            )

        variables = institution[
            "variables"
        ]

        portfolio_amount = float(
            institution[
                "portfolio_amount"
            ]
        )

        currency = institution[
            "currency"
        ]

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------

        constraints = (
            institution.get(
                "constraints",
                {},
            )
        )

        if not isinstance(
            constraints,
            dict,
        ):

            raise ValueError(
                "'constraints' must "
                "be a dictionary."
            )

        min_weight = float(
            constraints.get(
                "min_weight",
                0.0,
            )
        )

        max_weight = float(
            constraints.get(
                "max_weight",
                1.0,
            )
        )

        if min_weight < 0:

            raise ValueError(
                "min_weight cannot be negative."
            )

        if max_weight > 1:

            raise ValueError(
                "max_weight cannot exceed 1."
            )

        if min_weight > max_weight:

            raise ValueError(
                "min_weight cannot exceed "
                "max_weight."
            )

        # --------------------------------------------------------------
        # Validate portfolio feasibility
        # --------------------------------------------------------------

        number_assets = len(
            assets
        )

        if (
            number_assets
            * min_weight
            > 1.0
        ):

            raise ValueError(
                "Portfolio constraints are "
                "infeasible: minimum weights "
                "exceed 100%."
            )

        if (
            number_assets
            * max_weight
            < 1.0
        ):

            raise ValueError(
                "Portfolio constraints are "
                "infeasible: maximum weights "
                "cannot reach 100%."
            )

        # --------------------------------------------------------------
        # Extract weights
        # --------------------------------------------------------------

        weights = {}

        for (
            asset_name,
            asset_config,
        ) in assets.items():

            if not isinstance(
                asset_config,
                dict,
            ):

                raise ValueError(
                    f"Asset '{asset_name}' "
                    "must contain a dictionary."
                )

            if "weight" not in (
                asset_config
            ):

                raise ValueError(
                    f"Asset '{asset_name}' "
                    "is missing 'weight'."
                )

            weight = float(
                asset_config[
                    "weight"
                ]
            )

            if weight < 0:

                raise ValueError(
                    f"Asset '{asset_name}' "
                    "has negative weight."
                )

            if (
                weight
                < min_weight
                - 1e-8
            ):

                raise ValueError(
                    f"Asset '{asset_name}' "
                    f"weight {weight} is below "
                    f"min_weight {min_weight}."
                )

            if (
                weight
                > max_weight
                + 1e-8
            ):

                raise ValueError(
                    f"Asset '{asset_name}' "
                    f"weight {weight} exceeds "
                    f"max_weight {max_weight}."
                )

            weights[
                asset_name
            ] = weight

        weight_sum = sum(
            weights.values()
        )

        if not self._approximately_equal(
            weight_sum,
            1.0,
        ):

            raise ValueError(
                "Portfolio asset weights "
                f"must sum to 1.0. "
                f"Current sum = {weight_sum:.10f}"
            )

        # --------------------------------------------------------------
        # Risk-driver sensitivities
        # --------------------------------------------------------------

        normalized_assets = {}

        risk_driver_sensitivities = {}

        for (
            asset_name,
            asset_config,
        ) in assets.items():

            raw_drivers = (
                asset_config.get(
                    "risk_drivers"
                )
            )

            if not isinstance(
                raw_drivers,
                dict,
            ):

                raise ValueError(
                    f"Asset '{asset_name}' "
                    "must contain a "
                    "'risk_drivers' dictionary."
                )

            normalized_drivers = {}

            for (
                driver,
                sensitivity,
            ) in raw_drivers.items():

                driver_name = str(
                    driver
                ).strip().upper()

                if (
                    driver_name
                    not in self.SUPPORTED_VARIABLES
                ):

                    raise ValueError(
                        f"Asset '{asset_name}' "
                        f"contains unsupported "
                        f"risk driver "
                        f"'{driver_name}'."
                    )

                # A driver does not have to appear
                # in every asset, but if supplied it
                # must be numeric.
                sensitivity = float(
                    sensitivity
                )

                if not (
                    sensitivity
                    == sensitivity
                ):

                    raise ValueError(
                        f"NaN sensitivity for "
                        f"{asset_name}/"
                        f"{driver_name}."
                    )

                normalized_drivers[
                    driver_name
                ] = sensitivity

            normalized_assets[
                asset_name
            ] = {

                "weight": (
                    weights[
                        asset_name
                    ]
                ),

                "risk_drivers": (
                    normalized_drivers
                ),
            }

            risk_driver_sensitivities[
                asset_name
            ] = normalized_drivers

        # --------------------------------------------------------------
        # TimeGAN mapping
        # --------------------------------------------------------------

        timegan_mapping = {}

        timegan_features = []

        for variable in variables:

            feature = (
                self.VARIABLE_TO_TIMEGAN_FEATURE[
                    variable
                ]
            )

            timegan_mapping[
                variable
            ] = feature

            timegan_features.append(
                feature
            )

        # --------------------------------------------------------------
        # Monetary allocations
        # --------------------------------------------------------------

        portfolio_allocations = {}

        for (
            asset_name,
            asset_config,
        ) in normalized_assets.items():

            weight = float(
                asset_config[
                    "weight"
                ]
            )

            portfolio_allocations[
                asset_name
            ] = {

                "weight": weight,

                "weight_percent": (
                    weight
                    * 100.0
                ),

                "amount": (
                    weight
                    * portfolio_amount
                ),

                "currency": (
                    currency
                ),

                "risk_drivers": (
                    asset_config[
                        "risk_drivers"
                    ]
                ),
            }

        # --------------------------------------------------------------
        # Validation information
        # --------------------------------------------------------------

        validation = {

            "portfolio_weight_sum": (
                weight_sum
            ),

            "portfolio_weight_sum_valid": (
                self._approximately_equal(
                    weight_sum,
                    1.0,
                )
            ),

            "all_variables_supported": True,

            "all_assets_have_weights": True,

            "all_assets_have_risk_drivers": (
                True
            ),

            "weights_within_constraints": (
                all(
                    (
                        min_weight
                        - 1e-8
                    )
                    <= weight
                    <= (
                        max_weight
                        + 1e-8
                    )
                    for weight in (
                        weights.values()
                    )
                )
            ),

            "constraints_feasible": (
                (
                    number_assets
                    * min_weight
                    <= 1.0
                )
                and
                (
                    number_assets
                    * max_weight
                    >= 1.0
                )
            ),

            "portfolio_amount_valid": (
                portfolio_amount > 0
            ),

            "risk_driver_model_valid": (
                True
            ),
        }

        return {

            "institution": {

                "type": (
                    institution[
                        "type"
                    ]
                ),

                "portfolio_amount": (
                    portfolio_amount
                ),

                "currency": (
                    currency
                ),

                "variables": (
                    variables
                ),
            },

            "timegan_mapping": (
                timegan_mapping
            ),

            "timegan_features": (
                timegan_features
            ),

            "portfolio_assets": (
                normalized_assets
            ),

            "portfolio_allocations": (
                portfolio_allocations
            ),

            "risk_driver_sensitivities": (
                risk_driver_sensitivities
            ),

            "constraints": {

                "min_weight": (
                    min_weight
                ),

                "max_weight": (
                    max_weight
                ),

                "sum_weights": 1.0,
            },

            "validation": validation,
        }

    # ==================================================================
    # LEGACY PORTFOLIO FORMAT
    # ==================================================================

    def _parse_legacy_portfolio(
        self,
        institution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detect the old V1 format.

        We do NOT silently treat macro variables as actual assets.

        Instead, return a clear migration error explaining what
        the institute script needs to change.
        """

        raise ValueError(
            "Legacy 'portfolio_weights' format detected. "
            "The V2 optimizer requires actual "
            "'portfolio_assets' with asset-level "
            "'risk_drivers'. "
            "Do not use NIFTY50, CRUDE_OIL, USD_INR, "
            "INDIA_VIX, or INDIA_10Y_YIELD as portfolio "
            "assets. They must remain macro risk drivers."
        )

    # ==================================================================
    # NUMERIC COMPARISON
    # ==================================================================

    def _approximately_equal(
        self,
        a: float,
        b: float,
        tolerance: float = 1e-6,
    ) -> bool:

        return (
            abs(
                float(a)
                - float(b)
            )
            <= tolerance
        )

    # ==================================================================
    # FAILURE RESULT
    # ==================================================================

    def _failure(
        self,
        message: str,
        save_output: bool = True,
    ) -> Dict[str, Any]:

        result = {

            "status": "failed",

            "error": str(
                message
            ),

            "analyzer": {

                "name": (
                    "StockScriptAnalyzer"
                ),

                "version": (
                    self.VERSION
                ),

                "source_file": (
                    self.source_file
                ),

                "execution_performed": (
                    False
                ),

                "ast_parsing": True,

                "safe_mode": True,
            },
        }

        if save_output:

            self._save_result(
                result
            )

        return result

    # ==================================================================
    # SAVE
    # ==================================================================

    def _save_result(
        self,
        result: Dict[str, Any],
    ) -> None:

        self.OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )


# ==========================================================================
# DIRECT TEST
# ==========================================================================

def main():

    print("=" * 80)
    print(
        "MACROSTRESS-GAN"
    )
    print(
        "STOCK SCRIPT ANALYZER V2"
    )
    print("=" * 80)

    # ----------------------------------------------------------------------
    # Example institute configuration
    #
    # These values are TEST values only.
    # ----------------------------------------------------------------------

    test_script = """
institution = {

    "type": "stock_market",

    "portfolio_amount": 100000000,

    "currency": "INR",

    "variables": [
        "NIFTY50",
        "CRUDE_OIL",
        "USD_INR",
        "INDIA_VIX",
        "INDIA_10Y_YIELD"
    ],

    "portfolio_assets": {

        "Asset_A": {

            "weight": 0.40,

            "risk_drivers": {
                "NIFTY50": 0.80,
                "USD_INR": -0.10,
                "INDIA_VIX": -0.10
            }
        },

        "Asset_B": {

            "weight": 0.35,

            "risk_drivers": {
                "NIFTY50": 0.60,
                "CRUDE_OIL": -0.20,
                "USD_INR": -0.20
            }
        },

        "Asset_C": {

            "weight": 0.25,

            "risk_drivers": {
                "NIFTY50": 0.40,
                "INDIA_10Y_YIELD": -0.60
            }
        }
    },

    "constraints": {

        "min_weight": 0.05,

        "max_weight": 0.60
    }
}
"""

    # ----------------------------------------------------------------------
    # Analyze
    # ----------------------------------------------------------------------

    analyzer = (
        StockScriptAnalyzer()
    )

    result = analyzer.analyze_source(
        source=test_script,
        source_file=None,
        save_output=True,
    )

    # ----------------------------------------------------------------------
    # Display status
    # ----------------------------------------------------------------------

    print()

    print(
        "STATUS"
    )

    print(
        result[
            "status"
        ]
    )

    if result[
        "status"
    ] != "validated":

        print()

        print(
            "ERROR"
        )

        print(
            result.get(
                "error",
                "Unknown error",
            )
        )

        return

    # ----------------------------------------------------------------------
    # Institution
    # ----------------------------------------------------------------------

    print()

    print(
        "INSTITUTION"
    )

    print(
        json.dumps(
            result[
                "institution"
            ],
            indent=2,
        )
    )

    # ----------------------------------------------------------------------
    # TimeGAN mapping
    # ----------------------------------------------------------------------

    print()

    print(
        "TIMEGAN MAPPING"
    )

    for (
        variable,
        feature,
    ) in result[
        "timegan_mapping"
    ].items():

        print(
            f"{variable:20s}"
            f" -> "
            f"{feature}"
        )

    # ----------------------------------------------------------------------
    # Portfolio assets
    # ----------------------------------------------------------------------

    print()

    print(
        "PORTFOLIO ASSETS"
    )

    for (
        asset_name,
        asset_config,
    ) in result[
        "portfolio_assets"
    ].items():

        print()

        print(
            f"{asset_name}"
        )

        print(
            f"  Weight: "
            f"{asset_config['weight'] * 100:.2f}%"
        )

        print(
            "  Risk Drivers:"
        )

        for (
            driver,
            sensitivity,
        ) in asset_config[
            "risk_drivers"
        ].items():

            print(
                f"    "
                f"{driver:20s}"
                f"{sensitivity:+.4f}"
            )

    # ----------------------------------------------------------------------
    # Monetary allocations
    # ----------------------------------------------------------------------

    print()

    print(
        "PORTFOLIO ALLOCATIONS"
    )

    for (
        asset_name,
        allocation,
    ) in result[
        "portfolio_allocations"
    ].items():

        print(
            f"{asset_name:20s}"
            f" {allocation['weight_percent']:8.2f}%"
            f" "
            f"{allocation['amount']:,.2f} "
            f"{allocation['currency']}"
        )

    # ----------------------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------------------

    print()

    print(
        "CONSTRAINTS"
    )

    print(
        json.dumps(
            result[
                "constraints"
            ],
            indent=2,
        )
    )

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------

    print()

    print(
        "VALIDATION"
    )

    print(
        json.dumps(
            result[
                "validation"
            ],
            indent=2,
        )
    )

    # ----------------------------------------------------------------------
    # Security
    # ----------------------------------------------------------------------

    print()

    print(
        "SECURITY"
    )

    print(
        "AST parsing             : True"
    )

    print(
        "Script execution        : False"
    )

    print(
        "Safe configuration mode : True"
    )

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

    print()

    print(
        "OUTPUT"
    )

    print(
        f"Saved to: "
        f"{analyzer.OUTPUT_PATH}"
    )

    print("=" * 80)


# ==========================================================================
# ENTRY POINT
# ==========================================================================

if __name__ == "__main__":

    main()