"""
MacroStress-GAN
Stock Data Discovery & Validation Agent
Version: StockDataDiscoveryAgent-V1

Purpose
-------
1. Read variables extracted by StockScriptAnalyzer.
2. Map each variable to a real-data source.
3. Search the project for locally downloaded datasets.
4. Validate local CSV data:
   - required variable exists
   - date column exists
   - numeric values exist
   - missing values
   - duplicate dates
   - date coverage
5. Never execute downloaded Python code.
6. Never silently create synthetic data.
7. Clearly distinguish:
   - source_available
   - local_data_found
   - validated
   - missing_local_data
   - validation_failed

This agent is a DATA DISCOVERY / VALIDATION layer.
It does not train TimeGAN.
It does not modify the TimeGAN model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

INSTITUTION_ROOT = PROCESSED_ROOT / "institutions" / "stock"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DISCOVERY_OUTPUT = OUTPUT_ROOT / "stock_data_discovery.json"


# =============================================================================
# SUPPORTED STOCK VARIABLES
# =============================================================================

SUPPORTED_VARIABLES = [
    "NIFTY50",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_VIX",
    "INDIA_10Y_YIELD",
]


# =============================================================================
# TIMEGAN MAPPING
# =============================================================================

TIMEGAN_MAPPING = {
    "NIFTY50": "NIFTY50_Return",
    "CRUDE_OIL": "CRUDE_OIL_Return",
    "USD_INR": "USD_INR_Return",
    "INDIA_VIX": "INDIA_VIX_Change",
    "INDIA_10Y_YIELD": "INDIA_10Y_YIELD_Change",
}


# =============================================================================
# REAL DATA SOURCE REGISTRY
# =============================================================================
#
# These are SOURCE DEFINITIONS.
#
# The agent does NOT claim that a source has been downloaded merely because
# the source exists.
#
# "local_data_found" and "validated" are only true after checking a local file.
#
# =============================================================================

SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {

    "NIFTY50": {
        "variable": "NIFTY50",
        "source_name": "NSE India",
        "source_type": "official_exchange",
        "description": "NIFTY 50 historical index data",
        "official_url": "https://www.nseindia.com/all-reports",
        "historical_data_page": (
            "https://www.nseindia.com/all-reports"
        ),
        "known_symbols": [
            "NIFTY50",
            "^NSEI",
            "NSEI",
        ],
        "expected_value_columns": [
            "Close",
            "close",
            "CLOSE",
            "NIFTY50",
            "NIFTY 50",
            "Nifty50",
        ],
        "notes": (
            "NSE provides historical NIFTY 50 index data. "
            "NIFTY 50 historical data is available from NSE."
        ),
    },

    "CRUDE_OIL": {
        "variable": "CRUDE_OIL",
        "source_name": "U.S. Energy Information Administration",
        "source_type": "official_government",
        "description": "Crude oil historical spot-price data",
        "official_url": (
            "https://www.eia.gov/dnav/pet/"
            "pet_pri_spt_s1_d.htm"
        ),
        "known_symbols": [
            "WTI",
            "BRENT",
            "CL=F",
        ],
        "expected_value_columns": [
            "Close",
            "close",
            "WTI",
            "Brent",
            "Brent Crude",
            "Crude Oil",
            "CRUDE_OIL",
        ],
        "notes": (
            "EIA provides long historical WTI and Brent crude-oil "
            "price series. The institution should specify WTI or Brent "
            "if the distinction matters."
        ),
    },

    "USD_INR": {
        "variable": "USD_INR",
        "source_name": "Reserve Bank of India / NSE",
        "source_type": "official_financial",
        "description": "USD/INR exchange-rate data",
        "official_url": (
            "https://www.nseindia.com/all-reports"
        ),
        "known_symbols": [
            "USDINR",
            "USD/INR",
            "INR=X",
        ],
        "expected_value_columns": [
            "Close",
            "close",
            "USD_INR",
            "USDINR",
            "USD/INR",
            "Rate",
            "RATE",
        ],
        "notes": (
            "NSE historical reports expose RBI reference-rate resources. "
            "The exact rate definition should remain consistent across "
            "the institution's dataset."
        ),
    },

    "INDIA_VIX": {
        "variable": "INDIA_VIX",
        "source_name": "NSE India",
        "source_type": "official_exchange",
        "description": "India VIX historical volatility-index data",
        "official_url": (
            "https://www.nseindia.com/static/"
            "products-services/indices-indiavix-index"
        ),
        "historical_data_page": (
            "https://www.nseindia.com/all-reports"
        ),
        "known_symbols": [
            "INDIA_VIX",
            "INDIAVIX",
            "^INDIAVIX",
        ],
        "expected_value_columns": [
            "Close",
            "close",
            "INDIA_VIX",
            "India VIX",
            "INDIAVIX",
        ],
        "notes": (
            "NSE provides historical India VIX values. "
            "India VIX represents expected near-term volatility "
            "based on NIFTY option prices."
        ),
    },

    "INDIA_10Y_YIELD": {
        "variable": "INDIA_10Y_YIELD",
        "source_name": "Indian Government Bond / RBI-related market data",
        "source_type": "official_financial",
        "description": "India 10-Year Government Bond Yield",
        "official_url": (
            "https://www.rbi.org.in/"
        ),
        "known_symbols": [
            "INDIA_10Y_YIELD",
            "INDIA 10Y YIELD",
            "10Y",
            "10-Year",
            "10 Year",
            "India 10Y",
        ],
        "expected_value_columns": [
            "Close",
            "close",
            "INDIA_10Y_YIELD",
            "India 10Y",
            "10Y",
            "Yield",
            "YIELD",
        ],
        "notes": (
            "The project already supports a downloaded India 10-Year "
            "Government Bond Yield dataset. The discovery agent validates "
            "the local file rather than generating replacement data."
        ),
    },
}


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class ValidationResult:
    variable: str
    status: str

    source_name: str
    source_type: str
    official_url: str

    local_file: Optional[str]

    rows: int
    columns: List[str]

    date_column: Optional[str]
    value_column: Optional[str]

    start_date: Optional[str]
    end_date: Optional[str]

    missing_values: Optional[int]
    duplicate_dates: Optional[int]

    numeric_values: Optional[int]

    timegan_feature: Optional[str]

    message: str


# =============================================================================
# AGENT
# =============================================================================

class StockDataDiscoveryAgent:
    """
    Discovers and validates real stock-market datasets.

    Important:
        This class NEVER executes an uploaded Python script.
        The script should already have been parsed by StockScriptAnalyzer.
    """

    VERSION = "StockDataDiscoveryAgent-V1"

    # -------------------------------------------------------------------------
    # DATE COLUMN CANDIDATES
    # -------------------------------------------------------------------------

    DATE_COLUMNS = [
        "Date",
        "date",
        "DATE",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp",
        "Time",
        "time",
    ]

    # -------------------------------------------------------------------------
    # COMMON PROJECT FILE NAMES
    # -------------------------------------------------------------------------

    COMMON_DATA_FILE_NAMES = [
        "stock_data.csv",
        "stock_market_data.csv",
        "stock_dataset.csv",
        "macro_stress_5vars_processed.csv",
        "stock_timegan_data.csv",
        "stock_timegan_dataset.csv",
        "india_market_data.csv",
        "market_data.csv",
        "nifty50.csv",
        "crude_oil.csv",
        "usd_inr.csv",
        "india_vix.csv",
        "india_10y_yield.csv",
    ]

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
    ) -> None:

        if project_root is None:
            self.project_root = PROJECT_ROOT
        else:
            self.project_root = Path(project_root).resolve()

        self.data_root = self.project_root / "data"
        self.output_root = self.project_root / "outputs"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def discover(
        self,
        institution_config: Dict[str, Any],
        save_result: bool = True,
    ) -> Dict[str, Any]:

        variables = institution_config.get("variables", [])

        if not isinstance(variables, list):
            raise ValueError(
                "'variables' must be a list."
            )

        variables = [
            str(variable).strip().upper()
            for variable in variables
        ]

        unsupported = [
            variable
            for variable in variables
            if variable not in SUPPORTED_VARIABLES
        ]

        if unsupported:
            raise ValueError(
                f"Unsupported stock variables: {unsupported}"
            )

        results: List[Dict[str, Any]] = []

        for variable in variables:

            source = SOURCE_REGISTRY[variable]

            candidate_files = self.find_local_files(
                variable=variable
            )

            if not candidate_files:

                result = ValidationResult(
                    variable=variable,
                    status="source_available_but_local_data_missing",
                    source_name=source["source_name"],
                    source_type=source["source_type"],
                    official_url=source["official_url"],
                    local_file=None,
                    rows=0,
                    columns=[],
                    date_column=None,
                    value_column=None,
                    start_date=None,
                    end_date=None,
                    missing_values=None,
                    duplicate_dates=None,
                    numeric_values=None,
                    timegan_feature=TIMEGAN_MAPPING.get(variable),
                    message=(
                        "A real-data source is registered, but no "
                        "matching local dataset was found. "
                        "No synthetic fallback was created."
                    ),
                )

                results.append(asdict(result))
                continue

            best_result = self.validate_candidates(
                variable=variable,
                candidate_files=candidate_files,
            )

            results.append(asdict(best_result))

        validated_count = sum(
            1
            for item in results
            if item["status"] == "validated"
        )

        missing_count = sum(
            1
            for item in results
            if item["status"]
            == "source_available_but_local_data_missing"
        )

        failed_count = sum(
            1
            for item in results
            if item["status"] == "validation_failed"
        )

        overall_status = self._overall_status(
            total=len(results),
            validated=validated_count,
            missing=missing_count,
            failed=failed_count,
        )

        result = {
            "status": overall_status,

            "agent": {
                "name": "StockDataDiscoveryAgent",
                "version": self.VERSION,
                "execution_performed": True,
                "synthetic_data_created": False,
            },

            "institution": {
                "type": institution_config.get("type"),
                "variables": variables,
            },

            "timegan_mapping": {
                variable: TIMEGAN_MAPPING[variable]
                for variable in variables
            },

            "source_registry": {
                variable: {
                    "source_name": SOURCE_REGISTRY[variable][
                        "source_name"
                    ],
                    "source_type": SOURCE_REGISTRY[variable][
                        "source_type"
                    ],
                    "official_url": SOURCE_REGISTRY[variable][
                        "official_url"
                    ],
                }
                for variable in variables
            },

            "summary": {
                "requested_variables": len(variables),
                "validated_variables": validated_count,
                "missing_local_data": missing_count,
                "validation_failures": failed_count,
                "all_variables_validated": (
                    validated_count == len(variables)
                ),
            },

            "results": results,

            "next_step": self._next_step(
                overall_status=overall_status,
                results=results,
            ),
        }

        if save_result:
            self.save_result(result)

        return result

    # =========================================================================
    # LOCAL FILE DISCOVERY
    # =========================================================================

    def find_local_files(
        self,
        variable: str,
    ) -> List[Path]:

        variable_upper = variable.upper()

        search_roots = [
            self.data_root,
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "processed" / "institutions",
            self.project_root / "data" / "processed" / "institutions" / "stock",
        ]

        candidates: List[Path] = []

        # ---------------------------------------------------------------------
        # First: known file names
        # ---------------------------------------------------------------------

        for root in search_roots:

            if not root.exists():
                continue

            for filename in self.COMMON_DATA_FILE_NAMES:

                path = root / filename

                if path.exists() and path.is_file():
                    candidates.append(path)

        # ---------------------------------------------------------------------
        # Second: recursive CSV discovery
        # ---------------------------------------------------------------------

        for root in search_roots:

            if not root.exists():
                continue

            try:
                csv_files = root.rglob("*.csv")
            except Exception:
                continue

            for path in csv_files:

                if not path.is_file():
                    continue

                filename_upper = path.name.upper()

                variable_tokens = self._variable_tokens(
                    variable
                )

                if any(
                    token in filename_upper
                    for token in variable_tokens
                ):
                    candidates.append(path)

        # ---------------------------------------------------------------------
        # Remove duplicates
        # ---------------------------------------------------------------------

        unique = {}

        for path in candidates:
            try:
                unique[str(path.resolve())] = path.resolve()
            except Exception:
                unique[str(path)] = path

        # ---------------------------------------------------------------------
        # Also search for multi-variable datasets.
        #
        # Example:
        # macro_stress_5vars_processed.csv
        # ---------------------------------------------------------------------

        for root in search_roots:

            if not root.exists():
                continue

            try:
                csv_files = root.rglob("*.csv")
            except Exception:
                continue

            for path in csv_files:

                try:
                    preview = pd.read_csv(
                        path,
                        nrows=5
                    )
                except Exception:
                    continue

                columns_upper = {
                    str(column).upper()
                    for column in preview.columns
                }

                if self._dataset_contains_variable(
                    variable,
                    columns_upper,
                ):
                    unique[str(path.resolve())] = path.resolve()

        return sorted(
            unique.values(),
            key=lambda p: str(p).lower(),
        )

    # =========================================================================
    # VALIDATE CANDIDATES
    # =========================================================================

    def validate_candidates(
        self,
        variable: str,
        candidate_files: List[Path],
    ) -> ValidationResult:

        best_failed_result: Optional[ValidationResult] = None

        for file_path in candidate_files:

            result = self.validate_file(
                variable=variable,
                file_path=file_path,
            )

            if result.status == "validated":
                return result

            best_failed_result = result

        if best_failed_result is not None:
            return best_failed_result

        source = SOURCE_REGISTRY[variable]

        return ValidationResult(
            variable=variable,
            status="validation_failed",
            source_name=source["source_name"],
            source_type=source["source_type"],
            official_url=source["official_url"],
            local_file=None,
            rows=0,
            columns=[],
            date_column=None,
            value_column=None,
            start_date=None,
            end_date=None,
            missing_values=None,
            duplicate_dates=None,
            numeric_values=None,
            timegan_feature=TIMEGAN_MAPPING.get(variable),
            message="No readable candidate dataset was found.",
        )

    # =========================================================================
    # VALIDATE ONE FILE
    # =========================================================================

    def validate_file(
        self,
        variable: str,
        file_path: Path,
    ) -> ValidationResult:

        source = SOURCE_REGISTRY[variable]

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=0,
                columns=[],
                date_column=None,
                value_column=None,
                start_date=None,
                end_date=None,
                missing_values=None,
                duplicate_dates=None,
                numeric_values=None,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message=f"Could not read CSV: {exc}",
            )

        if df.empty:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=0,
                columns=list(df.columns),
                date_column=None,
                value_column=None,
                start_date=None,
                end_date=None,
                missing_values=None,
                duplicate_dates=None,
                numeric_values=None,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message="Dataset is empty.",
            )

        # ---------------------------------------------------------------------
        # Find date column
        # ---------------------------------------------------------------------

        date_column = self.find_date_column(
            df.columns
        )

        if date_column is None:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=len(df),
                columns=list(df.columns),
                date_column=None,
                value_column=None,
                start_date=None,
                end_date=None,
                missing_values=None,
                duplicate_dates=None,
                numeric_values=None,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message=(
                    "No recognizable date column was found."
                ),
            )

        # ---------------------------------------------------------------------
        # Find value column
        # ---------------------------------------------------------------------

        value_column = self.find_value_column(
            df=df,
            variable=variable,
        )

        if value_column is None:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=len(df),
                columns=list(df.columns),
                date_column=date_column,
                value_column=None,
                start_date=None,
                end_date=None,
                missing_values=None,
                duplicate_dates=None,
                numeric_values=None,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message=(
                    "No numeric value column matching the "
                    "requested variable was found."
                ),
            )

        # ---------------------------------------------------------------------
        # Parse dates
        # ---------------------------------------------------------------------

        parsed_dates = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        invalid_dates = int(
            parsed_dates.isna().sum()
        )

        if invalid_dates > 0:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=len(df),
                columns=list(df.columns),
                date_column=date_column,
                value_column=value_column,
                start_date=None,
                end_date=None,
                missing_values=None,
                duplicate_dates=None,
                numeric_values=None,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message=(
                    f"{invalid_dates} invalid date values "
                    "found in '{date_column}'."
                ),
            )

        # ---------------------------------------------------------------------
        # Numeric values
        # ---------------------------------------------------------------------

        numeric_values = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        numeric_invalid = int(
            numeric_values.isna().sum()
        )

        if numeric_invalid > 0:

            return ValidationResult(
                variable=variable,
                status="validation_failed",
                source_name=source["source_name"],
                source_type=source["source_type"],
                official_url=source["official_url"],
                local_file=str(file_path),
                rows=len(df),
                columns=list(df.columns),
                date_column=date_column,
                value_column=value_column,
                start_date=None,
                end_date=None,
                missing_values=numeric_invalid,
                duplicate_dates=None,
                numeric_values=len(df) - numeric_invalid,
                timegan_feature=TIMEGAN_MAPPING.get(variable),
                message=(
                    f"Value column '{value_column}' contains "
                    f"{numeric_invalid} non-numeric/missing values."
                ),
            )

        # ---------------------------------------------------------------------
        # Missing values
        # ---------------------------------------------------------------------

        missing_values = int(
            numeric_values.isna().sum()
        )

        # ---------------------------------------------------------------------
        # Duplicate dates
        # ---------------------------------------------------------------------

        duplicate_dates = int(
            parsed_dates.duplicated().sum()
        )

        # ---------------------------------------------------------------------
        # Sort dates
        # ---------------------------------------------------------------------

        start_date = parsed_dates.min()
        end_date = parsed_dates.max()

        start_date_string = (
            start_date.strftime("%Y-%m-%d")
            if pd.notna(start_date)
            else None
        )

        end_date_string = (
            end_date.strftime("%Y-%m-%d")
            if pd.notna(end_date)
            else None
        )

        # ---------------------------------------------------------------------
        # Check positive price/yield values where appropriate
        # ---------------------------------------------------------------------

        if variable in {
            "NIFTY50",
            "CRUDE_OIL",
            "USD_INR",
            "INDIA_VIX",
            "INDIA_10Y_YIELD",
        }:

            negative_count = int(
                (numeric_values < 0).sum()
            )

            if negative_count > 0:

                return ValidationResult(
                    variable=variable,
                    status="validation_failed",
                    source_name=source["source_name"],
                    source_type=source["source_type"],
                    official_url=source["official_url"],
                    local_file=str(file_path),
                    rows=len(df),
                    columns=list(df.columns),
                    date_column=date_column,
                    value_column=value_column,
                    start_date=start_date_string,
                    end_date=end_date_string,
                    missing_values=missing_values,
                    duplicate_dates=duplicate_dates,
                    numeric_values=len(numeric_values),
                    timegan_feature=TIMEGAN_MAPPING.get(variable),
                    message=(
                        f"Dataset contains {negative_count} "
                        "negative raw values. Review units/definition."
                    ),
                )

        # ---------------------------------------------------------------------
        # Duplicate dates are a warning.
        #
        # We do not automatically reject the dataset because some market
        # datasets can contain multiple observations per date.
        # ---------------------------------------------------------------------

        warning_parts = []

        if duplicate_dates > 0:
            warning_parts.append(
                f"{duplicate_dates} duplicate dates"
            )

        # ---------------------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------------------

        message = (
            "Real-data file found and validated successfully."
        )

        if warning_parts:
            message += (
                " Warnings: "
                + ", ".join(warning_parts)
                + "."
            )

        return ValidationResult(
            variable=variable,
            status="validated",
            source_name=source["source_name"],
            source_type=source["source_type"],
            official_url=source["official_url"],
            local_file=str(file_path),
            rows=len(df),
            columns=list(df.columns),
            date_column=date_column,
            value_column=value_column,
            start_date=start_date_string,
            end_date=end_date_string,
            missing_values=missing_values,
            duplicate_dates=duplicate_dates,
            numeric_values=len(numeric_values),
            timegan_feature=TIMEGAN_MAPPING.get(variable),
            message=message,
        )

    # =========================================================================
    # DATE COLUMN DETECTION
    # =========================================================================

    def find_date_column(
        self,
        columns,
    ) -> Optional[str]:

        # Exact candidates first
        for candidate in self.DATE_COLUMNS:

            for column in columns:

                if str(column).strip() == candidate:
                    return str(column)

        # Case-insensitive fallback
        for column in columns:

            normalized = (
                str(column)
                .strip()
                .lower()
                .replace("_", "")
                .replace(" ", "")
            )

            if normalized in {
                "date",
                "datetime",
                "timestamp",
                "time",
            }:
                return str(column)

        return None

    # =========================================================================
    # VALUE COLUMN DETECTION
    # =========================================================================

    def find_value_column(
        self,
        df: pd.DataFrame,
        variable: str,
    ) -> Optional[str]:

        source = SOURCE_REGISTRY[variable]

        columns = list(df.columns)

        # ---------------------------------------------------------------------
        # Exact expected names
        # ---------------------------------------------------------------------

        for expected in source["expected_value_columns"]:

            for column in columns:

                if str(column).strip() == expected:
                    if self._is_numeric_column(
                        df[column]
                    ):
                        return str(column)

        # ---------------------------------------------------------------------
        # Case-insensitive matching
        # ---------------------------------------------------------------------

        for expected in source["expected_value_columns"]:

            expected_normalized = self._normalize_name(
                expected
            )

            for column in columns:

                column_normalized = self._normalize_name(
                    column
                )

                if column_normalized == expected_normalized:

                    if self._is_numeric_column(
                        df[column]
                    ):
                        return str(column)

        # ---------------------------------------------------------------------
        # TimeGAN feature itself
        # ---------------------------------------------------------------------

        timegan_feature = TIMEGAN_MAPPING.get(
            variable
        )

        if timegan_feature:

            for column in columns:

                if (
                    self._normalize_name(column)
                    == self._normalize_name(timegan_feature)
                ):

                    if self._is_numeric_column(
                        df[column]
                    ):
                        return str(column)

        # ---------------------------------------------------------------------
        # Fallback:
        # Find numeric column that is not the date.
        #
        # This is intentionally conservative:
        # only one numeric candidate is accepted.
        # ---------------------------------------------------------------------

        date_column = self.find_date_column(
            columns
        )

        numeric_candidates = []

        for column in columns:

            if column == date_column:
                continue

            if self._is_numeric_column(
                df[column]
            ):
                numeric_candidates.append(
                    str(column)
                )

        if len(numeric_candidates) == 1:
            return numeric_candidates[0]

        return None

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _is_numeric_column(
        series: pd.Series,
    ) -> bool:

        converted = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid_count = int(
            converted.notna().sum()
        )

        return valid_count > 0

    @staticmethod
    def _normalize_name(
        value: Any,
    ) -> str:

        return re.sub(
            r"[^A-Z0-9]",
            "",
            str(value).upper(),
        )

    @staticmethod
    def _variable_tokens(
        variable: str,
    ) -> List[str]:

        token_map = {

            "NIFTY50": [
                "NIFTY",
                "NIFTY50",
                "NSEI",
            ],

            "CRUDE_OIL": [
                "CRUDE",
                "OIL",
                "WTI",
                "BRENT",
            ],

            "USD_INR": [
                "USD",
                "INR",
                "USDINR",
            ],

            "INDIA_VIX": [
                "VIX",
                "INDIAVIX",
            ],

            "INDIA_10Y_YIELD": [
                "10Y",
                "10YEAR",
                "YIELD",
                "GOVTBOND",
            ],
        }

        return token_map.get(
            variable,
            [variable],
        )

    def _dataset_contains_variable(
        self,
        variable: str,
        columns_upper: set[str],
    ) -> bool:

        # Direct variable
        variable_normalized = self._normalize_name(
            variable
        )

        normalized_columns = {
            self._normalize_name(column)
            for column in columns_upper
        }

        if variable_normalized in normalized_columns:
            return True

        # TimeGAN feature
        timegan_feature = TIMEGAN_MAPPING.get(
            variable
        )

        if timegan_feature:

            if (
                self._normalize_name(timegan_feature)
                in normalized_columns
            ):
                return True

        # Source value names
        source = SOURCE_REGISTRY[variable]

        for expected in source[
            "expected_value_columns"
        ]:

            if (
                self._normalize_name(expected)
                in normalized_columns
            ):
                return True

        return False

    @staticmethod
    def _overall_status(
        total: int,
        validated: int,
        missing: int,
        failed: int,
    ) -> str:

        if total == 0:
            return "no_variables_requested"

        if validated == total:
            return "validated"

        if validated > 0:
            return "partially_validated"

        if missing == total:
            return "data_missing"

        if failed == total:
            return "validation_failed"

        return "data_validation_required"

    @staticmethod
    def _next_step(
        overall_status: str,
        results: List[Dict[str, Any]],
    ) -> str:

        if overall_status == "validated":
            return (
                "All requested real datasets are validated. "
                "Proceed to TimeGAN V2 stress simulation."
            )

        if overall_status == "data_missing":
            return (
                "Download the missing real datasets from their "
                "registered sources and place them under the "
                "project data directory."
            )

        if overall_status == "partially_validated":
            return (
                "Resolve missing/failed datasets before running "
                "the final stress pipeline."
            )

        return (
            "Review the validation results before continuing."
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Optional[str | Path] = None,
    ) -> Path:

        if output_path is None:
            output_path = (
                self.output_root
                / "stock_data_discovery.json"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path


# =============================================================================
# DIRECT TEST
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("MACROSTRESS-GAN")
    print("STOCK DATA DISCOVERY & VALIDATION AGENT")
    print("=" * 80)

    institution_config = {
        "type": "stock_market",

        "portfolio_amount": 100000000,

        "currency": "INR",

        "variables": [
            "NIFTY50",
            "CRUDE_OIL",
            "USD_INR",
            "INDIA_VIX",
            "INDIA_10Y_YIELD",
        ],
    }

    agent = StockDataDiscoveryAgent()

    result = agent.discover(
        institution_config=institution_config,
        save_result=True,
    )

    print("\nSTATUS")
    print("-" * 80)
    print(result["status"])

    print("\nSUMMARY")
    print("-" * 80)

    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )

    print("\nVARIABLE VALIDATION")
    print("-" * 80)

    for item in result["results"]:

        print(
            f"{item['variable']:20s} | "
            f"{item['status']:42s} | "
            f"{item['local_file']}"
        )

    print("\nNEXT STEP")
    print("-" * 80)
    print(result["next_step"])

    print("\nSaved:")
    print(
        PROJECT_ROOT
        / "outputs"
        / "stock_data_discovery.json"
    )


if __name__ == "__main__":
    main()