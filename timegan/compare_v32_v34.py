from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_ROOT = ROOT / "outputs" / "validation"

# ------------------------------------------------------------
# V3.2
# ------------------------------------------------------------

V32_DIR = VALIDATION_ROOT / "v3_v32_backup"

V32_EXTREME_DIR = V32_DIR / "extreme"

V32_COMPLETE_JSON = (
    V32_DIR / "v3_validation_summary.json"
)

V32_RISK_CSV = (
    V32_DIR / "v3_risk_metrics.csv"
)

# ------------------------------------------------------------
# V3.4
# ------------------------------------------------------------

V34_DIR = (
    VALIDATION_ROOT / "v34" / "complete"
)

V34_EXTREME_DIR = (
    VALIDATION_ROOT / "v34" / "extreme"
)

V34_COMPLETE_JSON = (
    V34_DIR / "validation_summary_v34.json"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUT_DIR = (
    VALIDATION_ROOT / "v32_vs_v34"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.
    """

    if value is None:
        return np.nan

    try:

        if pd.isna(value):
            return np.nan

        return float(value)

    except (TypeError, ValueError):

        return np.nan


def normalize_text(value):
    """
    Normalize text for flexible column/metric matching.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def load_json(path):
    """
    Load JSON safely.
    """

    if not path.exists():

        print(
            f"WARNING: File not found: {path}"
        )

        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"ERROR loading JSON {path}: {e}"
        )

        return {}


def get_metric(data, key):
    """
    Safely retrieve a metric from a dictionary.
    """

    if not isinstance(data, dict):

        return np.nan

    value = data.get(key)

    return safe_float(value)


# ============================================================
# LOAD V3.2 COMPLETE SUMMARY
# ============================================================

def load_v32_complete():

    path = V32_COMPLETE_JSON

    print(
        f"V3.2 complete summary: {path}"
    )

    data = load_json(path)

    if data:

        print(
            "V3.2 complete summary loaded successfully."
        )

    return path, data


# ============================================================
# LOAD V3.4 COMPLETE SUMMARY
# ============================================================

def load_v34_complete():

    path = V34_COMPLETE_JSON

    print(
        f"V3.4 complete summary: {path}"
    )

    data = load_json(path)

    if data:

        print(
            "V3.4 complete summary loaded successfully."
        )

    return path, data


# ============================================================
# V3.2 METRIC MAPPING
# ============================================================

def extract_v32_complete_metrics(data):
    """
    V3.2 and V3.4 use different JSON metric names.

    This function converts V3.2 names into the common
    comparison names used by this script.
    """

    metrics = {}

    # --------------------------------------------------------
    # V3.2 JSON
    # --------------------------------------------------------

    metrics[
        "mean_absolute_correlation_error"
    ] = get_metric(
        data,
        "correlation_mae"
    )

    metrics[
        "average_lag1_error"
    ] = get_metric(
        data,
        "mean_temporal_lag1_difference"
    )

    metrics[
        "average_30d_volatility_error"
    ] = get_metric(
        data,
        "mean_volatility_difference"
    )

    metrics[
        "average_ks_statistic"
    ] = get_metric(
        data,
        "mean_ks_statistic"
    )

    metrics[
        "mean_wasserstein_distance"
    ] = get_metric(
        data,
        "mean_wasserstein_distance"
    )

    # These two metrics were not stored in the V3.2 JSON.
    metrics[
        "average_quantile_absolute_error"
    ] = np.nan

    metrics[
        "mean_absolute_error"
    ] = np.nan

    metrics[
        "std_absolute_error"
    ] = np.nan

    # Risk metrics are loaded separately from
    # v3_risk_metrics.csv.
    metrics[
        "nifty_var95_synthetic"
    ] = np.nan

    metrics[
        "nifty_es95_synthetic"
    ] = np.nan

    metrics[
        "nifty_mean_mdd_synthetic"
    ] = np.nan

    metrics[
        "nifty_median_mdd_synthetic"
    ] = np.nan

    metrics[
        "nifty_worst_mdd_synthetic"
    ] = np.nan

    return metrics


# ============================================================
# V3.4 METRIC MAPPING
# ============================================================

def extract_v34_complete_metrics(data):
    """
    Extract V3.4 metrics using the names generated by
    validate_v34.py.
    """

    metrics = {}

    metric_names = [

        "mean_absolute_error",

        "std_absolute_error",

        "mean_absolute_correlation_error",

        "average_lag1_error",

        "average_30d_volatility_error",

        "average_ks_statistic",

        "average_quantile_absolute_error",

        "nifty_var95_synthetic",

        "nifty_es95_synthetic",

        "nifty_mean_mdd_synthetic",

        "nifty_median_mdd_synthetic",

        "nifty_worst_mdd_synthetic",

    ]

    for metric in metric_names:

        metrics[metric] = get_metric(
            data,
            metric
        )

    # V3.4 validation may contain Wasserstein
    # in some versions. Keep it if available.
    metrics[
        "mean_wasserstein_distance"
    ] = get_metric(
        data,
        "mean_wasserstein_distance"
    )

    return metrics


# ============================================================
# LOAD V3.2 RISK CSV
# ============================================================

def load_v32_risk_metrics():

    path = V32_RISK_CSV

    print()
    print(
        f"V3.2 risk metrics: {path}"
    )

    if not path.exists():

        print(
            "WARNING: V3.2 risk metrics CSV not found."
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        print(
            "V3.2 risk metrics loaded successfully."
        )

        print(
            f"Rows: {len(df)}"
        )

        print(
            f"Columns: {list(df.columns)}"
        )

        return df

    except Exception as e:

        print(
            f"ERROR loading V3.2 risk CSV: {e}"
        )

        return pd.DataFrame()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    candidates
):
    """
    Find a column using normalized candidate names.
    """

    if df.empty:

        return None

    normalized_columns = {}

    for col in df.columns:

        normalized_columns[
            normalize_text(col)
        ] = col

    # Exact normalized match
    for candidate in candidates:

        normalized_candidate = (
            normalize_text(candidate)
        )

        if normalized_candidate in normalized_columns:

            return normalized_columns[
                normalized_candidate
            ]

    # Partial match
    for candidate in candidates:

        normalized_candidate = (
            normalize_text(candidate)
        )

        for normalized_col, original_col in (
            normalized_columns.items()
        ):

            if (
                normalized_candidate in normalized_col
                or
                normalized_col in normalized_candidate
            ):

                return original_col

    return None


# ============================================================
# FIND METRIC ROW
# ============================================================

def find_metric_row(
    df,
    keywords
):
    """
    Find a row containing one or more metric keywords.
    """

    if df.empty:

        return None

    for index, row in df.iterrows():

        combined_text = " ".join(
            str(value)
            for value in row.values
        ).lower()

        if all(
            keyword.lower() in combined_text
            for keyword in keywords
        ):

            return index

    return None


# ============================================================
# EXTRACT SYNTHETIC RISK VALUE
# ============================================================

def extract_synthetic_risk_value(
    df,
    keywords
):
    """
    Extract the synthetic risk value from the V3.2
    risk-metrics CSV using flexible matching.
    """

    if df.empty:

        return np.nan

    # --------------------------------------------------------
    # Try to identify synthetic column
    # --------------------------------------------------------

    synthetic_column = find_column(
        df,
        [
            "Synthetic",
            "Synthetic_Value",
            "Synthetic Value",
            "Synthetic_Value_",
            "Synthetic_V3",
            "Generated",
            "Generated_Value",
        ]
    )

    # --------------------------------------------------------
    # Find row containing metric
    # --------------------------------------------------------

    row_index = find_metric_row(
        df,
        keywords
    )

    if row_index is None:

        return np.nan

    row = df.loc[row_index]

    # --------------------------------------------------------
    # If a synthetic column exists
    # --------------------------------------------------------

    if synthetic_column is not None:

        return safe_float(
            row[synthetic_column]
        )

    # --------------------------------------------------------
    # Otherwise inspect numeric values
    # --------------------------------------------------------

    numeric_values = []

    for value in row.values:

        number = safe_float(value)

        if not np.isnan(number):

            numeric_values.append(
                number
            )

    # Usually the first/second numeric value
    # corresponds to real/synthetic.
    if len(numeric_values) >= 2:

        return numeric_values[-1]

    if len(numeric_values) == 1:

        return numeric_values[0]

    return np.nan


# ============================================================
# EXTRACT V3.2 RISK METRICS
# ============================================================

def extract_v32_risk_metrics(
    df
):
    """
    Extract V3.2 NIFTY risk metrics.

    Supports different possible labels in the CSV.
    """

    metrics = {}

    # --------------------------------------------------------
    # VaR 95
    # --------------------------------------------------------

    metrics[
        "nifty_var95_synthetic"
    ] = extract_synthetic_risk_value(
        df,
        [
            "var",
            "95"
        ]
    )

    # --------------------------------------------------------
    # Expected Shortfall / ES 95
    # --------------------------------------------------------

    metrics[
        "nifty_es95_synthetic"
    ] = extract_synthetic_risk_value(
        df,
        [
            "es",
            "95"
        ]
    )

    # If ES wasn't found, try Expected Shortfall.
    if np.isnan(
        metrics["nifty_es95_synthetic"]
    ):

        metrics[
            "nifty_es95_synthetic"
        ] = extract_synthetic_risk_value(
            df,
            [
                "expected",
                "shortfall"
            ]
        )

    # --------------------------------------------------------
    # Mean MDD
    # --------------------------------------------------------

    metrics[
        "nifty_mean_mdd_synthetic"
    ] = extract_synthetic_risk_value(
        df,
        [
            "mean",
            "mdd"
        ]
    )

    # Try maximum drawdown wording.
    if np.isnan(
        metrics["nifty_mean_mdd_synthetic"]
    ):

        metrics[
            "nifty_mean_mdd_synthetic"
        ] = extract_synthetic_risk_value(
            df,
            [
                "mean",
                "drawdown"
            ]
        )

    # --------------------------------------------------------
    # Median MDD
    # --------------------------------------------------------

    metrics[
        "nifty_median_mdd_synthetic"
    ] = extract_synthetic_risk_value(
        df,
        [
            "median",
            "mdd"
        ]
    )

    if np.isnan(
        metrics["nifty_median_mdd_synthetic"]
    ):

        metrics[
            "nifty_median_mdd_synthetic"
        ] = extract_synthetic_risk_value(
            df,
            [
                "median",
                "drawdown"
            ]
        )

    # --------------------------------------------------------
    # Worst MDD
    # --------------------------------------------------------

    metrics[
        "nifty_worst_mdd_synthetic"
    ] = extract_synthetic_risk_value(
        df,
        [
            "worst",
            "mdd"
        ]
    )

    if np.isnan(
        metrics["nifty_worst_mdd_synthetic"]
    ):

        metrics[
            "nifty_worst_mdd_synthetic"
        ] = extract_synthetic_risk_value(
            df,
            [
                "worst",
                "drawdown"
            ]
        )

    return metrics


# ============================================================
# COMPLETE STATISTICAL COMPARISON
# ============================================================

def create_complete_comparison(
    v32_metrics,
    v34_metrics
):

    metrics = [

        "mean_absolute_error",

        "std_absolute_error",

        "mean_absolute_correlation_error",

        "average_lag1_error",

        "average_30d_volatility_error",

        "average_ks_statistic",

        "average_quantile_absolute_error",

        "mean_wasserstein_distance",

        "nifty_var95_synthetic",

        "nifty_es95_synthetic",

        "nifty_mean_mdd_synthetic",

        "nifty_median_mdd_synthetic",

        "nifty_worst_mdd_synthetic",

    ]

    rows = []

    for metric in metrics:

        v32_value = safe_float(
            v32_metrics.get(metric)
        )

        v34_value = safe_float(
            v34_metrics.get(metric)
        )

        if (
            not np.isnan(v32_value)
            and
            not np.isnan(v34_value)
        ):

            difference = (
                v32_value - v34_value
            )

        else:

            difference = np.nan

        rows.append(
            {
                "Metric": metric,

                "V3.2": v32_value,

                "V3.4": v34_value,

                "V3.2_minus_V3.4":
                    difference,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD EXTREME SUMMARY
# ============================================================

def load_extreme(version):

    if version == "V3.2":

        path = (
            V32_EXTREME_DIR
            / "extreme_validation_summary.json"
        )

    else:

        path = (
            V34_EXTREME_DIR
            / "extreme_validation_summary.json"
        )

    print(
        f"{version} extreme summary: {path}"
    )

    data = load_json(path)

    if data:

        print(
            f"{version} extreme summary loaded successfully."
        )

    return path, data


# ============================================================
# EXTRACT EXTREME SUMMARY
# ============================================================

def extract_extreme_summary(
    version,
    data
):

    if not isinstance(data, dict):

        return {}

    real_shape = data.get(
        "real_shape"
    )

    synthetic_shape = data.get(
        "synthetic_shape"
    )

    result = {

        "Version":
            version,

        "Model":
            data.get("model"),

        "Validation_Type":
            data.get("validation_type"),

        "Real_Sequences":
            real_shape[0]
            if isinstance(real_shape, list)
            and len(real_shape) > 0
            else np.nan,

        "Synthetic_Sequences":
            synthetic_shape[0]
            if isinstance(synthetic_shape, list)
            and len(synthetic_shape) > 0
            else np.nan,

        "Real_Global_Min":
            safe_float(
                data.get(
                    "real_global_min"
                )
            ),

        "Real_Global_Max":
            safe_float(
                data.get(
                    "real_global_max"
                )
            ),

        "Synthetic_Global_Min":
            safe_float(
                data.get(
                    "synthetic_global_min"
                )
            ),

        "Synthetic_Global_Max":
            safe_float(
                data.get(
                    "synthetic_global_max"
                )
            ),
    }

    features = data.get(
        "features"
    )

    if isinstance(
        features,
        list
    ):

        result[
            "Feature_Count"
        ] = len(features)

        result[
            "Features"
        ] = ", ".join(
            str(x)
            for x in features
        )

    elif isinstance(
        features,
        dict
    ):

        result[
            "Feature_Count"
        ] = len(features)

        result[
            "Features"
        ] = ", ".join(
            str(x)
            for x in features.keys()
        )

    else:

        result[
            "Feature_Count"
        ] = np.nan

        result[
            "Features"
        ] = ""

    return result


# ============================================================
# COMBINE EXTREME CSV
# ============================================================

def combine_extreme_csv(
    filename
):

    v32_path = (
        V32_EXTREME_DIR
        / filename
    )

    v34_path = (
        V34_EXTREME_DIR
        / filename
    )

    frames = []

    # --------------------------------------------------------
    # V3.2
    # --------------------------------------------------------

    if v32_path.exists():

        try:

            df = pd.read_csv(
                v32_path
            )

            df.insert(
                0,
                "Version",
                "V3.2"
            )

            frames.append(df)

        except Exception as e:

            print(
                f"ERROR reading {v32_path}: {e}"
            )

    # --------------------------------------------------------
    # V3.4
    # --------------------------------------------------------

    if v34_path.exists():

        try:

            df = pd.read_csv(
                v34_path
            )

            df.insert(
                0,
                "Version",
                "V3.4"
            )

            frames.append(df)

        except Exception as e:

            print(
                f"ERROR reading {v34_path}: {e}"
            )

    if not frames:

        print(
            f"No files found for {filename}"
        )

        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# SAVE DATAFRAME
# ============================================================

def save_dataframe(
    df,
    filename
):

    output_path = (
        OUT_DIR / filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "MACROSTRESS-GAN V3.2 vs V3.4 "
        "VALIDATION COMPARISON"
    )

    print("=" * 70)

    # ========================================================
    # 1. COMPLETE STATISTICAL VALIDATION
    # ========================================================

    print()
    print("-" * 70)

    print(
        "COMPLETE STATISTICAL VALIDATION"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Load V3.2 JSON
    # --------------------------------------------------------

    v32_path, v32_json = (
        load_v32_complete()
    )

    # --------------------------------------------------------
    # Load V3.4 JSON
    # --------------------------------------------------------

    v34_path, v34_json = (
        load_v34_complete()
    )

    # --------------------------------------------------------
    # Convert to common metric names
    # --------------------------------------------------------

    v32_metrics = (
        extract_v32_complete_metrics(
            v32_json
        )
    )

    v34_metrics = (
        extract_v34_complete_metrics(
            v34_json
        )
    )

    # --------------------------------------------------------
    # Load V3.2 risk CSV
    # --------------------------------------------------------

    v32_risk_df = (
        load_v32_risk_metrics()
    )

    # --------------------------------------------------------
    # Extract V3.2 risk metrics
    # --------------------------------------------------------

    v32_risk_metrics = (
        extract_v32_risk_metrics(
            v32_risk_df
        )
    )

    # --------------------------------------------------------
    # Merge V3.2 risk metrics
    # --------------------------------------------------------

    v32_metrics.update(
        v32_risk_metrics
    )

    # --------------------------------------------------------
    # Create comparison
    # --------------------------------------------------------

    comparison = (
        create_complete_comparison(
            v32_metrics,
            v34_metrics
        )
    )

    # --------------------------------------------------------
    # Save complete comparison
    # --------------------------------------------------------

    complete_output = (
        OUT_DIR
        / "v32_vs_v34_complete_comparison.csv"
    )

    comparison.to_csv(
        complete_output,
        index=False
    )

    print()
    print(
        f"Saved: {complete_output}"
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # 2. EXTREME VALIDATION
    # ========================================================

    print()
    print("-" * 70)

    print(
        "EXTREME VALIDATION"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Load V3.2 extreme
    # --------------------------------------------------------

    (
        v32_extreme_path,
        v32_extreme
    ) = load_extreme(
        "V3.2"
    )

    # --------------------------------------------------------
    # Load V3.4 extreme
    # --------------------------------------------------------

    (
        v34_extreme_path,
        v34_extreme
    ) = load_extreme(
        "V3.4"
    )

    extreme_rows = []

    if v32_extreme:

        extreme_rows.append(
            extract_extreme_summary(
                "V3.2",
                v32_extreme
            )
        )

    if v34_extreme:

        extreme_rows.append(
            extract_extreme_summary(
                "V3.4",
                v34_extreme
            )
        )

    if extreme_rows:

        extreme_df = pd.DataFrame(
            extreme_rows
        )

        extreme_output = (
            OUT_DIR
            / "v32_vs_v34_extreme_summary.csv"
        )

        extreme_df.to_csv(
            extreme_output,
            index=False
        )

        print()
        print(
            f"Saved: {extreme_output}"
        )

        print()
        print(
            extreme_df.to_string(
                index=False
            )
        )

    # ========================================================
    # 3. EXTREME CSV COMPARISONS
    # ========================================================

    extreme_files = [

        "feature_extreme_event_rates.csv",

        "joint_stress_events.csv",

        "sequence_extreme_events.csv",

        "absolute_extremes.csv",

        "extreme_event_clustering.csv",
    ]

    print()
    print("-" * 70)

    print(
        "EXTREME CSV COMPARISONS"
    )

    print("-" * 70)

    for filename in extreme_files:

        df = combine_extreme_csv(
            filename
        )

        if df.empty:

            continue

        output_path = (
            OUT_DIR
            / f"v32_vs_v34_{filename}"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"Saved: {output_path}"
        )

    # ========================================================
    # 4. SAVE V3.2 RISK COMPARISON
    # ========================================================

    print()
    print("-" * 70)

    print(
        "RISK METRIC COMPARISON"
    )

    print("-" * 70)

    risk_metrics = [

        "nifty_var95_synthetic",

        "nifty_es95_synthetic",

        "nifty_mean_mdd_synthetic",

        "nifty_median_mdd_synthetic",

        "nifty_worst_mdd_synthetic",
    ]

    risk_rows = []

    for metric in risk_metrics:

        v32_value = safe_float(
            v32_metrics.get(
                metric
            )
        )

        v34_value = safe_float(
            v34_metrics.get(
                metric
            )
        )

        if (
            not np.isnan(v32_value)
            and
            not np.isnan(v34_value)
        ):

            difference = (
                v32_value - v34_value
            )

        else:

            difference = np.nan

        risk_rows.append(
            {
                "Metric":
                    metric,

                "V3.2":
                    v32_value,

                "V3.4":
                    v34_value,

                "V3.2_minus_V3.4":
                    difference,
            }
        )

    risk_comparison = pd.DataFrame(
        risk_rows
    )

    risk_output = (
        OUT_DIR
        / "v32_vs_v34_risk_comparison.csv"
    )

    risk_comparison.to_csv(
        risk_output,
        index=False
    )

    print(
        f"Saved: {risk_output}"
    )

    print()
    print(
        risk_comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # 5. SAVE NORMALIZED METRICS JSON
    # ========================================================

    normalized_metrics = {

        "V3.2": {
            key: (
                None
                if np.isnan(
                    safe_float(value)
                )
                else float(value)
            )
            for key, value
            in v32_metrics.items()
        },

        "V3.4": {
            key: (
                None
                if np.isnan(
                    safe_float(value)
                )
                else float(value)
            )
            for key, value
            in v34_metrics.items()
        },
    }

    normalized_json_path = (
        OUT_DIR
        / "normalized_v32_v34_metrics.json"
    )

    with open(
        normalized_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            normalized_metrics,
            f,
            indent=4
        )

    print()
    print(
        f"Saved: {normalized_json_path}"
    )

    # ========================================================
    # 6. FINAL SUMMARY JSON
    # ========================================================

    summary = {

        "comparison":
            "MacroStress-GAN TimeGAN V3.2 vs V3.4",

        "v32_model":
            v32_json.get(
                "model"
            ),

        "v32_sequence_length":
            v32_json.get(
                "sequence_length"
            ),

        "v32_features":
            v32_json.get(
                "features"
            ),

        "v34_model":
            v34_json.get(
                "model"
            ),

        "v32_complete_summary":
            str(
                v32_path.relative_to(
                    ROOT
                )
            ),

        "v34_complete_summary":
            str(
                v34_path.relative_to(
                    ROOT
                )
            ),

        "v32_risk_metrics":
            str(
                V32_RISK_CSV.relative_to(
                    ROOT
                )
            ),

        "v32_extreme_summary":
            str(
                v32_extreme_path.relative_to(
                    ROOT
                )
            ),

        "v34_extreme_summary":
            str(
                v34_extreme_path.relative_to(
                    ROOT
                )
            ),

        "output_directory":
            str(
                OUT_DIR.relative_to(
                    ROOT
                )
            ),

        "files_generated": [

            "v32_vs_v34_complete_comparison.csv",

            "v32_vs_v34_risk_comparison.csv",

            "v32_vs_v34_extreme_summary.csv",

            "v32_vs_v34_feature_extreme_event_rates.csv",

            "v32_vs_v34_joint_stress_events.csv",

            "v32_vs_v34_sequence_extreme_events.csv",

            "v32_vs_v34_absolute_extremes.csv",

            "v32_vs_v34_extreme_event_clustering.csv",

            "normalized_v32_v34_metrics.json",

            "v32_vs_v34_summary.json",
        ],
    }

    summary_path = (
        OUT_DIR
        / "v32_vs_v34_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # ========================================================
    # 7. FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "COMPARISON COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        f"Output directory:"
    )

    print(
        OUT_DIR
    )

    print()

    print(
        "Generated files:"
    )

    for file in sorted(
        OUT_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f"  - {file.name}"
            )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()