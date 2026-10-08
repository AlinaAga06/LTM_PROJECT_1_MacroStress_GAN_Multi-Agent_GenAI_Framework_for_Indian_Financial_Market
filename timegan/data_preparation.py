
"""
MacroStress-GAN
============================================================
TimeGAN Data Preparation Module

Purpose:
    - Load processed financial market data
    - Select 5 core market variables
    - Clean and validate data
    - Normalize financial variables
    - Create 30-day time-series sequences
    - Save sequences for TimeGAN training
    - Save scaler for inverse transformation later

Input:
    data/processed/macro_stress_5vars_processed.csv

Output:
    data/processed/timegan_sequences.npy
    data/processed/timegan_scaler.pkl
    data/processed/timegan_training_data.csv
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_stress_5vars_processed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SEQUENCE_PATH = (
    OUTPUT_DIR
    / "timegan_sequences.npy"
)

SCALER_PATH = (
    OUTPUT_DIR
    / "timegan_scaler.pkl"
)

TRAINING_DATA_PATH = (
    OUTPUT_DIR
    / "timegan_training_data.csv"
)


# ============================================================
# 2. TIMEGAN CONFIGURATION
# ============================================================

# These are the five core market variables
# that TimeGAN will learn.

FEATURES = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_10Y_YIELD",
]

# TimeGAN will learn 30 consecutive trading days
# at a time.

SEQUENCE_LENGTH = 30


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    """
    Load the processed financial market dataset.
    """

    print()
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    # Check whether the input file exists
    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "\nDataset not found.\n\n"
            f"Expected file:\n{DATA_PATH}\n\n"
            "Please make sure your processed CSV exists at:\n"
            "data/processed/macro_stress_5vars_processed.csv"
        )

    # Read CSV
    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print()
    print(f"File      : {DATA_PATH}")
    print(f"Rows      : {data.shape[0]}")
    print(f"Columns   : {data.shape[1]}")

    return data


# ============================================================
# 4. VALIDATE REQUIRED FEATURES
# ============================================================

def validate_features(data):
    """
    Verify that all five required market variables
    exist in the dataset.
    """

    print()
    print("=" * 70)
    print("STEP 2: VALIDATING FEATURES")
    print("=" * 70)

    missing_features = []

    for feature in FEATURES:

        if feature not in data.columns:

            missing_features.append(feature)

    # If any feature is missing, stop execution
    if missing_features:

        print()
        print("ERROR: Required features are missing:")

        for feature in missing_features:
            print(f"  ✗ {feature}")

        raise ValueError(
            "\nTimeGAN cannot continue because required "
            "features are missing."
        )

    print()
    print("All required features are available:")

    for feature in FEATURES:

        print(f"  ✓ {feature}")


# ============================================================
# 5. PREPROCESS DATA
# ============================================================

def preprocess_data(data):
    """
    Clean and prepare the financial market data.
    """

    print()
    print("=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)

    data = data.copy()

    # --------------------------------------------------------
    # DATE PROCESSING
    # --------------------------------------------------------

    if "Date" in data.columns:

        # Convert Date to datetime
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        # Remove rows with invalid dates
        invalid_dates = data["Date"].isna().sum()

        if invalid_dates > 0:

            print(
                f"Removing {invalid_dates} rows "
                "with invalid dates."
            )

            data = data.dropna(
                subset=["Date"]
            )

        # Sort by date
        data = data.sort_values(
            by="Date"
        )

        # Remove duplicate dates
        duplicate_dates = data["Date"].duplicated().sum()

        if duplicate_dates > 0:

            print(
                f"Removing {duplicate_dates} "
                "duplicate date rows."
            )

            data = data.drop_duplicates(
                subset=["Date"],
                keep="first"
            )

        # Reset index
        data = data.reset_index(
            drop=True
        )

        print()
        print("Date processing completed.")

        print(
            f"Start date : "
            f"{data['Date'].min().date()}"
        )

        print(
            f"End date   : "
            f"{data['Date'].max().date()}"
        )

    else:

        print(
            "WARNING: Date column not found."
        )


    # --------------------------------------------------------
    # SELECT FIVE CORE VARIABLES
    # --------------------------------------------------------

    selected_columns = FEATURES.copy()

    if "Date" in data.columns:

        selected_columns.insert(
            0,
            "Date"
        )

    data = data[
        selected_columns
    ].copy()

    print()
    print("Selected variables:")

    for feature in FEATURES:

        print(f"  ✓ {feature}")


    # --------------------------------------------------------
    # CONVERT VARIABLES TO NUMERIC
    # --------------------------------------------------------

    for feature in FEATURES:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce"
        )


    # --------------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------------

    print()
    print("Missing values before cleaning:")

    missing_before = (
        data[FEATURES]
        .isna()
        .sum()
    )

    for feature in FEATURES:

        print(
            f"  {feature:<22} "
            f"{missing_before[feature]}"
        )


    # --------------------------------------------------------
    # REPLACE INFINITE VALUES
    # --------------------------------------------------------

    data[FEATURES] = data[
        FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # FORWARD FILL
    # --------------------------------------------------------

    data[FEATURES] = (
        data[FEATURES]
        .ffill()
    )


    # --------------------------------------------------------
    # BACKWARD FILL
    # --------------------------------------------------------

    data[FEATURES] = (
        data[FEATURES]
        .bfill()
    )


    # --------------------------------------------------------
    # REMOVE REMAINING MISSING VALUES
    # --------------------------------------------------------

    rows_before_drop = len(data)

    data = data.dropna(
        subset=FEATURES
    )

    rows_removed = (
        rows_before_drop
        - len(data)
    )

    if rows_removed > 0:

        print()
        print(
            f"Removed {rows_removed} "
            "rows with remaining missing values."
        )


    # Reset index
    data = data.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("Missing values after cleaning:")

    missing_after = (
        data[FEATURES]
        .isna()
        .sum()
    )

    for feature in FEATURES:

        print(
            f"  {feature:<22} "
            f"{missing_after[feature]}"
        )


    # --------------------------------------------------------
    # CHECK NEGATIVE / ZERO VALUES
    # --------------------------------------------------------

    print()
    print("Checking feature ranges:")

    for feature in FEATURES:

        minimum = data[feature].min()
        maximum = data[feature].max()

        print(
            f"  {feature:<22} "
            f"Min={minimum:.4f}  "
            f"Max={maximum:.4f}"
        )


    print()
    print(
        f"Rows after preprocessing: "
        f"{len(data)}"
    )

    return data


# ============================================================
# 6. NORMALIZE DATA
# ============================================================

def normalize_data(data):
    """
    Normalize the five market variables using
    Min-Max normalization.

    Formula:

        X_scaled =
        (X - X_min) /
        (X_max - X_min)

    Result:

        approximately 0 to 1
    """

    print()
    print("=" * 70)
    print("STEP 4: DATA NORMALIZATION")
    print("=" * 70)

    # Create scaler
    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # Fit and transform
    normalized_values = (
        scaler.fit_transform(
            data[FEATURES]
        )
    )

    # Convert back to DataFrame
    normalized_data = pd.DataFrame(
        normalized_values,
        columns=FEATURES
    )

    # Keep same index
    normalized_data.index = (
        data.index
    )

    print(
        "Min-Max normalization completed."
    )

    print()
    print("Normalized feature ranges:")

    for feature in FEATURES:

        minimum = (
            normalized_data[feature]
            .min()
        )

        maximum = (
            normalized_data[feature]
            .max()
        )

        print(
            f"  {feature:<22} "
            f"{minimum:.4f} → "
            f"{maximum:.4f}"
        )

    return (
        normalized_data,
        scaler
    )


# ============================================================
# 7. CREATE 30-DAY SEQUENCES
# ============================================================

def create_sequences(normalized_data):
    """
    Convert normalized daily observations
    into overlapping 30-day sequences.

    Example:

        Days 1-30   -> Sequence 1
        Days 2-31   -> Sequence 2
        Days 3-32   -> Sequence 3
        ...
        Days 3989-4018 -> Final sequence

    Output shape:

        (number_of_sequences, 30, 5)
    """

    print()
    print("=" * 70)
    print("STEP 5: CREATING TIMEGAN SEQUENCES")
    print("=" * 70)

    # Convert DataFrame to NumPy array
    values = normalized_data[
        FEATURES
    ].values

    number_of_rows = len(values)

    # Make sure enough data exists
    if number_of_rows < SEQUENCE_LENGTH:

        raise ValueError(
            f"\nNot enough observations.\n"
            f"Available rows: {number_of_rows}\n"
            f"Required rows: {SEQUENCE_LENGTH}"
        )


    # List to store sequences
    sequences = []


    # --------------------------------------------------------
    # SLIDING WINDOW
    # --------------------------------------------------------

    for start_index in range(
        number_of_rows
        - SEQUENCE_LENGTH
        + 1
    ):

        end_index = (
            start_index
            + SEQUENCE_LENGTH
        )

        sequence = values[
            start_index:end_index
        ]

        sequences.append(
            sequence
        )


    # Convert list to NumPy array
    sequences = np.array(
        sequences,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    print(
        "30-day sequences created successfully."
    )

    print()
    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH} days"
    )

    print(
        f"Number of features : "
        f"{len(FEATURES)}"
    )

    print(
        f"Number of sequences : "
        f"{len(sequences)}"
    )

    print(
        f"Sequence shape : "
        f"{sequences.shape}"
    )

    return sequences


# ============================================================
# 8. SAVE SCALER
# ============================================================

def save_scaler(scaler):
    """
    Save the fitted Min-Max scaler.

    This scaler will later be used to convert
    generated TimeGAN values back to their
    original financial units.
    """

    print()
    print("=" * 70)
    print("STEP 6: SAVING SCALER")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save scaler
    with open(
        SCALER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    print(
        "Scaler saved successfully."
    )

    print(
        f"Location:\n{SCALER_PATH}"
    )


# ============================================================
# 9. SAVE TIMEGAN SEQUENCES
# ============================================================

def save_sequences(sequences):
    """
    Save TimeGAN sequences as a NumPy file.
    """

    print()
    print("=" * 70)
    print("STEP 7: SAVING TIMEGAN SEQUENCES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        SEQUENCE_PATH,
        sequences
    )

    print(
        "TimeGAN sequences saved successfully."
    )

    print(
        f"Location:\n{SEQUENCE_PATH}"
    )


# ============================================================
# 10. SAVE NORMALIZED TRAINING DATA
# ============================================================

def save_training_data(
    normalized_data,
    original_data
):
    """
    Save normalized daily observations
    as a CSV file.

    Date is retained for reference.
    """

    print()
    print("=" * 70)
    print("STEP 8: SAVING NORMALIZED TRAINING DATA")
    print("=" * 70)

    # Copy normalized data
    output_data = (
        normalized_data.copy()
    )

    # Add Date column if available
    if "Date" in original_data.columns:

        output_data.insert(
            0,
            "Date",
            original_data["Date"].values
        )

    # Save CSV
    output_data.to_csv(
        TRAINING_DATA_PATH,
        index=False
    )

    print(
        "Normalized training data saved."
    )

    print(
        f"Location:\n{TRAINING_DATA_PATH}"
    )


# ============================================================
# 11. DISPLAY SAMPLE SEQUENCE
# ============================================================

def display_sample(sequences):
    """
    Display the first five days of the first
    TimeGAN sequence.
    """

    print()
    print("=" * 70)
    print("STEP 9: SAMPLE TIMEGAN SEQUENCE")
    print("=" * 70)

    # First sequence
    first_sequence = (
        sequences[0]
    )

    print()
    print(
        f"First sequence shape: "
        f"{first_sequence.shape}"
    )

    # First five days
    sample = pd.DataFrame(
        first_sequence[:5],
        columns=FEATURES
    )

    print()
    print(
        "First 5 days of first sequence:"
    )

    print(
        sample.to_string(
            index=False
        )
    )


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

def validate_output(sequences):
    """
    Perform final validation of the TimeGAN dataset.
    """

    print()
    print("=" * 70)
    print("STEP 10: FINAL VALIDATION")
    print("=" * 70)

    # Check data type
    print(
        f"Data type: "
        f"{sequences.dtype}"
    )

    # Check shape
    print(
        f"Shape: "
        f"{sequences.shape}"
    )

    # Check NaN
    nan_count = np.isnan(
        sequences
    ).sum()

    print(
        f"NaN values: "
        f"{nan_count}"
    )

    # Check infinity
    inf_count = np.isinf(
        sequences
    ).sum()

    print(
        f"Infinite values: "
        f"{inf_count}"
    )

    # Check normalized range
    minimum = sequences.min()
    maximum = sequences.max()

    print(
        f"Minimum value: "
        f"{minimum:.6f}"
    )

    print(
        f"Maximum value: "
        f"{maximum:.6f}"
    )

    # Validation
    if nan_count != 0:

        raise ValueError(
            "ERROR: NaN values found in "
            "TimeGAN sequences."
        )

    if inf_count != 0:

        raise ValueError(
            "ERROR: Infinite values found in "
            "TimeGAN sequences."
        )

    if minimum < 0 or maximum > 1:

        raise ValueError(
            "ERROR: Normalized values are "
            "outside the expected 0-1 range."
        )

    print()
    print(
        "✓ No NaN values"
    )

    print(
        "✓ No infinite values"
    )

    print(
        "✓ Values are normalized between 0 and 1"
    )

    print(
        "✓ Sequence shape is valid"
    )


# ============================================================
# 13. MAIN FUNCTION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("        MACROSTRESS-GAN")
    print("        TIMEGAN DATA PREPARATION")
    print("=" * 70)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    data = load_data()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    validate_features(
        data
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    data = preprocess_data(
        data
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    (
        normalized_data,
        scaler
    ) = normalize_data(
        data
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    sequences = create_sequences(
        normalized_data
    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    save_scaler(
        scaler
    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    save_sequences(
        sequences
    )


    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    save_training_data(
        normalized_data,
        data
    )


    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    display_sample(
        sequences
    )


    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    validate_output(
        sequences
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("       TIMEGAN DATA PREPARATION COMPLETED")
    print("=" * 70)

    print()
    print("INPUT DATA")
    print("-" * 70)

    print(
        f"Original rows: "
        f"{len(data)}"
    )

    print(
        f"Features used: "
        f"{len(FEATURES)}"
    )

    print()

    for feature in FEATURES:

        print(
            f"  ✓ {feature}"
        )


    print()
    print("TIMEGAN CONFIGURATION")
    print("-" * 70)

    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH} days"
    )

    print(
        f"Features        : "
        f"{len(FEATURES)}"
    )

    print(
        f"Sequences       : "
        f"{len(sequences)}"
    )

    print(
        f"Final shape     : "
        f"{sequences.shape}"
    )


    print()
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        f"✓ {SEQUENCE_PATH}"
    )

    print(
        f"✓ {SCALER_PATH}"
    )

    print(
        f"✓ {TRAINING_DATA_PATH}"
    )


    print()
    print("NEXT STEP")
    print("-" * 70)

    print(
        "TimeGAN model training"
    )

    print()
    print(
        "The prepared sequences are now ready "
        "for the TimeGAN Embedder, Recovery, "
        "Supervisor, Generator and Discriminator."
    )

    print()
    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()

