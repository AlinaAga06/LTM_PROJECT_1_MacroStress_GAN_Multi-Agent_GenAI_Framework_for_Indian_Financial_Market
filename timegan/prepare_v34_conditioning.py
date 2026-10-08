"""
MacroStress-GAN V3.4 Conditioning Dataset Preparation
======================================================

Purpose:
    Create regime + continuous stress conditioning vectors for V3.4.

Important:
    TimeGAN has 3,989 sequences.
    Regime context has 3,960 sequences.

    They are aligned using exact Sequence_End_Date values.
    We DO NOT assume that sequence index 0 in one dataset corresponds
    to sequence index 0 in the other dataset.

Outputs:
    data/processed/timegan_v34/
        timegan_v34_sequences.npy
        regime_ids.npy
        regime_onehot.npy
        conditioning_raw.npy
        conditioning_scaled.npy
        conditioning_vector.npy
        conditioning_scaler.pkl
        conditioning_metadata.csv
        conditioning_feature_description.csv
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIMEGAN_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_sequences.npy"
)

TIMEGAN_TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_training_data.csv"
)

CONTEXT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "regime_analysis"
    / "sequence_context"
    / "sequence_context_alignment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v34"
)

SEQUENCE_LENGTH = 30
N_FEATURES = 5

EXPECTED_TIMEGAN_SEQUENCES = 3989
EXPECTED_CONTEXT_SEQUENCES = 3960

REGIME_MAP = {
    "Normal": 0,
    "Elevated Volatility": 1,
    "Market Stress": 2,
    "Crisis": 3,
}

CONDITIONING_COLUMNS = [
    "Mean_Stress_Score",
    "Mean_Volatility_Score",
    "Mean_Movement_Score",
    "Mean_NIFTY_Drawdown",
    "Mean_VIX_Stress",
]

TIMEGAN_FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# HELPER
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("MacroStress-GAN V3.4 Conditioning Dataset Preparation")
    print("=" * 80)

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD TIMEGAN SEQUENCES
    # --------------------------------------------------------

    print_section("1. Loading TimeGAN sequences")

    if not TIMEGAN_SEQUENCE_PATH.exists():
        raise FileNotFoundError(
            f"TimeGAN sequence file not found:\n{TIMEGAN_SEQUENCE_PATH}"
        )

    sequences = np.load(TIMEGAN_SEQUENCE_PATH)

    print(f"Sequence shape: {sequences.shape}")
    print(f"Sequence dtype: {sequences.dtype}")

    if sequences.ndim != 3:
        raise ValueError(
            f"Expected 3D TimeGAN sequences, got shape {sequences.shape}"
        )

    n_sequences, sequence_length, n_features = sequences.shape

    if sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, "
            f"got {sequence_length}"
        )

    if n_features != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, got {n_features}"
        )

    if n_sequences != EXPECTED_TIMEGAN_SEQUENCES:
        print(
            f"WARNING: Expected {EXPECTED_TIMEGAN_SEQUENCES} TimeGAN "
            f"sequences, got {n_sequences}"
        )

    if not np.isfinite(sequences).all():
        raise ValueError("TimeGAN sequences contain NaN or Inf values.")

    # --------------------------------------------------------
    # LOAD TIMEGAN TRAINING DATA
    # --------------------------------------------------------

    print_section("2. Loading TimeGAN training data")

    if not TIMEGAN_TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"TimeGAN training data not found:\n"
            f"{TIMEGAN_TRAINING_DATA_PATH}"
        )

    training_df = pd.read_csv(TIMEGAN_TRAINING_DATA_PATH)

    if "Date" not in training_df.columns:
        raise ValueError(
            "TimeGAN training data does not contain a 'Date' column."
        )

    training_df["Date"] = pd.to_datetime(
        training_df["Date"],
        errors="raise"
    )

    training_df = training_df.sort_values("Date").reset_index(drop=True)

    print(f"Training data shape: {training_df.shape}")
    print(
        f"Training date range: "
        f"{training_df['Date'].min().date()} -> "
        f"{training_df['Date'].max().date()}"
    )

    expected_training_rows = (
        n_sequences + SEQUENCE_LENGTH - 1
    )

    if len(training_df) != expected_training_rows:
        raise ValueError(
            "TimeGAN sequence/date alignment check failed.\n"
            f"Expected training rows: {expected_training_rows}\n"
            f"Actual training rows: {len(training_df)}"
        )

    # --------------------------------------------------------
    # CONSTRUCT EXACT TIMEGAN SEQUENCE DATE MAP
    # --------------------------------------------------------

    print_section("3. Constructing TimeGAN sequence date map")

    #
    # Sequence i contains:
    #
    #     training_df.iloc[i : i + 30]
    #
    # Therefore its end date is:
    #
    #     training_df.iloc[i + 29]["Date"]
    #

    timegan_date_map = pd.DataFrame(
        {
            "TimeGAN_Sequence_Index": np.arange(n_sequences),
            "Sequence_Start_Date": training_df["Date"]
            .iloc[:n_sequences]
            .reset_index(drop=True),
            "Sequence_End_Date": training_df["Date"]
            .iloc[SEQUENCE_LENGTH - 1:]
            .reset_index(drop=True),
        }
    )

    print(
        f"TimeGAN sequence date range: "
        f"{timegan_date_map['Sequence_End_Date'].min().date()} -> "
        f"{timegan_date_map['Sequence_End_Date'].max().date()}"
    )

    # --------------------------------------------------------
    # LOAD REGIME CONTEXT
    # --------------------------------------------------------

    print_section("4. Loading regime sequence context")

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Regime context file not found:\n{CONTEXT_PATH}"
        )

    context_df = pd.read_csv(CONTEXT_PATH)

    required_context_columns = [
        "Sequence_End_Date",
        "Sequence_Start_Date",
        "Initial_Regime",
        "Final_Regime",
        "Dominant_Regime",
        "Dominant_Count",
        "Dominant_Fraction",
        "Regime_Changes",
        "Mean_Stress_Score",
        "Median_Stress_Score",
        "Final_Stress_Score",
        "Maximum_Stress_Score",
        "Mean_Volatility_Score",
        "Mean_Movement_Score",
        "Mean_NIFTY_Drawdown",
        "Mean_VIX_Stress",
        "Normal_Count",
        "Elevated_Count",
        "Market_Stress_Count",
        "Crisis_Count",
    ]

    missing_context_columns = [
        c for c in required_context_columns
        if c not in context_df.columns
    ]

    if missing_context_columns:
        raise ValueError(
            "Missing columns in regime context:\n"
            + "\n".join(missing_context_columns)
        )

    context_df["Sequence_End_Date"] = pd.to_datetime(
        context_df["Sequence_End_Date"],
        errors="raise"
    )

    context_df["Sequence_Start_Date"] = pd.to_datetime(
        context_df["Sequence_Start_Date"],
        errors="raise"
    )

    context_df = (
        context_df
        .sort_values("Sequence_End_Date")
        .reset_index(drop=True)
    )

    print(f"Context shape: {context_df.shape}")
    print(
        f"Context date range: "
        f"{context_df['Sequence_End_Date'].min().date()} -> "
        f"{context_df['Sequence_End_Date'].max().date()}"
    )

    if len(context_df) != EXPECTED_CONTEXT_SEQUENCES:
        print(
            f"WARNING: Expected {EXPECTED_CONTEXT_SEQUENCES} context rows, "
            f"got {len(context_df)}"
        )

    # --------------------------------------------------------
    # CHECK DUPLICATE DATES
    # --------------------------------------------------------

    print_section("5. Checking sequence-date uniqueness")

    duplicate_timegan = timegan_date_map[
        timegan_date_map["Sequence_End_Date"].duplicated(keep=False)
    ]

    duplicate_context = context_df[
        context_df["Sequence_End_Date"].duplicated(keep=False)
    ]

    if len(duplicate_timegan) > 0:
        raise ValueError(
            "Duplicate TimeGAN sequence end dates detected."
        )

    if len(duplicate_context) > 0:
        raise ValueError(
            "Duplicate regime-context sequence end dates detected."
        )

    print("TimeGAN end dates: UNIQUE")
    print("Context end dates: UNIQUE")

    # --------------------------------------------------------
    # EXACT DATE ALIGNMENT
    # --------------------------------------------------------

    print_section("6. Performing exact date-based alignment")

    aligned = context_df.merge(
        timegan_date_map,
        on="Sequence_End_Date",
        how="left",
        suffixes=("_Context", "_TimeGAN"),
        validate="one_to_one",
    )

    missing_timegan_matches = aligned[
        aligned["TimeGAN_Sequence_Index"].isna()
    ]

    if len(missing_timegan_matches) > 0:

        print(
            "ERROR: Some regime-context sequences do not have "
            "matching TimeGAN sequence dates."
        )

        print(
            missing_timegan_matches[
                [
                    "Sequence_End_Date",
                    "Sequence_Start_Date_Context",
                ]
            ].head(20).to_string(index=False)
        )

        raise ValueError(
            f"{len(missing_timegan_matches)} context sequences "
            f"could not be aligned to TimeGAN sequences."
        )

    # --------------------------------------------------------
    # VERIFY START DATES TOO
    # --------------------------------------------------------

    start_date_mismatch = aligned[
        aligned["Sequence_Start_Date_Context"]
        != aligned["Sequence_Start_Date_TimeGAN"]
    ]

    if len(start_date_mismatch) > 0:

        print(
            "ERROR: Sequence start-date mismatch detected."
        )

        print(
            start_date_mismatch[
                [
                    "Sequence_End_Date",
                    "Sequence_Start_Date_Context",
                    "Sequence_Start_Date_TimeGAN",
                ]
            ].head(20).to_string(index=False)
        )

        raise ValueError(
            f"{len(start_date_mismatch)} sequences have "
            "matching end dates but different start dates."
        )

    print("End-date alignment: PASSED")
    print("Start-date alignment: PASSED")

    # --------------------------------------------------------
    # SORT BY ACTUAL TIMEGAN SEQUENCE INDEX
    # --------------------------------------------------------

    aligned = aligned.sort_values(
        "TimeGAN_Sequence_Index"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # EXTRACT ALIGNED SEQUENCES
    # --------------------------------------------------------

    print_section("7. Extracting aligned TimeGAN sequences")

    aligned_indices = (
        aligned["TimeGAN_Sequence_Index"]
        .astype(int)
        .to_numpy()
    )

    aligned_sequences = sequences[aligned_indices]

    print(f"Original TimeGAN sequences: {len(sequences)}")
    print(f"Aligned V3.4 sequences:     {len(aligned_sequences)}")
    print(
        f"Excluded TimeGAN sequences: "
        f"{len(sequences) - len(aligned_sequences)}"
    )

    # --------------------------------------------------------
    # VERIFY INDEX UNIQUENESS
    # --------------------------------------------------------

    if len(np.unique(aligned_indices)) != len(aligned_indices):
        raise ValueError(
            "Duplicate TimeGAN sequence indices detected after alignment."
        )

    # --------------------------------------------------------
    # REGIME IDs
    # --------------------------------------------------------

    print_section("8. Creating regime conditioning")

    unknown_regimes = sorted(
        set(aligned["Dominant_Regime"].astype(str))
        - set(REGIME_MAP.keys())
    )

    if unknown_regimes:
        raise ValueError(
            f"Unknown dominant regimes found: {unknown_regimes}"
        )

    regime_ids = (
        aligned["Dominant_Regime"]
        .map(REGIME_MAP)
        .astype(np.int64)
        .to_numpy()
    )

    regime_onehot = np.zeros(
        (len(regime_ids), 4),
        dtype=np.float32
    )

    regime_onehot[
        np.arange(len(regime_ids)),
        regime_ids
    ] = 1.0

    # --------------------------------------------------------
    # CONTINUOUS CONDITIONING
    # --------------------------------------------------------

    print_section("9. Creating continuous stress conditioning")

    conditioning_raw = (
        aligned[CONDITIONING_COLUMNS]
        .astype(np.float64)
        .to_numpy()
    )

    if not np.isfinite(conditioning_raw).all():
        raise ValueError(
            "Continuous conditioning contains NaN or Inf values."
        )

    conditioning_scaler = StandardScaler()

    conditioning_scaled = (
        conditioning_scaler
        .fit_transform(conditioning_raw)
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # COMBINED CONDITIONING VECTOR
    # --------------------------------------------------------

    conditioning_vector = np.concatenate(
        [
            regime_onehot,
            conditioning_scaled,
        ],
        axis=1,
    ).astype(np.float32)

    print(f"Regime one-hot shape:       {regime_onehot.shape}")
    print(f"Continuous context shape:   {conditioning_scaled.shape}")
    print(f"Final conditioning shape:   {conditioning_vector.shape}")

    # --------------------------------------------------------
    # CONDITIONING STATISTICS
    # --------------------------------------------------------

    print_section("10. Conditioning statistics")

    print("\nDominant regime distribution:")

    regime_counts = (
        aligned["Dominant_Regime"]
        .value_counts()
        .reindex(REGIME_MAP.keys(), fill_value=0)
    )

    for regime, count in regime_counts.items():

        percentage = (
            count / len(aligned) * 100
        )

        print(
            f"  {regime:20s}: "
            f"{count:4d} "
            f"({percentage:7.3f}%)"
        )

    print("\nContinuous conditioning means after scaling:")

    for i, column in enumerate(CONDITIONING_COLUMNS):

        print(
            f"  {column:25s}: "
            f"mean={conditioning_scaled[:, i].mean(): .6f}, "
            f"std={conditioning_scaled[:, i].std(): .6f}"
        )

    # --------------------------------------------------------
    # ALIGNMENT SUMMARY
    # --------------------------------------------------------

    print_section("11. Alignment summary")

    first_row = aligned.iloc[0]
    last_row = aligned.iloc[-1]

    print(
        f"First aligned sequence:"
        f" index={int(first_row['TimeGAN_Sequence_Index'])},"
        f" start={first_row['Sequence_Start_Date_Context'].date()},"
        f" end={first_row['Sequence_End_Date'].date()}"
    )

    print(
        f"Last aligned sequence:"
        f" index={int(last_row['TimeGAN_Sequence_Index'])},"
        f" start={last_row['Sequence_Start_Date_Context'].date()},"
        f" end={last_row['Sequence_End_Date'].date()}"
    )

    print(
        f"Total TimeGAN sequences: {len(sequences)}"
    )

    print(
        f"Total context sequences: {len(context_df)}"
    )

    print(
        f"Successfully aligned:    {len(aligned)}"
    )

    print(
        f"Excluded early sequences: "
        f"{len(sequences) - len(aligned)}"
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print_section("12. Final validation")

    if len(aligned_sequences) != len(context_df):
        raise ValueError(
            "Final sequence/context count mismatch."
        )

    if aligned_sequences.shape != (
        len(context_df),
        SEQUENCE_LENGTH,
        N_FEATURES,
    ):
        raise ValueError(
            "Unexpected aligned sequence shape: "
            f"{aligned_sequences.shape}"
        )

    if not np.isfinite(aligned_sequences).all():
        raise ValueError(
            "Aligned TimeGAN sequences contain NaN or Inf."
        )

    if not np.isfinite(conditioning_vector).all():
        raise ValueError(
            "Conditioning vector contains NaN or Inf."
        )

    if not np.allclose(
        regime_onehot.sum(axis=1),
        1.0,
    ):
        raise ValueError(
            "Invalid regime one-hot encoding."
        )

    print("Sequence shape:              PASS")
    print("Sequence finite check:       PASS")
    print("Context finite check:        PASS")
    print("Regime encoding check:       PASS")
    print("Date alignment check:        PASS")
    print("Start-date alignment:        PASS")
    print("Conditioning vector check:   PASS")

    # --------------------------------------------------------
    # SAVE ALIGNED DATA
    # --------------------------------------------------------

    print_section("13. Saving V3.4 conditioning dataset")

    np.save(
        OUTPUT_DIR / "timegan_v34_sequences.npy",
        aligned_sequences.astype(np.float32),
    )

    np.save(
        OUTPUT_DIR / "regime_ids.npy",
        regime_ids,
    )

    np.save(
        OUTPUT_DIR / "regime_onehot.npy",
        regime_onehot,
    )

    np.save(
        OUTPUT_DIR / "conditioning_raw.npy",
        conditioning_raw.astype(np.float32),
    )

    np.save(
        OUTPUT_DIR / "conditioning_scaled.npy",
        conditioning_scaled,
    )

    np.save(
        OUTPUT_DIR / "conditioning_vector.npy",
        conditioning_vector,
    )

    joblib.dump(
        conditioning_scaler,
        OUTPUT_DIR / "conditioning_scaler.pkl",
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata_columns = [
        "V34_Sequence_Index",
        "TimeGAN_Sequence_Index",
        "Sequence_Start_Date",
        "Sequence_End_Date",
        "Initial_Regime",
        "Final_Regime",
        "Dominant_Regime",
        "Dominant_Count",
        "Dominant_Fraction",
        "Regime_Changes",
        "Mean_Stress_Score",
        "Median_Stress_Score",
        "Final_Stress_Score",
        "Maximum_Stress_Score",
        "Mean_Volatility_Score",
        "Mean_Movement_Score",
        "Mean_NIFTY_Drawdown",
        "Mean_VIX_Stress",
        "Normal_Count",
        "Elevated_Count",
        "Market_Stress_Count",
        "Crisis_Count",
    ]

    metadata = pd.DataFrame(
        {
            "V34_Sequence_Index": np.arange(len(aligned)),
            "TimeGAN_Sequence_Index": aligned[
                "TimeGAN_Sequence_Index"
            ].astype(int),
            "Sequence_Start_Date": aligned[
                "Sequence_Start_Date_Context"
            ].dt.strftime("%Y-%m-%d"),
            "Sequence_End_Date": aligned[
                "Sequence_End_Date"
            ].dt.strftime("%Y-%m-%d"),
            "Initial_Regime": aligned["Initial_Regime"],
            "Final_Regime": aligned["Final_Regime"],
            "Dominant_Regime": aligned["Dominant_Regime"],
            "Dominant_Count": aligned["Dominant_Count"],
            "Dominant_Fraction": aligned["Dominant_Fraction"],
            "Regime_Changes": aligned["Regime_Changes"],
            "Mean_Stress_Score": aligned["Mean_Stress_Score"],
            "Median_Stress_Score": aligned["Median_Stress_Score"],
            "Final_Stress_Score": aligned["Final_Stress_Score"],
            "Maximum_Stress_Score": aligned["Maximum_Stress_Score"],
            "Mean_Volatility_Score": aligned["Mean_Volatility_Score"],
            "Mean_Movement_Score": aligned["Mean_Movement_Score"],
            "Mean_NIFTY_Drawdown": aligned["Mean_NIFTY_Drawdown"],
            "Mean_VIX_Stress": aligned["Mean_VIX_Stress"],
            "Normal_Count": aligned["Normal_Count"],
            "Elevated_Count": aligned["Elevated_Count"],
            "Market_Stress_Count": aligned["Market_Stress_Count"],
            "Crisis_Count": aligned["Crisis_Count"],
        }
    )

    metadata = metadata[metadata_columns]

    metadata.to_csv(
        OUTPUT_DIR / "conditioning_metadata.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FEATURE DESCRIPTION
    # --------------------------------------------------------

    feature_description = pd.DataFrame(
        [
            {
                "Feature": "Regime_Normal",
                "Type": "Categorical",
                "Encoding": "One-hot",
                "Description": "Dominant regime is Normal",
            },
            {
                "Feature": "Regime_Elevated_Volatility",
                "Type": "Categorical",
                "Encoding": "One-hot",
                "Description": "Dominant regime is Elevated Volatility",
            },
            {
                "Feature": "Regime_Market_Stress",
                "Type": "Categorical",
                "Encoding": "One-hot",
                "Description": "Dominant regime is Market Stress",
            },
            {
                "Feature": "Regime_Crisis",
                "Type": "Categorical",
                "Encoding": "One-hot",
                "Description": "Dominant regime is Crisis",
            },
            {
                "Feature": "Mean_Stress_Score",
                "Type": "Continuous",
                "Encoding": "StandardScaler",
                "Description": "Mean composite stress intensity over the 30-day window",
            },
            {
                "Feature": "Mean_Volatility_Score",
                "Type": "Continuous",
                "Encoding": "StandardScaler",
                "Description": "Mean rolling volatility context",
            },
            {
                "Feature": "Mean_Movement_Score",
                "Type": "Continuous",
                "Encoding": "StandardScaler",
                "Description": "Mean absolute market movement context",
            },
            {
                "Feature": "Mean_NIFTY_Drawdown",
                "Type": "Continuous",
                "Encoding": "StandardScaler",
                "Description": "Mean 30-day NIFTY drawdown context",
            },
            {
                "Feature": "Mean_VIX_Stress",
                "Type": "Continuous",
                "Encoding": "StandardScaler",
                "Description": "Mean VIX stress context",
            },
        ]
    )

    feature_description.to_csv(
        OUTPUT_DIR / "conditioning_feature_description.csv",
        index=False,
    )

    # --------------------------------------------------------
    # RELOAD CHECK
    # --------------------------------------------------------

    print_section("14. Reload verification")

    saved_sequences = np.load(
        OUTPUT_DIR / "timegan_v34_sequences.npy"
    )

    saved_regime_ids = np.load(
        OUTPUT_DIR / "regime_ids.npy"
    )

    saved_onehot = np.load(
        OUTPUT_DIR / "regime_onehot.npy"
    )

    saved_raw = np.load(
        OUTPUT_DIR / "conditioning_raw.npy"
    )

    saved_scaled = np.load(
        OUTPUT_DIR / "conditioning_scaled.npy"
    )

    saved_vector = np.load(
        OUTPUT_DIR / "conditioning_vector.npy"
    )

    if not np.array_equal(
        saved_sequences,
        aligned_sequences.astype(np.float32),
    ):
        raise ValueError(
            "Reloaded sequence data does not match saved data."
        )

    if not np.array_equal(
        saved_regime_ids,
        regime_ids,
    ):
        raise ValueError(
            "Reloaded regime IDs do not match."
        )

    if not np.array_equal(
        saved_onehot,
        regime_onehot,
    ):
        raise ValueError(
            "Reloaded regime one-hot data does not match."
        )

    if not np.array_equal(
        saved_vector,
        conditioning_vector,
    ):
        raise ValueError(
            "Reloaded conditioning vector does not match."
        )

    print("Reload verification:        PASS")

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print_section("V3.4 CONDITIONING DATASET READY")

    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Saved files:")

    output_files = [
        "timegan_v34_sequences.npy",
        "regime_ids.npy",
        "regime_onehot.npy",
        "conditioning_raw.npy",
        "conditioning_scaled.npy",
        "conditioning_vector.npy",
        "conditioning_scaler.pkl",
        "conditioning_metadata.csv",
        "conditioning_feature_description.csv",
    ]

    for filename in output_files:
        path = OUTPUT_DIR / filename
        print(
            f"  {filename:40s} "
            f"{path.stat().st_size:,} bytes"
        )

    print()
    print("Final shapes:")
    print(
        f"  TimeGAN sequences:       {saved_sequences.shape}"
    )
    print(
        f"  Regime IDs:              {saved_regime_ids.shape}"
    )
    print(
        f"  Regime one-hot:          {saved_onehot.shape}"
    )
    print(
        f"  Continuous context:      {saved_scaled.shape}"
    )
    print(
        f"  Conditioning vector:     {saved_vector.shape}"
    )

    print()
    print("Alignment:")
    print(
        f"  Original TimeGAN:        {len(sequences)}"
    )
    print(
        f"  Regime context:          {len(context_df)}"
    )
    print(
        f"  Aligned sequences:       {len(aligned)}"
    )
    print(
        f"  Excluded early windows:  "
        f"{len(sequences) - len(aligned)}"
    )

    print()
    print("V3.2 model/data were NOT modified.")

    print()
    print("=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)