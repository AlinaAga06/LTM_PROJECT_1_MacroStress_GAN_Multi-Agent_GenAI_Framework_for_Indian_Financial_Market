"""
============================================================
MacroStress-GAN - TimeGAN Synthetic Data Generation
============================================================

Purpose:
    Generate synthetic 30-day financial market sequences
    using the trained TimeGAN model.

Input:
    models/timegan/timegan_model.pt
    data/processed/timegan_scaler.pkl

Output:
    outputs/synthetic/synthetic_sequences.npy
    outputs/synthetic/synthetic_financial_data.csv

Features:
    NIFTY50
    INDIA_VIX
    CRUDE_OIL
    USD_INR
    INDIA_10Y_YIELD
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from model import TimeGAN


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "timegan_model.pt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_scaler.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
)

SEQUENCES_PATH = (
    OUTPUT_DIR
    / "synthetic_sequences.npy"
)

CSV_PATH = (
    OUTPUT_DIR
    / "synthetic_financial_data.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 5
HIDDEN_DIM = 24
NOISE_DIM = 5
NUM_LAYERS = 2

SEQUENCE_LENGTH = 30

# Number of synthetic sequences
NUM_SYNTHETIC_SEQUENCES = 1000

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "NIFTY50",
    "INDIA_VIX",
    "CRUDE_OIL",
    "USD_INR",
    "INDIA_10Y_YIELD"
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING TRAINED TIMEGAN")
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nTrained model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = TimeGAN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        noise_dim=NOISE_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\n✓ Trained model loaded")

    print(
        f"Device: {DEVICE}"
    )

    return model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    print("\n")
    print("=" * 70)
    print("LOADING DATA SCALER")
    print("=" * 70)

    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"\nScaler not found:\n{SCALER_PATH}"
        )

    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        scaler = pickle.load(f)

    print("\n✓ Scaler loaded")

    return scaler


# ============================================================
# GENERATE NOISE
# ============================================================

def generate_noise(
    num_sequences
):

    noise = torch.rand(
        num_sequences,
        SEQUENCE_LENGTH,
        NOISE_DIM,
        device=DEVICE
    )

    return noise


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

def generate_synthetic_data(
    model,
    num_sequences
):

    print("\n")
    print("=" * 70)
    print("GENERATING SYNTHETIC FINANCIAL DATA")
    print("=" * 70)

    print(
        f"\nNumber of sequences: "
        f"{num_sequences}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH} days"
    )

    print(
        f"Number of variables: "
        f"{INPUT_DIM}"
    )

    all_sequences = []

    # --------------------------------------------------------
    # Generate in batches
    # --------------------------------------------------------

    batch_size = 64

    with torch.no_grad():

        for start in range(
            0,
            num_sequences,
            batch_size
        ):

            end = min(
                start + batch_size,
                num_sequences
            )

            current_batch = end - start

            noise = generate_noise(
                current_batch
            )

            # ------------------------------------------------
            # Generator
            # ------------------------------------------------

            generated_latent = (
                model.generator(noise)
            )

            # ------------------------------------------------
            # Supervisor
            # ------------------------------------------------

            supervised_latent = (
                model.supervisor(
                    generated_latent
                )
            )

            # ------------------------------------------------
            # Recovery
            # ------------------------------------------------

            generated_data = (
                model.recovery(
                    supervised_latent
                )
            )

            generated_data = (
                generated_data
                .cpu()
                .numpy()
            )

            all_sequences.append(
                generated_data
            )

            print(
                f"Generated "
                f"{end}/{num_sequences} sequences"
            )

    synthetic_data = np.concatenate(
        all_sequences,
        axis=0
    )

    print("\n✓ Synthetic generation completed")

    print(
        f"Generated shape: "
        f"{synthetic_data.shape}"
    )

    return synthetic_data


# ============================================================
# INVERSE TRANSFORM
# ============================================================

def inverse_transform(
    synthetic_data,
    scaler
):

    print("\n")
    print("=" * 70)
    print("CONVERTING TO ORIGINAL FINANCIAL SCALE")
    print("=" * 70)

    num_sequences = synthetic_data.shape[0]

    sequence_length = synthetic_data.shape[1]

    num_features = synthetic_data.shape[2]

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flattened = synthetic_data.reshape(
        -1,
        num_features
    )

    # --------------------------------------------------------
    # Inverse MinMaxScaler
    # --------------------------------------------------------

    original_scale = scaler.inverse_transform(
        flattened
    )

    # --------------------------------------------------------
    # Restore sequence structure
    # --------------------------------------------------------

    original_scale = original_scale.reshape(
        num_sequences,
        sequence_length,
        num_features
    )

    print("\n✓ Inverse transformation completed")

    return original_scale


# ============================================================
# SAVE NUMPY SEQUENCES
# ============================================================

def save_sequences(
    synthetic_data
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        SEQUENCES_PATH,
        synthetic_data.astype(np.float32)
    )

    print("\nSynthetic sequences saved:")
    print(SEQUENCES_PATH)


# ============================================================
# CONVERT TO CSV
# ============================================================

def create_dataframe(
    synthetic_data
):

    rows = []

    for sequence_id in range(
        synthetic_data.shape[0]
    ):

        for day in range(
            synthetic_data.shape[1]
        ):

            row = {

                "Sequence_ID":
                    sequence_id + 1,

                "Day":
                    day + 1
            }

            for feature_index, feature in enumerate(
                FEATURES
            ):

                row[feature] = synthetic_data[
                    sequence_id,
                    day,
                    feature_index
                ]

            rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    synthetic_data
):

    df = create_dataframe(
        synthetic_data
    )

    df.to_csv(
        CSV_PATH,
        index=False
    )

    print("\nSynthetic CSV saved:")
    print(CSV_PATH)

    print(
        f"\nCSV shape: {df.shape}"
    )

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_generated_data(
    synthetic_data
):

    print("\n")
    print("=" * 70)
    print("BASIC SYNTHETIC DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    nan_count = np.isnan(
        synthetic_data
    ).sum()

    # --------------------------------------------------------
    # Infinity
    # --------------------------------------------------------

    inf_count = np.isinf(
        synthetic_data
    ).sum()

    print(
        f"\nNaN values: {nan_count}"
    )

    print(
        f"Infinite values: {inf_count}"
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    flattened = synthetic_data.reshape(
        -1,
        INPUT_DIM
    )

    statistics = pd.DataFrame({

        "Feature":
            FEATURES,

        "Mean":
            flattened.mean(axis=0),

        "Std":
            flattened.std(axis=0),

        "Min":
            flattened.min(axis=0),

        "Max":
            flattened.max(axis=0)
    })

    print("\nSynthetic Data Statistics:")
    print(
        statistics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Validation status
    # --------------------------------------------------------

    if nan_count == 0 and inf_count == 0:

        print(
            "\n✓ No NaN or infinite values detected"
        )

    else:

        print(
            "\n⚠ WARNING: Invalid values detected"
        )

    return statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MACROSTRESS-GAN")
    print("TIMEGAN SYNTHETIC DATA GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = load_scaler()

    # --------------------------------------------------------
    # Generate normalized synthetic data
    # --------------------------------------------------------

    synthetic_normalized = (
        generate_synthetic_data(
            model,
            NUM_SYNTHETIC_SEQUENCES
        )
    )

    # --------------------------------------------------------
    # Convert to original scale
    # --------------------------------------------------------

    synthetic_original = inverse_transform(
        synthetic_normalized,
        scaler
    )

    # --------------------------------------------------------
    # Save NumPy sequences
    # --------------------------------------------------------

    save_sequences(
        synthetic_original
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = save_csv(
        synthetic_original
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_generated_data(
        synthetic_original
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION COMPLETED")
    print("=" * 70)

    print("\n✓ TimeGAN model loaded")
    print("✓ Synthetic sequences generated")
    print("✓ Original financial scale restored")
    print("✓ Synthetic sequences saved")
    print("✓ CSV file created")
    print("✓ Basic validation completed")

    print("\nOutput files:")

    print(
        f"\n1. {SEQUENCES_PATH}"
    )

    print(
        f"2. {CSV_PATH}"
    )

    print("\nNext step:")
    print(
        "Compare REAL vs SYNTHETIC data "
        "for TimeGAN validation."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()