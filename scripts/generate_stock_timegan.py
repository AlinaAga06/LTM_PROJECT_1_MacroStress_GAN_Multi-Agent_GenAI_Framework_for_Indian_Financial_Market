# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V1
# SYNTHETIC DATA GENERATION
#
# Generates synthetic 30-day sequences using:
#
# models/timegan/stock/timegan_stock_v1.pt
#
# Then inverse-transforms the generated values using:
#
# data/processed/institutions/stock/
#     stock_timegan_scaler.pkl
#
# OUTPUT:
#
# outputs/synthetic/stock/
#     stock_synthetic_sequences_scaled.npy
#     stock_synthetic_sequences.npy
#     stock_synthetic_financial_data.csv
# ============================================================


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "stock"
    / "timegan_stock_v1.pt"
)


SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "stock_timegan_scaler.pkl"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
    / "stock"
)


SCALED_NPY_OUTPUT = (
    OUTPUT_DIR
    / "stock_synthetic_sequences_scaled.npy"
)


RAW_NPY_OUTPUT = (
    OUTPUT_DIR
    / "stock_synthetic_sequences.npy"
)


CSV_OUTPUT = (
    OUTPUT_DIR
    / "stock_synthetic_financial_data.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_SEQUENCES = 1000

SEQ_LEN = 30

FEATURE_DIM = 5

HIDDEN_DIM = 24

NUM_LAYERS = 2


FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ):

        super().__init__()


        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )


        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# SUPERVISOR
# ============================================================

class Supervisor(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ):

        super().__init__()


        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )


        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# RECOVERY
# ============================================================

class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        output_dim=FEATURE_DIM,
        num_layers=NUM_LAYERS
    ):

        super().__init__()


        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )


        self.linear = nn.Linear(
            hidden_dim,
            output_dim
        )


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print()
    print("=" * 70)
    print("CHECKING STOCK TIMEGAN FILES")
    print("=" * 70)


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Stock TimeGAN model not found:

{MODEL_PATH}
"""
        )


    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"""
Stock TimeGAN scaler not found:

{SCALER_PATH}
"""
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"Model  : {MODEL_PATH}"
    )


    print(
        f"Scaler : {SCALER_PATH}"
    )


    print(
        f"Output : {OUTPUT_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING STOCK TIMEGAN MODEL")
    print("=" * 70)


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise ValueError(
            "Unexpected checkpoint format."
        )


    print(
        f"Model version : "
        f"{checkpoint.get('model_version', 'Unknown')}"
    )


    print(
        f"Institution   : "
        f"{checkpoint.get('institution', 'Unknown')}"
    )


    print(
        f"Data type     : "
        f"{checkpoint.get('data_type', 'Unknown')}"
    )


    checkpoint_features = checkpoint.get(
        "features",
        FEATURES
    )


    if checkpoint_features != FEATURES:

        raise ValueError(
            "Model feature configuration does not "
            "match the expected Stock features."
        )


    checkpoint_seq_len = checkpoint.get(
        "sequence_length",
        SEQ_LEN
    )


    if checkpoint_seq_len != SEQ_LEN:

        raise ValueError(
            f"Model sequence length is "
            f"{checkpoint_seq_len}, expected {SEQ_LEN}."
        )


    generator = Generator().to(
        DEVICE
    )


    supervisor = Supervisor().to(
        DEVICE
    )


    recovery = Recovery().to(
        DEVICE
    )


    # ========================================================
    # GENERATOR
    # ========================================================

    if "generator_state_dict" in checkpoint:

        generator.load_state_dict(
            checkpoint[
                "generator_state_dict"
            ]
        )

    elif "generator" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator"]
        )

    else:

        raise KeyError(
            "Generator weights not found."
        )


    # ========================================================
    # SUPERVISOR
    # ========================================================

    if "supervisor_state_dict" in checkpoint:

        supervisor.load_state_dict(
            checkpoint[
                "supervisor_state_dict"
            ]
        )

    elif "supervisor" in checkpoint:

        supervisor.load_state_dict(
            checkpoint["supervisor"]
        )

    else:

        raise KeyError(
            "Supervisor weights not found."
        )


    # ========================================================
    # RECOVERY
    # ========================================================

    if "recovery_state_dict" in checkpoint:

        recovery.load_state_dict(
            checkpoint[
                "recovery_state_dict"
            ]
        )

    elif "recovery" in checkpoint:

        recovery.load_state_dict(
            checkpoint["recovery"]
        )

    else:

        raise KeyError(
            "Recovery weights not found."
        )


    generator.eval()

    supervisor.eval()

    recovery.eval()


    print()
    print("Generator loaded successfully.")

    print("Supervisor loaded successfully.")

    print("Recovery loaded successfully.")


    return (
        generator,
        supervisor,
        recovery
    )


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    print()
    print("=" * 70)
    print("LOADING STOCK TIMEGAN SCALER")
    print("=" * 70)


    scaler = joblib.load(
        SCALER_PATH
    )


    print(
        f"Scaler type: {type(scaler).__name__}"
    )


    return scaler


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

def generate_sequences(
    generator,
    supervisor,
    recovery
):

    print()
    print("=" * 70)
    print("GENERATING SYNTHETIC STOCK SEQUENCES")
    print("=" * 70)


    print(
        f"Sequences     : {NUM_SEQUENCES}"
    )


    print(
        f"Sequence length: {SEQ_LEN}"
    )


    print(
        f"Features       : {FEATURE_DIM}"
    )


    # ========================================================
    # RANDOM NOISE
    # ========================================================

    noise = torch.rand(
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )


    # ========================================================
    # GENERATION
    # ========================================================

    with torch.no_grad():

        generated_hidden = generator(
            noise
        )


        supervised_hidden = supervisor(
            generated_hidden
        )


        synthetic_scaled = recovery(
            supervised_hidden
        )


    # ========================================================
    # NUMPY
    # ========================================================

    synthetic_scaled = (
        synthetic_scaled
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    print()
    print(
        f"Generated shape: "
        f"{synthetic_scaled.shape}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if np.isnan(
        synthetic_scaled
    ).any():

        raise ValueError(
            "Generated data contains NaN."
        )


    if np.isinf(
        synthetic_scaled
    ).any():

        raise ValueError(
            "Generated data contains Inf."
        )


    print(
        f"Scaled minimum: "
        f"{synthetic_scaled.min():.6f}"
    )


    print(
        f"Scaled maximum: "
        f"{synthetic_scaled.max():.6f}"
    )


    # ========================================================
    # SAVE SCALED DATA
    # ========================================================

    np.save(
        SCALED_NPY_OUTPUT,
        synthetic_scaled
    )


    print()
    print(
        f"Scaled sequences saved:"
    )


    print(
        SCALED_NPY_OUTPUT
    )


    return synthetic_scaled


# ============================================================
# INVERSE TRANSFORM
# ============================================================

def inverse_transform(
    synthetic_scaled,
    scaler
):

    print()
    print("=" * 70)
    print("INVERSE TRANSFORMING STOCK DATA")
    print("=" * 70)


    original_shape = (
        synthetic_scaled.shape
    )


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flattened = synthetic_scaled.reshape(
        -1,
        FEATURE_DIM
    )


    # --------------------------------------------------------
    # Inverse QuantileTransformer
    # --------------------------------------------------------

    synthetic_raw = scaler.inverse_transform(
        flattened
    )


    synthetic_raw = synthetic_raw.reshape(
        original_shape
    )


    synthetic_raw = synthetic_raw.astype(
        np.float32
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if np.isnan(
        synthetic_raw
    ).any():

        raise ValueError(
            "Inverse-transformed data contains NaN."
        )


    if np.isinf(
        synthetic_raw
    ).any():

        raise ValueError(
            "Inverse-transformed data contains Inf."
        )


    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        RAW_NPY_OUTPUT,
        synthetic_raw
    )


    print(
        f"Raw sequence shape: "
        f"{synthetic_raw.shape}"
    )


    print(
        f"Raw minimum: "
        f"{synthetic_raw.min():.6f}"
    )


    print(
        f"Raw maximum: "
        f"{synthetic_raw.max():.6f}"
    )


    print()
    print(
        f"Inverse-transformed sequences saved:"
    )


    print(
        RAW_NPY_OUTPUT
    )


    return synthetic_raw


# ============================================================
# CREATE CSV
# ============================================================

def create_csv(
    synthetic_raw
):

    print()
    print("=" * 70)
    print("CREATING STOCK SYNTHETIC CSV")
    print("=" * 70)


    rows = []


    for sequence_id in range(
        NUM_SEQUENCES
    ):

        for timestep in range(
            SEQ_LEN
        ):

            row = {

                "sequence_id":
                    sequence_id,

                "timestep":
                    timestep
            }


            for feature_index, feature_name in enumerate(
                FEATURES
            ):

                row[
                    feature_name
                ] = synthetic_raw[
                    sequence_id,
                    timestep,
                    feature_index
                ]


            rows.append(
                row
            )


    synthetic_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if synthetic_df.empty:

        raise ValueError(
            "Generated CSV is empty."
        )


    nan_count = (
        synthetic_df.isna()
        .sum()
        .sum()
    )


    inf_count = np.isinf(
        synthetic_df[
            FEATURES
        ].values
    ).sum()


    if nan_count != 0:

        raise ValueError(
            f"CSV contains {nan_count} NaN values."
        )


    if inf_count != 0:

        raise ValueError(
            f"CSV contains {inf_count} Inf values."
        )


    # ========================================================
    # SAVE
    # ========================================================

    synthetic_df.to_csv(
        CSV_OUTPUT,
        index=False
    )


    print(
        f"CSV shape: "
        f"{synthetic_df.shape}"
    )


    print(
        f"CSV saved:"
    )


    print(
        CSV_OUTPUT
    )


    return synthetic_df


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    synthetic_df
):

    print()
    print("=" * 70)
    print("STOCK SYNTHETIC DATA SUMMARY")
    print("=" * 70)


    print()


    for feature in FEATURES:

        series = synthetic_df[
            feature
        ]


        print(
            f"{feature}"
        )


        print(
            f"  Mean : {series.mean():.8f}"
        )


        print(
            f"  Std  : {series.std():.8f}"
        )


        print(
            f"  Min  : {series.min():.8f}"
        )


        print(
            f"  Max  : {series.max():.8f}"
        )


        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V1")
    print("SYNTHETIC DATA GENERATION")
    print("=" * 70)


    print()
    print(
        f"Device         : {DEVICE}"
    )


    print(
        f"Sequences      : {NUM_SEQUENCES}"
    )


    print(
        f"Sequence length: {SEQ_LEN}"
    )


    print(
        f"Feature count  : {FEATURE_DIM}"
    )


    print()
    print("Features:")


    for feature in FEATURES:

        print(
            f"  - {feature}"
        )


    # ========================================================
    # FILE CHECK
    # ========================================================

    check_files()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    (
        generator,
        supervisor,
        recovery
    ) = load_model()


    # ========================================================
    # LOAD SCALER
    # ========================================================

    scaler = load_scaler()


    # ========================================================
    # GENERATE
    # ========================================================

    synthetic_scaled = generate_sequences(
        generator,
        supervisor,
        recovery
    )


    # ========================================================
    # INVERSE TRANSFORM
    # ========================================================

    synthetic_raw = inverse_transform(
        synthetic_scaled,
        scaler
    )


    # ========================================================
    # CSV
    # ========================================================

    synthetic_df = create_csv(
        synthetic_raw
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        synthetic_df
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("STOCK TIMEGAN V1 GENERATION SUCCESSFUL")
    print("=" * 70)


    print()
    print("Generated files:")
    print()


    print(
        f"1. Scaled NumPy:"
    )


    print(
        f"   {SCALED_NPY_OUTPUT}"
    )


    print()


    print(
        f"2. Inverse-transformed NumPy:"
    )


    print(
        f"   {RAW_NPY_OUTPUT}"
    )


    print()


    print(
        f"3. Financial CSV:"
    )


    print(
        f"   {CSV_OUTPUT}"
    )


    print()
    print(
        "The CSV contains synthetic STOCK-MARKET "
        "returns/changes in the original feature space."
    )


    print()
    print(
        "Next step: validate the synthetic Stock data "
        "against the real Stock dataset."
    )


    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()