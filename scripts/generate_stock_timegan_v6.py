"""
MacroStress-GAN
Stock Market TimeGAN V6 Generator

V6:
    StandardScaler
    Linear Recovery
    5 financial features
    30-day sequences

Loads:
    models/timegan/stock/timegan_stock_v6.pt
    data/processed/institutions/stock/v6/stock_v6_standard_scaler.pkl

Outputs:
    outputs/synthetic/stock/
        stock_v6_synthetic_sequences_scaled.npy
        stock_v6_synthetic_sequences.npy
        stock_v6_synthetic_financial_data.csv
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock",
    "timegan_stock_v6.pt"
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "v6",
    "stock_v6_standard_scaler.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "stock"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

SCALED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v6_synthetic_sequences_scaled.npy"
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v6_synthetic_sequences.npy"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v6_synthetic_financial_data.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2

NUM_SEQUENCES = 1000
BATCH_SIZE = 64


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Tanh()


    def forward(self, z):

        h, _ = self.gru(z)

        h = self.fc(h)

        h = self.activation(h)

        return h


# ============================================================
# SUPERVISOR
# ============================================================

class Supervisor(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Tanh()


    def forward(self, h):

        h_supervised, _ = self.gru(h)

        h_supervised = self.fc(
            h_supervised
        )

        h_supervised = self.activation(
            h_supervised
        )

        return h_supervised


# ============================================================
# RECOVERY
#
# V6 uses StandardScaler.
#
# Therefore there is NO sigmoid.
# Standardized values can be negative
# and can be greater than 1.
# ============================================================

class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim,
        output_dim,
        num_layers
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        return x


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V6 GENERATION")
print("=" * 78)

print()
print("Device:", DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 78)
print("1. LOADING V6 CHECKPOINT")
print("=" * 78)

print()
print("Checkpoint:")
print(MODEL_FILE)


if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"\nV6 checkpoint not found:\n"
        f"{MODEL_FILE}"
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


print()
print(
    "Model version:",
    checkpoint.get(
        "model_version",
        "Unknown"
    )
)

print(
    "Institution:",
    checkpoint.get(
        "institution",
        "Unknown"
    )
)

print(
    "Scaler type:",
    checkpoint.get(
        "scaler_type",
        "Unknown"
    )
)


# ============================================================
# VERIFY CHECKPOINT CONFIG
# ============================================================

checkpoint_seq_len = checkpoint.get(
    "seq_len"
)

checkpoint_feature_dim = checkpoint.get(
    "feature_dim"
)

checkpoint_hidden_dim = checkpoint.get(
    "hidden_dim"
)

checkpoint_num_layers = checkpoint.get(
    "num_layers"
)


if checkpoint_seq_len is not None:

    if checkpoint_seq_len != SEQ_LEN:

        raise ValueError(
            f"Checkpoint SEQ_LEN = "
            f"{checkpoint_seq_len}, "
            f"expected {SEQ_LEN}"
        )


if checkpoint_feature_dim is not None:

    if checkpoint_feature_dim != FEATURE_DIM:

        raise ValueError(
            f"Checkpoint FEATURE_DIM = "
            f"{checkpoint_feature_dim}, "
            f"expected {FEATURE_DIM}"
        )


if checkpoint_hidden_dim is not None:

    if checkpoint_hidden_dim != HIDDEN_DIM:

        raise ValueError(
            f"Checkpoint HIDDEN_DIM = "
            f"{checkpoint_hidden_dim}, "
            f"expected {HIDDEN_DIM}"
        )


if checkpoint_num_layers is not None:

    if checkpoint_num_layers != NUM_LAYERS:

        raise ValueError(
            f"Checkpoint NUM_LAYERS = "
            f"{checkpoint_num_layers}, "
            f"expected {NUM_LAYERS}"
        )


# ============================================================
# CREATE MODEL COMPONENTS
# ============================================================

print()
print("=" * 78)
print("2. LOADING MODEL COMPONENTS")
print("=" * 78)


generator = Generator(
    FEATURE_DIM,
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)


supervisor = Supervisor(
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)


recovery = Recovery(
    HIDDEN_DIM,
    FEATURE_DIM,
    NUM_LAYERS
).to(DEVICE)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

generator.load_state_dict(
    checkpoint[
        "generator_state_dict"
    ]
)

supervisor.load_state_dict(
    checkpoint[
        "supervisor_state_dict"
    ]
)

recovery.load_state_dict(
    checkpoint[
        "recovery_state_dict"
    ]
)


generator.eval()
supervisor.eval()
recovery.eval()


print()
print("Generator loaded successfully.")
print("Supervisor loaded successfully.")
print("Recovery loaded successfully.")


# ============================================================
# LOAD STANDARD SCALER
#
# IMPORTANT:
# The V6 preprocessing script saved the scaler using
# joblib, not pickle.
#
# Therefore we use:
#
#     joblib.load()
#
# instead of:
#
#     pickle.load()
# ============================================================

print()
print("=" * 78)
print("3. LOADING STANDARD SCALER")
print("=" * 78)

print()
print("Scaler:")
print(SCALER_FILE)


if not os.path.exists(SCALER_FILE):

    raise FileNotFoundError(
        f"\nV6 scaler not found:\n"
        f"{SCALER_FILE}"
    )


scaler = joblib.load(
    SCALER_FILE
)


print()
print(
    "Scaler loaded successfully."
)

print(
    "Scaler class:",
    type(scaler).__name__
)


# ============================================================
# VERIFY SCALER
# ============================================================

if not hasattr(
    scaler,
    "inverse_transform"
):

    raise TypeError(
        "\nLoaded scaler does not provide "
        "inverse_transform()."
    )


if hasattr(
    scaler,
    "n_features_in_"
):

    print(
        "Scaler features:",
        scaler.n_features_in_
    )

    if scaler.n_features_in_ != FEATURE_DIM:

        raise ValueError(
            f"Scaler expects "
            f"{scaler.n_features_in_} features, "
            f"but V6 expects {FEATURE_DIM}."
        )


# ============================================================
# GENERATE SYNTHETIC SEQUENCES
# ============================================================

print()
print("=" * 78)
print("4. GENERATING SYNTHETIC SEQUENCES")
print("=" * 78)

print()
print(
    "Number of sequences:",
    NUM_SEQUENCES
)

print(
    "Sequence length:",
    SEQ_LEN
)

print(
    "Feature dimension:",
    FEATURE_DIM
)

print(
    "Batch size:",
    BATCH_SIZE
)


generated_batches = []


with torch.no_grad():

    for start in range(
        0,
        NUM_SEQUENCES,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            NUM_SEQUENCES
        )

        current_batch_size = (
            end - start
        )

        # ----------------------------------------------------
        # RANDOM NOISE
        # ----------------------------------------------------

        z = torch.randn(
            current_batch_size,
            SEQ_LEN,
            FEATURE_DIM,
            device=DEVICE
        )

        # ----------------------------------------------------
        # GENERATOR
        # ----------------------------------------------------

        generated_hidden = generator(
            z
        )

        # ----------------------------------------------------
        # SUPERVISOR
        # ----------------------------------------------------

        supervised_hidden = supervisor(
            generated_hidden
        )

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        generated_scaled = recovery(
            supervised_hidden
        )

        generated_batches.append(
            generated_scaled.cpu().numpy()
        )

        print(
            f"Generated "
            f"{end:4d}/{NUM_SEQUENCES} sequences"
        )


# ============================================================
# COMBINE
# ============================================================

synthetic_scaled = np.concatenate(
    generated_batches,
    axis=0
).astype(
    np.float32
)


# ============================================================
# SCALED DATA VALIDATION
# ============================================================

print()
print("=" * 78)
print("5. GENERATED DATA VALIDATION")
print("=" * 78)

print()
print(
    "Generated shape:",
    synthetic_scaled.shape
)

print(
    "NaN count:",
    np.isnan(
        synthetic_scaled
    ).sum()
)

print(
    "Inf count:",
    np.isinf(
        synthetic_scaled
    ).sum()
)

print(
    "Scaled minimum:",
    f"{synthetic_scaled.min():.8f}"
)

print(
    "Scaled maximum:",
    f"{synthetic_scaled.max():.8f}"
)


if np.isnan(
    synthetic_scaled
).any():

    raise ValueError(
        "Generated scaled data contains NaN."
    )


if np.isinf(
    synthetic_scaled
).any():

    raise ValueError(
        "Generated scaled data contains Inf."
    )


# ============================================================
# INVERSE TRANSFORM
# ============================================================

print()
print("=" * 78)
print("6. INVERSE TRANSFORMING TO REAL FEATURE SPACE")
print("=" * 78)


flat_scaled = synthetic_scaled.reshape(
    -1,
    FEATURE_DIM
)


synthetic_flat_raw = scaler.inverse_transform(
    flat_scaled
)


synthetic_raw = synthetic_flat_raw.reshape(
    NUM_SEQUENCES,
    SEQ_LEN,
    FEATURE_DIM
).astype(
    np.float32
)


print()
print(
    "Raw shape:",
    synthetic_raw.shape
)

print(
    "Raw minimum:",
    f"{synthetic_raw.min():.8f}"
)

print(
    "Raw maximum:",
    f"{synthetic_raw.max():.8f}"
)

print(
    "Raw NaN count:",
    np.isnan(
        synthetic_raw
    ).sum()
)

print(
    "Raw Inf count:",
    np.isinf(
        synthetic_raw
    ).sum()
)


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


# ============================================================
# SAVE SCALED DATA
# ============================================================

print()
print("=" * 78)
print("7. SAVING NUMPY OUTPUTS")
print("=" * 78)


np.save(
    SCALED_OUTPUT,
    synthetic_scaled
)


np.save(
    RAW_OUTPUT,
    synthetic_raw
)


print()
print(
    "Scaled output:"
)

print(
    SCALED_OUTPUT
)

print()
print(
    "Raw output:"
)

print(
    RAW_OUTPUT
)


# ============================================================
# CREATE LONG-FORM CSV
# ============================================================

print()
print("=" * 78)
print("8. CREATING CSV")
print("=" * 78)


rows = []


for sequence_id in range(
    NUM_SEQUENCES
):

    for day in range(
        SEQ_LEN
    ):

        row = {
            "Sequence_ID":
                sequence_id + 1,

            "Day":
                day + 1,
        }

        for feature_index, feature_name in enumerate(
            FEATURE_NAMES
        ):

            row[
                feature_name
            ] = float(
                synthetic_raw[
                    sequence_id,
                    day,
                    feature_index
                ]
            )

        rows.append(
            row
        )


synthetic_df = pd.DataFrame(
    rows
)


synthetic_df.to_csv(
    CSV_OUTPUT,
    index=False
)


print()
print(
    "CSV shape:",
    synthetic_df.shape
)

print()
print(
    "CSV output:"
)

print(
    CSV_OUTPUT
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print()
print("=" * 78)
print("9. SYNTHETIC DATA SUMMARY")
print("=" * 78)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    values = synthetic_raw[
        :,
        :,
        feature_index
    ].reshape(-1)

    print()
    print(
        feature_name
    )

    print(
        "  Mean:",
        f"{np.mean(values):.8f}"
    )

    print(
        "  Std :",
        f"{np.std(values):.8f}"
    )

    print(
        "  Min :",
        f"{np.min(values):.8f}"
    )

    print(
        "  Max :",
        f"{np.max(values):.8f}"
    )


# ============================================================
# PERCENTILES
# ============================================================

print()
print("=" * 78)
print("10. SYNTHETIC FEATURE PERCENTILES")
print("=" * 78)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    values = synthetic_raw[
        :,
        :,
        feature_index
    ].reshape(-1)

    percentiles = np.percentile(
        values,
        [
            1,
            5,
            25,
            50,
            75,
            95,
            99
        ]
    )

    print()
    print(
        feature_name
    )

    print(
        "  P01:",
        f"{percentiles[0]:.8f}"
    )

    print(
        "  P05:",
        f"{percentiles[1]:.8f}"
    )

    print(
        "  P25:",
        f"{percentiles[2]:.8f}"
    )

    print(
        "  P50:",
        f"{percentiles[3]:.8f}"
    )

    print(
        "  P75:",
        f"{percentiles[4]:.8f}"
    )

    print(
        "  P95:",
        f"{percentiles[5]:.8f}"
    )

    print(
        "  P99:",
        f"{percentiles[6]:.8f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("11. GENERATION COMPLETED SUCCESSFULLY")
print("=" * 78)

print()
print(
    "Model:",
    "TimeGAN_Stock_V6"
)

print(
    "Scaler:",
    "StandardScaler"
)

print(
    "Recovery:",
    "Linear"
)

print(
    "Sequences:",
    NUM_SEQUENCES
)

print(
    "Shape:",
    synthetic_raw.shape
)

print()
print(
    "Files created:"
)

print(
    "1.",
    SCALED_OUTPUT
)

print(
    "2.",
    RAW_OUTPUT
)

print(
    "3.",
    CSV_OUTPUT
)

print()
print("=" * 78)
print("GENERATION FINISHED")
print("=" * 78)