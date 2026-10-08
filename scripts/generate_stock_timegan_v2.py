"""
==============================================================================
MACROSTRESS-GAN
STOCK MARKET TIMEGAN V2 — SYNTHETIC DATA GENERATION
==============================================================================

Uses:
    models/timegan/stock/timegan_stock_v2.pt

Scaler:
    data/processed/institutions/stock/stock_timegan_scaler.pkl

Generates:
    1000 synthetic sequences
    30 days per sequence
    5 financial features

Outputs:
    outputs/synthetic/stock/
        stock_v2_synthetic_sequences_scaled.npy
        stock_v2_synthetic_sequences.npy
        stock_v2_synthetic_financial_data.csv

V1 is NOT modified.
==============================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# PATHS
# =============================================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock",
    "timegan_stock_v2.pt"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_scaler.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "stock"
)

SCALED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "stock_v2_synthetic_sequences_scaled.npy"
)

RAW_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "stock_v2_synthetic_sequences.npy"
)

CSV_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "stock_v2_synthetic_financial_data.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2

NUM_SEQUENCES = 1000

RANDOM_SEED = 42

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# EMBEDDER
# =============================================================================

class Embedder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.fc(h)

        h = self.activation(h)

        return h


# =============================================================================
# RECOVERY
# =============================================================================

class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim,
        output_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        x = self.activation(x)

        return x


# =============================================================================
# GENERATOR
# =============================================================================

class Generator(nn.Module):

    def __init__(
        self,
        noise_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            noise_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, z):

        h, _ = self.gru(z)

        h = self.fc(h)

        h = self.activation(h)

        return h


# =============================================================================
# SUPERVISOR
# =============================================================================
#
# IMPORTANT:
# The trained V2 checkpoint contains:
#
#     gru.weight_ih_l0
#     gru.weight_hh_l0
#     gru.bias_ih_l0
#     gru.bias_hh_l0
#
#     gru.weight_ih_l1
#     gru.weight_hh_l1
#     gru.bias_ih_l1
#     gru.bias_hh_l1
#
# Therefore the Supervisor MUST use NUM_LAYERS = 2.
#
# =============================================================================

class Supervisor(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        h, _ = self.gru(h)

        h = self.fc(h)

        h = self.activation(h)

        return h


# =============================================================================
# DISCRIMINATOR
# =============================================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, h):

        y, _ = self.gru(h)

        y = self.fc(y)

        return y


# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

def load_checkpoint():

    print()
    print("=" * 78)
    print("LOADING STOCK TIMEGAN V2 MODEL")
    print("=" * 78)

    print()
    print("Model path:")
    print(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nV2 model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print()
    print("Checkpoint loaded successfully.")

    return checkpoint


# =============================================================================
# LOAD MODELS
# =============================================================================

def load_models():

    checkpoint = load_checkpoint()

    print()
    print("=" * 78)
    print("CHECKPOINT INFORMATION")
    print("=" * 78)

    print()

    for key, value in checkpoint.items():

        if isinstance(value, dict):

            print(
                f"{key:<25} -> state_dict"
            )

        elif isinstance(value, list):

            print(
                f"{key:<25} -> list"
            )

        else:

            print(
                f"{key:<25} -> {value}"
            )

    # =========================================================================
    # CREATE ARCHITECTURES
    # =========================================================================

    embedder = Embedder(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    recovery = Recovery(
        hidden_dim=HIDDEN_DIM,
        output_dim=FEATURE_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    generator = Generator(
        noise_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # IMPORTANT:
    # V2 was trained with TWO Supervisor GRU layers.
    supervisor = Supervisor(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    discriminator = Discriminator(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # =========================================================================
    # LOAD STATE DICTIONARIES
    # =========================================================================

    embedder.load_state_dict(
        checkpoint["embedder"]
    )

    recovery.load_state_dict(
        checkpoint["recovery"]
    )

    generator.load_state_dict(
        checkpoint["generator"]
    )

    supervisor.load_state_dict(
        checkpoint["supervisor"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator"]
    )

    # =========================================================================
    # EVALUATION MODE
    # =========================================================================

    embedder.eval()
    recovery.eval()
    generator.eval()
    supervisor.eval()
    discriminator.eval()

    print()
    print("=" * 78)
    print("MODEL COMPONENTS LOADED SUCCESSFULLY")
    print("=" * 78)

    print()
    print("Embedder       : Loaded")
    print("Recovery       : Loaded")
    print("Generator      : Loaded")
    print("Supervisor     : Loaded")
    print("Discriminator  : Loaded")

    print()
    print("Architecture:")
    print(
        f"Feature dimension : {FEATURE_DIM}"
    )
    print(
        f"Sequence length   : {SEQ_LEN}"
    )
    print(
        f"Hidden dimension  : {HIDDEN_DIM}"
    )
    print(
        f"GRU layers        : {NUM_LAYERS}"
    )

    return (
        generator,
        supervisor,
        recovery
    )


# =============================================================================
# LOAD SCALER
# =============================================================================

def load_scaler():

    print()
    print("=" * 78)
    print("LOADING STOCK TIMEGAN SCALER")
    print("=" * 78)

    print()
    print("Scaler path:")
    print(SCALER_PATH)

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            f"\nScaler not found:\n{SCALER_PATH}"
        )

    with open(
        SCALER_PATH,
        "rb"
    ) as file:

        scaler = pickle.load(file)

    print()
    print("Scaler loaded successfully.")

    return scaler


# =============================================================================
# GENERATE SYNTHETIC DATA
# =============================================================================

def generate_synthetic_data(
    generator,
    supervisor,
    recovery
):

    print()
    print("=" * 78)
    print("GENERATING SYNTHETIC STOCK-MARKET SEQUENCES")
    print("=" * 78)

    print()
    print(
        f"Number of sequences : {NUM_SEQUENCES}"
    )

    print(
        f"Sequence length     : {SEQ_LEN}"
    )

    print(
        f"Feature count       : {FEATURE_DIM}"
    )

    print(
        f"Device              : {DEVICE}"
    )

    # =========================================================================
    # RANDOM NOISE
    # =========================================================================

    z = torch.rand(
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )

    print()
    print(
        f"Noise shape         : {tuple(z.shape)}"
    )

    # =========================================================================
    # GENERATE
    # =========================================================================

    with torch.no_grad():

        # Step 1
        generated_hidden = generator(z)

        # Step 2
        supervised_hidden = supervisor(
            generated_hidden
        )

        # Step 3
        synthetic_scaled = recovery(
            supervised_hidden
        )

    # =========================================================================
    # CONVERT TO NUMPY
    # =========================================================================

    synthetic_scaled = (
        synthetic_scaled
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # =========================================================================
    # CLIP TO QUANTILE-TRANSFORMER RANGE
    # =========================================================================

    synthetic_scaled = np.clip(
        synthetic_scaled,
        0.0,
        1.0
    )

    print()
    print("Generation completed.")

    print()
    print(
        f"Generated shape    : {synthetic_scaled.shape}"
    )

    print(
        f"Scaled minimum     : {synthetic_scaled.min():.8f}"
    )

    print(
        f"Scaled maximum     : {synthetic_scaled.max():.8f}"
    )

    return synthetic_scaled


# =============================================================================
# INVERSE TRANSFORM
# =============================================================================

def inverse_transform(
    synthetic_scaled,
    scaler
):

    print()
    print("=" * 78)
    print("INVERSE TRANSFORMING SYNTHETIC DATA")
    print("=" * 78)

    original_shape = synthetic_scaled.shape

    flattened = synthetic_scaled.reshape(
        -1,
        FEATURE_DIM
    )

    print()
    print(
        f"Flattened shape    : {flattened.shape}"
    )

    synthetic_raw = scaler.inverse_transform(
        flattened
    )

    synthetic_raw = synthetic_raw.reshape(
        original_shape
    )

    synthetic_raw = synthetic_raw.astype(
        np.float32
    )

    print()
    print(
        f"Raw shape          : {synthetic_raw.shape}"
    )

    print(
        f"Raw minimum        : {synthetic_raw.min():.8f}"
    )

    print(
        f"Raw maximum        : {synthetic_raw.max():.8f}"
    )

    return synthetic_raw


# =============================================================================
# BASIC VALIDATION
# =============================================================================

def validate_generated_data(
    synthetic_scaled,
    synthetic_raw
):

    print()
    print("=" * 78)
    print("BASIC GENERATED-DATA VALIDATION")
    print("=" * 78)

    expected_shape = (
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM
    )

    # =========================================================================
    # SHAPE
    # =========================================================================

    if synthetic_scaled.shape != expected_shape:

        raise ValueError(
            f"Unexpected scaled shape: "
            f"{synthetic_scaled.shape}"
        )

    if synthetic_raw.shape != expected_shape:

        raise ValueError(
            f"Unexpected raw shape: "
            f"{synthetic_raw.shape}"
        )

    # =========================================================================
    # NAN
    # =========================================================================

    scaled_nan = np.isnan(
        synthetic_scaled
    ).sum()

    raw_nan = np.isnan(
        synthetic_raw
    ).sum()

    # =========================================================================
    # INFINITY
    # =========================================================================

    scaled_inf = np.isinf(
        synthetic_scaled
    ).sum()

    raw_inf = np.isinf(
        synthetic_raw
    ).sum()

    print()
    print(
        f"Scaled NaN count   : {scaled_nan}"
    )

    print(
        f"Scaled Inf count   : {scaled_inf}"
    )

    print(
        f"Raw NaN count      : {raw_nan}"
    )

    print(
        f"Raw Inf count      : {raw_inf}"
    )

    if (
        scaled_nan > 0
        or scaled_inf > 0
        or raw_nan > 0
        or raw_inf > 0
    ):

        raise ValueError(
            "Generated data contains NaN or Inf values."
        )

    # =========================================================================
    # RANGE
    # =========================================================================

    scaled_min = float(
        synthetic_scaled.min()
    )

    scaled_max = float(
        synthetic_scaled.max()
    )

    print()
    print(
        f"Scaled range       : "
        f"{scaled_min:.8f} → {scaled_max:.8f}"
    )

    print()
    print("Basic validation: PASSED")


# =============================================================================
# SAVE NUMPY
# =============================================================================

def save_numpy(
    synthetic_scaled,
    synthetic_raw
):

    print()
    print("=" * 78)
    print("SAVING NUMPY OUTPUTS")
    print("=" * 78)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.save(
        SCALED_OUTPUT_PATH,
        synthetic_scaled
    )

    np.save(
        RAW_OUTPUT_PATH,
        synthetic_raw
    )

    print()
    print("Scaled NumPy:")
    print(
        SCALED_OUTPUT_PATH
    )

    print()
    print("Raw NumPy:")
    print(
        RAW_OUTPUT_PATH
    )


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(
    synthetic_raw
):

    print()
    print("=" * 78)
    print("SAVING SYNTHETIC FINANCIAL DATA CSV")
    print("=" * 78)

    rows = []

    for sequence_id in range(
        synthetic_raw.shape[0]
    ):

        for day in range(
            synthetic_raw.shape[1]
        ):

            row = {

                "Sequence_ID":
                    sequence_id + 1,

                "Day":
                    day + 1,

                "NIFTY50_Return":
                    float(
                        synthetic_raw[
                            sequence_id,
                            day,
                            0
                        ]
                    ),

                "CRUDE_OIL_Return":
                    float(
                        synthetic_raw[
                            sequence_id,
                            day,
                            1
                        ]
                    ),

                "USD_INR_Return":
                    float(
                        synthetic_raw[
                            sequence_id,
                            day,
                            2
                        ]
                    ),

                "INDIA_VIX_Change":
                    float(
                        synthetic_raw[
                            sequence_id,
                            day,
                            3
                        ]
                    ),

                "INDIA_10Y_YIELD_Change":
                    float(
                        synthetic_raw[
                            sequence_id,
                            day,
                            4
                        ]
                    )
            }

            rows.append(row)

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        CSV_OUTPUT_PATH,
        index=False
    )

    print()
    print("CSV path:")
    print(
        CSV_OUTPUT_PATH
    )

    print()
    print(
        f"CSV shape          : {df.shape}"
    )

    return df


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_summary(
    synthetic_raw
):

    print()
    print("=" * 78)
    print("SYNTHETIC DATA SUMMARY")
    print("=" * 78)

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        values = synthetic_raw[
            :,
            :,
            feature_index
        ].flatten()

        print()
        print(feature_name)

        print(
            f"  Mean : {np.mean(values):.8f}"
        )

        print(
            f"  Std  : {np.std(values):.8f}"
        )

        print(
            f"  Min  : {np.min(values):.8f}"
        )

        print(
            f"  Max  : {np.max(values):.8f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V2")
    print("SYNTHETIC DATA GENERATION")
    print("=" * 78)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Device:")
    print(DEVICE)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Scaler:")
    print(SCALER_PATH)

    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    (
        generator,
        supervisor,
        recovery
    ) = load_models()

    # =========================================================================
    # LOAD SCALER
    # =========================================================================

    scaler = load_scaler()

    # =========================================================================
    # GENERATE
    # =========================================================================

    synthetic_scaled = generate_synthetic_data(
        generator,
        supervisor,
        recovery
    )

    # =========================================================================
    # INVERSE TRANSFORM
    # =========================================================================

    synthetic_raw = inverse_transform(
        synthetic_scaled,
        scaler
    )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    validate_generated_data(
        synthetic_scaled,
        synthetic_raw
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_numpy(
        synthetic_scaled,
        synthetic_raw
    )

    df = save_csv(
        synthetic_raw
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_summary(
        synthetic_raw
    )

    # =========================================================================
    # FINAL MESSAGE
    # =========================================================================

    print()
    print("=" * 78)
    print("STOCK TIMEGAN V2 GENERATION COMPLETED")
    print("=" * 78)

    print()
    print(
        f"Generated sequences : {NUM_SEQUENCES}"
    )

    print(
        f"Sequence length     : {SEQ_LEN}"
    )

    print(
        f"Feature count       : {FEATURE_DIM}"
    )

    print(
        f"CSV rows            : {len(df)}"
    )

    print()
    print("Outputs:")

    print()
    print(
        "1. Scaled sequences:"
    )
    print(
        SCALED_OUTPUT_PATH
    )

    print()
    print(
        "2. Raw sequences:"
    )
    print(
        RAW_OUTPUT_PATH
    )

    print()
    print(
        "3. CSV:"
    )
    print(
        CSV_OUTPUT_PATH
    )

    print()
    print("=" * 78)
    print("V1 REMAINS COMPLETELY UNTOUCHED")
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()