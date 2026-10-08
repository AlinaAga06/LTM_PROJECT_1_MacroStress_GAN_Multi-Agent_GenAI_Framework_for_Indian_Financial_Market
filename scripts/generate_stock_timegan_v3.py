"""
==============================================================================
MACROSTRESS-GAN
STOCK MARKET TIMEGAN V3 — GENERATION
==============================================================================

Loads:
    models/timegan/stock/timegan_stock_v3.pt

Generates:
    1000 synthetic 30-day stock-market sequences.

Saves:
    outputs/synthetic/stock/stock_v3_synthetic_sequences_scaled.npy
    outputs/synthetic/stock/stock_v3_synthetic_sequences.npy
    outputs/synthetic/stock/stock_v3_synthetic_financial_data.csv

V1 and V2 outputs are NOT modified.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


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
    "timegan_stock_v3.pt"
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

SCALED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v3_synthetic_sequences_scaled.npy"
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v3_synthetic_sequences.npy"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v3_synthetic_financial_data.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_SEQUENCES = 1000

SEQ_LEN = 30
FEATURE_DIM = 5

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class Embedder(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_layers
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        h, _ = self.gru(x)

        return self.activation(h)


class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim,
        feature_dim,
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
            feature_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        return self.activation(x)


class Generator(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_layers
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
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

        return self.activation(h)


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

        self.activation = nn.Sigmoid()

    def forward(self, h):

        s, _ = self.gru(h)

        s = self.fc(s)

        return self.activation(s)


class Discriminator(nn.Module):

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
            1
        )

    def forward(self, h):

        d, _ = self.gru(h)

        d = self.fc(d)

        return d


# =============================================================================
# MAIN
# =============================================================================

def main():

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V3 GENERATION")
    print("=" * 78)

    print()
    print(
        f"Device: {DEVICE}"
    )

    # =========================================================================
    # CHECK FILES
    # =========================================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"V3 model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )

    # =========================================================================
    # LOAD CHECKPOINT
    # =========================================================================

    print()
    print("=" * 78)
    print("LOADING V3 CHECKPOINT")
    print("=" * 78)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print()

    for key in [
        "model_version",
        "institution",
        "data_type",
        "feature_dim",
        "seq_len",
        "hidden_dim",
        "num_layers",
        "batch_size",
        "embedder_epochs",
        "supervisor_epochs",
        "joint_epochs",
        "learning_rate",
        "statistics_weight",
        "correlation_weight",
        "temporal_weight",
        "tail_weight",
        "low_tail_quantile",
        "high_tail_quantile",
        "seed"
    ]:

        if key in checkpoint:

            print(
                f"{key}: {checkpoint[key]}"
            )

    # =========================================================================
    # READ ARCHITECTURE
    # =========================================================================

    feature_dim = checkpoint.get(
        "feature_dim",
        FEATURE_DIM
    )

    seq_len = checkpoint.get(
        "seq_len",
        SEQ_LEN
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        24
    )

    num_layers = checkpoint.get(
        "num_layers",
        2
    )

    # =========================================================================
    # CREATE COMPONENTS
    # =========================================================================

    print()
    print("=" * 78)
    print("CREATING V3 MODEL COMPONENTS")
    print("=" * 78)

    embedder = Embedder(
        feature_dim,
        hidden_dim,
        num_layers
    ).to(DEVICE)

    recovery = Recovery(
        hidden_dim,
        feature_dim,
        num_layers
    ).to(DEVICE)

    generator = Generator(
        feature_dim,
        hidden_dim,
        num_layers
    ).to(DEVICE)

    supervisor = Supervisor(
        hidden_dim,
        num_layers
    ).to(DEVICE)

    discriminator = Discriminator(
        hidden_dim,
        num_layers
    ).to(DEVICE)

    # =========================================================================
    # LOAD WEIGHTS
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

    embedder.eval()
    recovery.eval()
    generator.eval()
    supervisor.eval()
    discriminator.eval()

    print()
    print("Embedder loaded successfully.")
    print("Recovery loaded successfully.")
    print("Generator loaded successfully.")
    print("Supervisor loaded successfully.")
    print("Discriminator loaded successfully.")

    # =========================================================================
    # LOAD SCALER
    # =========================================================================

    print()
    print("=" * 78)
    print("LOADING SCALER")
    print("=" * 78)

    scaler = joblib.load(
        SCALER_PATH
    )

    print()
    print(
        f"Scaler loaded from:\n{SCALER_PATH}"
    )

    # =========================================================================
    # GENERATE
    # =========================================================================

    print()
    print("=" * 78)
    print("GENERATING SYNTHETIC STOCK SEQUENCES")
    print("=" * 78)

    print()
    print(
        f"Number of sequences : {NUM_SEQUENCES}"
    )

    print(
        f"Sequence length     : {seq_len}"
    )

    print(
        f"Feature dimension   : {feature_dim}"
    )

    with torch.no_grad():

        noise = torch.rand(
            NUM_SEQUENCES,
            seq_len,
            feature_dim,
            device=DEVICE
        )

        generated_hidden = generator(
            noise
        )

        supervised_hidden = supervisor(
            generated_hidden
        )

        synthetic_scaled_tensor = recovery(
            supervised_hidden
        )

    synthetic_scaled = (
        synthetic_scaled_tensor
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # =========================================================================
    # VALIDATE SCALED DATA
    # =========================================================================

    print()
    print("=" * 78)
    print("SCALED OUTPUT VALIDATION")
    print("=" * 78)

    print()
    print(
        f"Shape : {synthetic_scaled.shape}"
    )

    print(
        f"Min   : {synthetic_scaled.min():.8f}"
    )

    print(
        f"Max   : {synthetic_scaled.max():.8f}"
    )

    print(
        f"NaN   : {np.isnan(synthetic_scaled).sum()}"
    )

    print(
        f"Inf   : {np.isinf(synthetic_scaled).sum()}"
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

    # =========================================================================
    # INVERSE TRANSFORM
    # =========================================================================

    print()
    print("=" * 78)
    print("INVERSE TRANSFORMING V3 DATA")
    print("=" * 78)

    flattened_scaled = synthetic_scaled.reshape(
        -1,
        feature_dim
    )

    flattened_raw = scaler.inverse_transform(
        flattened_scaled
    )

    synthetic_raw = flattened_raw.reshape(
        NUM_SEQUENCES,
        seq_len,
        feature_dim
    ).astype(
        np.float32
    )

    print()
    print(
        f"Raw shape : {synthetic_raw.shape}"
    )

    print(
        f"Raw min   : {synthetic_raw.min():.8f}"
    )

    print(
        f"Raw max   : {synthetic_raw.max():.8f}"
    )

    print(
        f"NaN       : {np.isnan(synthetic_raw).sum()}"
    )

    print(
        f"Inf       : {np.isinf(synthetic_raw).sum()}"
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

    # =========================================================================
    # SAVE NUMPY FILES
    # =========================================================================

    print()
    print("=" * 78)
    print("SAVING NUMPY OUTPUTS")
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
        f"Scaled output:\n{SCALED_OUTPUT}"
    )

    print()
    print(
        f"Raw output:\n{RAW_OUTPUT}"
    )

    # =========================================================================
    # CREATE CSV
    # =========================================================================

    print()
    print("=" * 78)
    print("CREATING V3 CSV")
    print("=" * 78)

    rows = []

    for sequence_id in range(
        NUM_SEQUENCES
    ):

        for day in range(
            seq_len
        ):

            row = {

                "Sequence_ID":
                    sequence_id,

                "Day":
                    day + 1
            }

            for feature_index, feature_name in enumerate(
                FEATURE_NAMES
            ):

                row[
                    feature_name
                ] = synthetic_raw[
                    sequence_id,
                    day,
                    feature_index
                ]

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
        f"CSV output:\n{CSV_OUTPUT}"
    )

    print()
    print(
        f"CSV shape: {synthetic_df.shape}"
    )

    # =========================================================================
    # FEATURE SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print("V3 SYNTHETIC FEATURE SUMMARY")
    print("=" * 78)

    print()

    for index, feature in enumerate(
        FEATURE_NAMES
    ):

        values = synthetic_raw[
            :,
            :,
            index
        ].reshape(
            -1
        )

        print(
            f"{feature}"
        )

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

        print()

    # =========================================================================
    # FINAL
    # =========================================================================

    print("=" * 78)
    print("V3 GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 78)

    print()
    print("Generated files:")

    print(
        f"1. {SCALED_OUTPUT}"
    )

    print(
        f"2. {RAW_OUTPUT}"
    )

    print(
        f"3. {CSV_OUTPUT}"
    )

    print()
    print("V1 and V2 generated files were NOT modified.")

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()