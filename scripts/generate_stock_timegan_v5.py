# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V5 GENERATION
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_SEQUENCES = 1000

SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock",
    "timegan_stock_v5.pt",
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_scaler.pkl",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "stock",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SCALED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_synthetic_sequences_scaled.npy"
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_synthetic_sequences.npy"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v5_synthetic_financial_data.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V5 GENERATION")
print("=" * 78)

print(f"\nDevice: {DEVICE}")

print("\nConfiguration:")
print(f"Sequences      : {NUM_SEQUENCES}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Features       : {FEATURE_DIM}")
print(f"Hidden dim     : {HIDDEN_DIM}")
print(f"GRU layers     : {NUM_LAYERS}")


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class Embedder(nn.Module):

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
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.fc(h)

        return self.activation(h)


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
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        return self.activation(x)


class Generator(nn.Module):

    def __init__(
        self,
        noise_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
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
            batch_first=True,
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
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, h):

        d, _ = self.gru(h)

        return self.fc(d)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n")
print("=" * 78)
print("LOADING V5 CHECKPOINT")
print("=" * 78)

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{MODEL_FILE}"
    )

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

print(
    f"\nModel version: "
    f"{checkpoint.get('model_version')}"
)

print(
    f"Institution: "
    f"{checkpoint.get('institution')}"
)

print(
    f"Feature dimension: "
    f"{checkpoint.get('feature_dim')}"
)

print(
    f"Sequence length: "
    f"{checkpoint.get('seq_len')}"
)


# ============================================================
# CREATE COMPONENTS
# ============================================================

embedder = Embedder(
    FEATURE_DIM,
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)

recovery = Recovery(
    HIDDEN_DIM,
    FEATURE_DIM,
    NUM_LAYERS
).to(DEVICE)

generator = Generator(
    FEATURE_DIM,
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)

supervisor = Supervisor(
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)

discriminator = Discriminator(
    HIDDEN_DIM,
    NUM_LAYERS
).to(DEVICE)


# ============================================================
# LOAD STATE DICTS
# ============================================================

embedder.load_state_dict(
    checkpoint["embedder_state_dict"]
)

recovery.load_state_dict(
    checkpoint["recovery_state_dict"]
)

generator.load_state_dict(
    checkpoint["generator_state_dict"]
)

supervisor.load_state_dict(
    checkpoint["supervisor_state_dict"]
)

discriminator.load_state_dict(
    checkpoint["discriminator_state_dict"]
)


# ============================================================
# EVALUATION MODE
# ============================================================

embedder.eval()
recovery.eval()
generator.eval()
supervisor.eval()
discriminator.eval()

print("\nAll V5 model components loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\n")
print("=" * 78)
print("LOADING STOCK SCALER")
print("=" * 78)

if not os.path.exists(SCALER_FILE):

    raise FileNotFoundError(
        f"\nScaler not found:\n{SCALER_FILE}"
    )

scaler = joblib.load(
    SCALER_FILE
)

print(
    f"\nScaler loaded:\n{SCALER_FILE}"
)


# ============================================================
# GENERATE
# ============================================================

print("\n")
print("=" * 78)
print("GENERATING SYNTHETIC STOCK SEQUENCES")
print("=" * 78)

with torch.no_grad():

    z = torch.rand(
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )

    generated_hidden = generator(z)

    supervised_hidden = supervisor(
        generated_hidden
    )

    synthetic_scaled = recovery(
        supervised_hidden
    )

    synthetic_scaled = (
        synthetic_scaled
        .cpu()
        .numpy()
    )


# ============================================================
# CHECK SCALED DATA
# ============================================================

print(
    f"\nGenerated shape: "
    f"{synthetic_scaled.shape}"
)

print(
    f"Scaled minimum: "
    f"{synthetic_scaled.min():.8f}"
)

print(
    f"Scaled maximum: "
    f"{synthetic_scaled.max():.8f}"
)

print(
    f"NaN count: "
    f"{np.isnan(synthetic_scaled).sum()}"
)

print(
    f"Inf count: "
    f"{np.isinf(synthetic_scaled).sum()}"
)


# ============================================================
# CLIP TO VALID SCALER RANGE
# ============================================================

synthetic_scaled = np.clip(
    synthetic_scaled,
    0.0,
    1.0
).astype(np.float32)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

print("\n")
print("=" * 78)
print("INVERSE TRANSFORM")
print("=" * 78)

flat_scaled = synthetic_scaled.reshape(
    -1,
    FEATURE_DIM
)

flat_raw = scaler.inverse_transform(
    flat_scaled
)

synthetic_raw = flat_raw.reshape(
    NUM_SEQUENCES,
    SEQ_LEN,
    FEATURE_DIM
).astype(np.float32)


# ============================================================
# RAW CHECK
# ============================================================

print(
    f"\nRaw minimum: "
    f"{synthetic_raw.min():.8f}"
)

print(
    f"Raw maximum: "
    f"{synthetic_raw.max():.8f}"
)

print(
    f"Raw NaN count: "
    f"{np.isnan(synthetic_raw).sum()}"
)

print(
    f"Raw Inf count: "
    f"{np.isinf(synthetic_raw).sum()}"
)


# ============================================================
# SAVE NUMPY FILES
# ============================================================

print("\n")
print("=" * 78)
print("SAVING NUMPY FILES")
print("=" * 78)

np.save(
    SCALED_OUTPUT,
    synthetic_scaled
)

np.save(
    RAW_OUTPUT,
    synthetic_raw
)

print(
    f"\nScaled sequences saved:\n"
    f"{SCALED_OUTPUT}"
)

print(
    f"\nRaw sequences saved:\n"
    f"{RAW_OUTPUT}"
)


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
# CREATE CSV
# ============================================================

print("\n")
print("=" * 78)
print("CREATING FINANCIAL DATA CSV")
print("=" * 78)

rows = []

for sequence_id in range(
    NUM_SEQUENCES
):

    for timestep in range(
        SEQ_LEN
    ):

        row = {
            "Sequence_ID":
                sequence_id,

            "Time_Step":
                timestep,
        }

        for feature_index, feature_name in enumerate(
            FEATURE_NAMES
        ):

            row[feature_name] = float(
                synthetic_raw[
                    sequence_id,
                    timestep,
                    feature_index
                ]
            )

        rows.append(row)


synthetic_df = pd.DataFrame(
    rows
)

synthetic_df.to_csv(
    CSV_OUTPUT,
    index=False
)


print(
    f"\nCSV shape: "
    f"{synthetic_df.shape}"
)

print(
    f"CSV saved:\n"
    f"{CSV_OUTPUT}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("V5 SYNTHETIC DATA SUMMARY")
print("=" * 78)

summary_rows = []

for index, feature in enumerate(
    FEATURE_NAMES
):

    values = synthetic_raw[
        :, :,
        index
    ].reshape(-1)

    summary_rows.append({

        "Feature":
            feature,

        "Mean":
            np.mean(values),

        "Std":
            np.std(values),

        "Min":
            np.min(values),

        "Max":
            np.max(values),

    })


summary_df = pd.DataFrame(
    summary_rows
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 78)
print("TIMEGAN STOCK V5 GENERATION COMPLETED")
print("=" * 78)

print(
    "\nGenerated sequences:"
    f" {synthetic_scaled.shape}"
)

print(
    "\nFiles:"
)

print(
    f"1. {SCALED_OUTPUT}"
)

print(
    f"2. {RAW_OUTPUT}"
)

print(
    f"3. {CSV_OUTPUT}"
)

print("\nNext step:")
print(
    "Run the V5 validation script using the same "
    "metrics used for V1-V4."
)

print("\n")