import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V4 GENERATION
# ============================================================

print("\n" + "=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V4 GENERATION")
print("=" * 78)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2

NUM_SEQUENCES = 1000

CHECKPOINT_PATH = (
    "models/timegan/stock/timegan_stock_v4.pt"
)

SCALER_PATH = (
    "data/processed/institutions/stock/"
    "stock_timegan_scaler.pkl"
)

OUTPUT_DIR = "outputs/synthetic/stock"

SCALED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v4_synthetic_sequences_scaled.npy"
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v4_synthetic_sequences.npy"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v4_synthetic_financial_data.csv"
)

FEATURE_NAMES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {DEVICE}")

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

        self.activation = nn.Sigmoid()

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        x = self.activation(x)

        return x


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

        s = self.activation(s)

        return s


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


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 78)
print("LOADING V4 CHECKPOINT")
print("=" * 78)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

print(
    f"model_version: "
    f"{checkpoint.get('model_version', 'N/A')}"
)

print(
    f"institution: "
    f"{checkpoint.get('institution', 'N/A')}"
)

print(
    f"data_type: "
    f"{checkpoint.get('data_type', 'N/A')}"
)

print(
    f"feature_dim: "
    f"{checkpoint.get('feature_dim', FEATURE_DIM)}"
)

print(
    f"seq_len: "
    f"{checkpoint.get('seq_len', SEQ_LEN)}"
)

print(
    f"hidden_dim: "
    f"{checkpoint.get('hidden_dim', HIDDEN_DIM)}"
)

print(
    f"num_layers: "
    f"{checkpoint.get('num_layers', NUM_LAYERS)}"
)

# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

checkpoint_feature_dim = checkpoint.get(
    "feature_dim",
    FEATURE_DIM
)

checkpoint_seq_len = checkpoint.get(
    "seq_len",
    SEQ_LEN
)

checkpoint_hidden_dim = checkpoint.get(
    "hidden_dim",
    HIDDEN_DIM
)

checkpoint_num_layers = checkpoint.get(
    "num_layers",
    NUM_LAYERS
)

if checkpoint_feature_dim != FEATURE_DIM:
    raise ValueError(
        f"Feature dimension mismatch: "
        f"{checkpoint_feature_dim} != {FEATURE_DIM}"
    )

if checkpoint_seq_len != SEQ_LEN:
    raise ValueError(
        f"Sequence length mismatch: "
        f"{checkpoint_seq_len} != {SEQ_LEN}"
    )

if checkpoint_hidden_dim != HIDDEN_DIM:
    raise ValueError(
        f"Hidden dimension mismatch: "
        f"{checkpoint_hidden_dim} != {HIDDEN_DIM}"
    )

if checkpoint_num_layers != NUM_LAYERS:
    raise ValueError(
        f"GRU layer mismatch: "
        f"{checkpoint_num_layers} != {NUM_LAYERS}"
    )

# ============================================================
# CREATE MODEL COMPONENTS
# ============================================================

print("\n" + "=" * 78)
print("CREATING V4 MODEL COMPONENTS")
print("=" * 78)

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
# LOAD MODEL WEIGHTS
# ============================================================

required_keys = [
    "embedder",
    "recovery",
    "generator",
    "supervisor",
    "discriminator",
]

for key in required_keys:

    if key not in checkpoint:

        raise KeyError(
            f"Checkpoint does not contain "
            f"required key: {key}"
        )

embedder.load_state_dict(
    checkpoint["embedder"]
)

print("Embedder loaded successfully.")

recovery.load_state_dict(
    checkpoint["recovery"]
)

print("Recovery loaded successfully.")

generator.load_state_dict(
    checkpoint["generator"]
)

print("Generator loaded successfully.")

supervisor.load_state_dict(
    checkpoint["supervisor"]
)

print("Supervisor loaded successfully.")

discriminator.load_state_dict(
    checkpoint["discriminator"]
)

print("Discriminator loaded successfully.")

# ============================================================
# EVALUATION MODE
# ============================================================

embedder.eval()
recovery.eval()
generator.eval()
supervisor.eval()
discriminator.eval()

# ============================================================
# LOAD SCALER
# ============================================================

print("\n" + "=" * 78)
print("LOADING SCALER")
print("=" * 78)

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_PATH}"
    )

scaler = joblib.load(SCALER_PATH)

print(f"Scaler loaded from:\n{os.path.abspath(SCALER_PATH)}")

# ============================================================
# GENERATE SYNTHETIC SEQUENCES
# ============================================================

print("\n" + "=" * 78)
print("GENERATING SYNTHETIC STOCK SEQUENCES")
print("=" * 78)

print(f"Number of sequences : {NUM_SEQUENCES}")
print(f"Sequence length     : {SEQ_LEN}")
print(f"Feature dimension   : {FEATURE_DIM}")

with torch.no_grad():

    z = torch.randn(
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )

    # Generator
    generated_hidden = generator(z)

    # Supervisor
    supervised_hidden = supervisor(
        generated_hidden
    )

    # Recovery
    generated_scaled = recovery(
        supervised_hidden
    )

generated_scaled = (
    generated_scaled
    .detach()
    .cpu()
    .numpy()
)

# ============================================================
# SCALED OUTPUT VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("SCALED OUTPUT VALIDATION")
print("=" * 78)

print(
    f"Shape : {generated_scaled.shape}"
)

print(
    f"Min   : {generated_scaled.min():.8f}"
)

print(
    f"Max   : {generated_scaled.max():.8f}"
)

print(
    f"NaN   : {np.isnan(generated_scaled).sum()}"
)

print(
    f"Inf   : {np.isinf(generated_scaled).sum()}"
)

if np.isnan(generated_scaled).any():
    raise ValueError(
        "Generated scaled data contains NaN."
    )

if np.isinf(generated_scaled).any():
    raise ValueError(
        "Generated scaled data contains Inf."
    )

# ============================================================
# INVERSE TRANSFORM
# ============================================================

print("\n" + "=" * 78)
print("INVERSE TRANSFORMING V4 DATA")
print("=" * 78)

flat_scaled = generated_scaled.reshape(
    -1,
    FEATURE_DIM
)

flat_raw = scaler.inverse_transform(
    flat_scaled
)

generated_raw = flat_raw.reshape(
    NUM_SEQUENCES,
    SEQ_LEN,
    FEATURE_DIM
)

print(
    f"Raw shape : {generated_raw.shape}"
)

print(
    f"Raw min   : {generated_raw.min():.8f}"
)

print(
    f"Raw max   : {generated_raw.max():.8f}"
)

print(
    f"NaN       : {np.isnan(generated_raw).sum()}"
)

print(
    f"Inf       : {np.isinf(generated_raw).sum()}"
)

if np.isnan(generated_raw).any():
    raise ValueError(
        "Generated raw data contains NaN."
    )

if np.isinf(generated_raw).any():
    raise ValueError(
        "Generated raw data contains Inf."
    )

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# SAVE NUMPY OUTPUTS
# ============================================================

print("\n" + "=" * 78)
print("SAVING NUMPY OUTPUTS")
print("=" * 78)

np.save(
    SCALED_OUTPUT,
    generated_scaled
)

print(
    f"Scaled output:\n"
    f"{os.path.abspath(SCALED_OUTPUT)}"
)

np.save(
    RAW_OUTPUT,
    generated_raw
)

print(
    f"\nRaw output:\n"
    f"{os.path.abspath(RAW_OUTPUT)}"
)

# ============================================================
# CREATE CSV
# ============================================================

print("\n" + "=" * 78)
print("CREATING V4 CSV")
print("=" * 78)

records = []

for sequence_id in range(NUM_SEQUENCES):

    for timestep in range(SEQ_LEN):

        row = {
            "Sequence_ID": sequence_id,
            "Time_Step": timestep,
        }

        for feature_index, feature_name in enumerate(
            FEATURE_NAMES
        ):

            row[feature_name] = generated_raw[
                sequence_id,
                timestep,
                feature_index
            ]

        records.append(row)

synthetic_df = pd.DataFrame(records)

synthetic_df.to_csv(
    CSV_OUTPUT,
    index=False
)

print(
    f"CSV output:\n"
    f"{os.path.abspath(CSV_OUTPUT)}"
)

print(
    f"\nCSV shape: {synthetic_df.shape}"
)

# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("V4 SYNTHETIC FEATURE SUMMARY")
print("=" * 78)

flat_raw_for_summary = generated_raw.reshape(
    -1,
    FEATURE_DIM
)

for index, feature_name in enumerate(
    FEATURE_NAMES
):

    values = flat_raw_for_summary[:, index]

    print(f"\n{feature_name}")

    print(
        f"  Mean : {values.mean():.8f}"
    )

    print(
        f"  Std  : {values.std():.8f}"
    )

    print(
        f"  Min  : {values.min():.8f}"
    )

    print(
        f"  Max  : {values.max():.8f}"
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 78)
print("V4 GENERATION COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nGenerated files:")

print(
    f"1. {os.path.abspath(SCALED_OUTPUT)}"
)

print(
    f"2. {os.path.abspath(RAW_OUTPUT)}"
)

print(
    f"3. {os.path.abspath(CSV_OUTPUT)}"
)

print("\nV1, V2 and V3 generated files were NOT modified.")

print("\n" + "=" * 78)