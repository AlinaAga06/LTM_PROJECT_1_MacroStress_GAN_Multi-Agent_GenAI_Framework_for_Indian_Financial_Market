"""
MacroStress-GAN
Stock Market TimeGAN V6

V6 purpose:
    Test StandardScaler + linear Recovery instead of
    QuantileTransformer + Sigmoid Recovery.

Dataset:
    data/processed/institutions/stock/v6/
        stock_v6_timegan_sequences.npy
        stock_v6_standard_scaler.pkl
        stock_v6_feature_metadata.csv
        stock_v6_config.txt

Output:
    models/timegan/stock/timegan_stock_v6.pt
    models/timegan/stock/training_history_stock_v6.csv

IMPORTANT:
    V1-V5 are NOT modified by this script.
"""

import os
import random
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# WARNING CONTROL
# ============================================================

warnings.filterwarnings("ignore")


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
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "v6"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "training",
    "stock_v6"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FILES
# ============================================================

SEQUENCE_FILE = os.path.join(
    DATA_DIR,
    "stock_v6_timegan_sequences.npy"
)

SCALER_FILE = os.path.join(
    DATA_DIR,
    "stock_v6_standard_scaler.pkl"
)

METADATA_FILE = os.path.join(
    DATA_DIR,
    "stock_v6_feature_metadata.csv"
)

CONFIG_FILE = os.path.join(
    DATA_DIR,
    "stock_v6_config.txt"
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "timegan_stock_v6.pt"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "training_history_stock_v6.csv"
)


# ============================================================
# TIMEGAN CONFIGURATION
# ============================================================

SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 64

EMBEDDER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 400

LEARNING_RATE = 0.001

GRAD_CLIP = 1.0

# Loss weights
STATISTICS_WEIGHT = 1.0
CORRELATION_WEIGHT = 1.0
TEMPORAL_WEIGHT = 1.0

# Generator adversarial loss
ADVERSARIAL_WEIGHT = 1.0

# Supervised hidden-state loss
SUPERVISED_WEIGHT = 1.0

# Reconstruction contribution during joint training
RECONSTRUCTION_WEIGHT = 0.1


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
# UTILITY
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def set_requires_grad(model, value):
    for parameter in model.parameters():
        parameter.requires_grad = value


def freeze_all_except(model_to_train, models):
    for model in models:
        set_requires_grad(model, False)

    set_requires_grad(model_to_train, True)


# ============================================================
# MODEL 1 — EMBEDDER
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

        self.activation = nn.Tanh()

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.fc(h)

        h = self.activation(h)

        return h


# ============================================================
# MODEL 2 — RECOVERY
#
# IMPORTANT:
# V6 uses StandardScaler.
#
# Therefore we DO NOT use Sigmoid here.
#
# Standardized values can be negative and can be
# considerably larger than 1.
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

        # Linear output is intentional for V6.

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.fc(x)

        return x


# ============================================================
# MODEL 3 — GENERATOR
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
# MODEL 4 — SUPERVISOR
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
# MODEL 5 — DISCRIMINATOR
# ============================================================

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

        y, _ = self.gru(h)

        y = self.fc(y)

        # Return one probability per time step.
        return y


# ============================================================
# STATISTICS LOSS
# ============================================================

def statistics_loss(
    real,
    fake
):

    real_mean = torch.mean(
        real,
        dim=(0, 1)
    )

    fake_mean = torch.mean(
        fake,
        dim=(0, 1)
    )

    real_std = torch.std(
        real,
        dim=(0, 1),
        unbiased=False
    )

    fake_std = torch.std(
        fake,
        dim=(0, 1),
        unbiased=False
    )

    mean_loss = torch.mean(
        torch.abs(
            real_mean - fake_mean
        )
    )

    std_loss = torch.mean(
        torch.abs(
            real_std - fake_std
        )
    )

    return mean_loss + std_loss


# ============================================================
# CORRELATION LOSS
# ============================================================

def correlation_matrix(x):

    # Flatten sequence and batch dimensions.
    x = x.reshape(
        -1,
        x.shape[-1]
    )

    x = x - torch.mean(
        x,
        dim=0,
        keepdim=True
    )

    covariance = (
        x.T @ x
    ) / max(
        x.shape[0] - 1,
        1
    )

    std = torch.sqrt(
        torch.diag(covariance)
        + 1e-6
    )

    denominator = (
        std.unsqueeze(1)
        * std.unsqueeze(0)
    )

    correlation = (
        covariance / denominator
    )

    return correlation


def correlation_loss(
    real,
    fake
):

    real_corr = correlation_matrix(
        real
    )

    fake_corr = correlation_matrix(
        fake
    )

    return torch.mean(
        torch.abs(
            real_corr - fake_corr
        )
    )


# ============================================================
# TEMPORAL LOSS
# ============================================================

def temporal_loss(
    real,
    fake
):

    if real.shape[1] < 2:
        return torch.tensor(
            0.0,
            device=real.device
        )

    real_diff = (
        real[:, 1:, :]
        - real[:, :-1, :]
    )

    fake_diff = (
        fake[:, 1:, :]
        - fake[:, :-1, :]
    )

    return torch.mean(
        torch.abs(
            real_diff - fake_diff
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V6 TRAINING")
print("=" * 78)

print()

print("Device:", DEVICE)

print()
print("=" * 78)
print("1. LOADING V6 DATA")
print("=" * 78)

print()
print("Data file:")
print(SEQUENCE_FILE)

if not os.path.exists(SEQUENCE_FILE):

    raise FileNotFoundError(
        f"\nV6 sequence file not found:\n"
        f"{SEQUENCE_FILE}"
    )

data = np.load(
    SEQUENCE_FILE
).astype(
    np.float32
)

print()
print("Data shape:", data.shape)

print(
    "NaN count:",
    np.isnan(data).sum()
)

print(
    "Inf count:",
    np.isinf(data).sum()
)

print(
    "Minimum:",
    f"{np.min(data):.8f}"
)

print(
    "Maximum:",
    f"{np.max(data):.8f}"
)


# ============================================================
# DATA VALIDATION
# ============================================================

if data.ndim != 3:

    raise ValueError(
        f"Expected 3D array, got shape {data.shape}"
    )


if data.shape[1] != SEQ_LEN:

    raise ValueError(
        f"Expected sequence length {SEQ_LEN}, "
        f"got {data.shape[1]}"
    )


if data.shape[2] != FEATURE_DIM:

    raise ValueError(
        f"Expected feature dimension {FEATURE_DIM}, "
        f"got {data.shape[2]}"
    )


if np.isnan(data).any():

    raise ValueError(
        "Dataset contains NaN values."
    )


if np.isinf(data).any():

    raise ValueError(
        "Dataset contains Inf values."
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

print()
print("=" * 78)
print("2. MODEL CONFIGURATION")
print("=" * 78)

print()
print(
    "Sequence length :",
    SEQ_LEN
)

print(
    "Feature dimension:",
    FEATURE_DIM
)

print(
    "Hidden dimension :",
    HIDDEN_DIM
)

print(
    "GRU layers       :",
    NUM_LAYERS
)

print(
    "Batch size       :",
    BATCH_SIZE
)

print()
print(
    "Stage 1 epochs   :",
    EMBEDDER_EPOCHS
)

print(
    "Stage 2 epochs   :",
    SUPERVISOR_EPOCHS
)

print(
    "Stage 3 epochs   :",
    JOINT_EPOCHS
)

print(
    "Learning rate    :",
    LEARNING_RATE
)


# ============================================================
# CREATE MODELS
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
# PARAMETER COUNT
# ============================================================

total_parameters = (
    count_parameters(embedder)
    + count_parameters(recovery)
    + count_parameters(generator)
    + count_parameters(supervisor)
    + count_parameters(discriminator)
)

print()
print(
    "Total parameters :",
    f"{total_parameters:,}"
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

mse_loss = nn.MSELoss()

bce_loss = nn.BCEWithLogitsLoss()


# ============================================================
# DATASET / BATCHING
# ============================================================

num_sequences = data.shape[0]


def get_random_batch():

    indices = np.random.choice(
        num_sequences,
        size=min(
            BATCH_SIZE,
            num_sequences
        ),
        replace=False
    )

    batch = torch.from_numpy(
        data[indices]
    ).to(
        DEVICE
    )

    return batch


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []


# ============================================================
# STAGE 1
# EMBEDDER + RECOVERY
# ============================================================

print()
print("=" * 78)
print("3. STAGE 1 — EMBEDDER + RECOVERY")
print("=" * 78)

set_requires_grad(
    embedder,
    True
)

set_requires_grad(
    recovery,
    True
)

set_requires_grad(
    generator,
    False
)

set_requires_grad(
    supervisor,
    False
)

set_requires_grad(
    discriminator,
    False
)

optimizer_er = optim.Adam(
    list(embedder.parameters())
    + list(recovery.parameters()),
    lr=LEARNING_RATE
)

for epoch in range(
    1,
    EMBEDDER_EPOCHS + 1
):

    real_batch = get_random_batch()

    optimizer_er.zero_grad(
        set_to_none=True
    )

    H = embedder(
        real_batch
    )

    reconstructed = recovery(
        H
    )

    reconstruction_loss = mse_loss(
        reconstructed,
        real_batch
    )

    reconstruction_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(embedder.parameters())
        + list(recovery.parameters()),
        GRAD_CLIP
    )

    optimizer_er.step()

    history.append(
        {
            "stage": "stage1",
            "epoch": epoch,
            "reconstruction_loss":
                float(
                    reconstruction_loss.item()
                ),
            "supervised_loss": np.nan,
            "generator_loss": np.nan,
            "discriminator_loss": np.nan,
            "statistics_loss": np.nan,
            "correlation_loss": np.nan,
            "temporal_loss": np.nan,
        }
    )

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EMBEDDER_EPOCHS
    ):

        print(
            f"Epoch {epoch:3d}/{EMBEDDER_EPOCHS} "
            f"| Reconstruction Loss: "
            f"{reconstruction_loss.item():.6f}"
        )


# ============================================================
# STAGE 2
# SUPERVISOR
# ============================================================

print()
print("=" * 78)
print("4. STAGE 2 — SUPERVISOR")
print("=" * 78)

set_requires_grad(
    embedder,
    False
)

set_requires_grad(
    recovery,
    False
)

set_requires_grad(
    generator,
    False
)

set_requires_grad(
    supervisor,
    True
)

set_requires_grad(
    discriminator,
    False
)

optimizer_s = optim.Adam(
    supervisor.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(
    1,
    SUPERVISOR_EPOCHS + 1
):

    real_batch = get_random_batch()

    optimizer_s.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():

        H = embedder(
            real_batch
        )

    H_supervised = supervisor(
        H
    )

    # Next-step supervised objective.
    supervised_loss = mse_loss(
        H_supervised[:, :-1, :],
        H[:, 1:, :]
    )

    supervised_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        supervisor.parameters(),
        GRAD_CLIP
    )

    optimizer_s.step()

    history.append(
        {
            "stage": "stage2",
            "epoch": epoch,
            "reconstruction_loss": np.nan,
            "supervised_loss":
                float(
                    supervised_loss.item()
                ),
            "generator_loss": np.nan,
            "discriminator_loss": np.nan,
            "statistics_loss": np.nan,
            "correlation_loss": np.nan,
            "temporal_loss": np.nan,
        }
    )

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == SUPERVISOR_EPOCHS
    ):

        print(
            f"Epoch {epoch:3d}/{SUPERVISOR_EPOCHS} "
            f"| Supervised Loss: "
            f"{supervised_loss.item():.6f}"
        )


# ============================================================
# STAGE 3
# JOINT TRAINING
# ============================================================

print()
print("=" * 78)
print("5. STAGE 3 — JOINT TRAINING")
print("=" * 78)

print()
print("Weights:")
print(
    "  Statistics :",
    STATISTICS_WEIGHT
)

print(
    "  Correlation:",
    CORRELATION_WEIGHT
)

print(
    "  Temporal   :",
    TEMPORAL_WEIGHT
)


# ============================================================
# EXPLICITLY ENABLE GRADIENTS
# ============================================================

set_requires_grad(
    embedder,
    True
)

set_requires_grad(
    recovery,
    True
)

set_requires_grad(
    generator,
    True
)

set_requires_grad(
    supervisor,
    True
)

set_requires_grad(
    discriminator,
    True
)


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_g = optim.Adam(
    list(generator.parameters())
    + list(supervisor.parameters())
    + list(recovery.parameters()),
    lr=LEARNING_RATE
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# JOINT TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    JOINT_EPOCHS + 1
):

    # ========================================================
    # GET REAL BATCH
    # ========================================================

    real_batch = get_random_batch()

    current_batch_size = (
        real_batch.shape[0]
    )

    # ========================================================
    # ========================================================
    # A. DISCRIMINATOR UPDATE
    # ========================================================
    # ========================================================

    optimizer_d.zero_grad(
        set_to_none=True
    )

    # Freeze everything except discriminator.
    set_requires_grad(
        embedder,
        False
    )

    set_requires_grad(
        recovery,
        False
    )

    set_requires_grad(
        generator,
        False
    )

    set_requires_grad(
        supervisor,
        False
    )

    set_requires_grad(
        discriminator,
        True
    )

    discriminator.train()

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Do NOT use torch.no_grad() around discriminator().
    #
    # We detach the discriminator INPUT instead.
    # The discriminator itself must build a graph.
    # --------------------------------------------------------

    with torch.enable_grad():

        # ----------------------------------------------------
        # REAL HIDDEN REPRESENTATION
        # ----------------------------------------------------

        with torch.no_grad():

            H_real = embedder(
                real_batch
            )

        H_real = H_real.detach()

        Y_real = discriminator(
            H_real
        )

        real_labels = torch.ones_like(
            Y_real
        )

        real_loss = bce_loss(
            Y_real,
            real_labels
        )

        # ----------------------------------------------------
        # FAKE HIDDEN REPRESENTATION
        # ----------------------------------------------------

        with torch.no_grad():

            Z_d = torch.randn(
                current_batch_size,
                SEQ_LEN,
                FEATURE_DIM,
                device=DEVICE
            )

            E_hat_d = generator(
                Z_d
            )

            H_hat_d = supervisor(
                E_hat_d
            )

        H_hat_d = H_hat_d.detach()

        Y_fake = discriminator(
            H_hat_d
        )

        fake_labels = torch.zeros_like(
            Y_fake
        )

        fake_loss = bce_loss(
            Y_fake,
            fake_labels
        )

        discriminator_loss = (
            real_loss
            + fake_loss
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if not discriminator_loss.requires_grad:

            raise RuntimeError(
                "\n"
                "================================================\n"
                "DISCRIMINATOR GRAPH ERROR\n"
                "================================================\n"
                "discriminator_loss.requires_grad = False\n"
                "The discriminator output is not connected "
                "to its parameters.\n"
                "================================================"
            )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        discriminator_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            GRAD_CLIP
        )

        optimizer_d.step()


    # ========================================================
    # B. GENERATOR + SUPERVISOR + RECOVERY UPDATE
    # ========================================================

    optimizer_g.zero_grad(
        set_to_none=True
    )

    # Freeze discriminator.
    set_requires_grad(
        discriminator,
        False
    )

    # Generator path is trainable.
    set_requires_grad(
        generator,
        True
    )

    set_requires_grad(
        supervisor,
        True
    )

    set_requires_grad(
        recovery,
        True
    )

    # Embedder remains frozen during generator update.
    set_requires_grad(
        embedder,
        False
    )

    # --------------------------------------------------------
    # REAL DATA REPRESENTATION
    # --------------------------------------------------------

    with torch.no_grad():

        H_real_for_stats = embedder(
            real_batch
        )

    # --------------------------------------------------------
    # GENERATE SYNTHETIC DATA
    # --------------------------------------------------------

    Z = torch.randn(
        current_batch_size,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )

    E_hat = generator(
        Z
    )

    H_hat = supervisor(
        E_hat
    )

    X_hat = recovery(
        H_hat
    )

    # --------------------------------------------------------
    # ADVERSARIAL LOSS
    #
    # Discriminator parameters are frozen, but the graph
    # through discriminator remains active so gradients can
    # flow backward into Generator/Supervisor.
    # --------------------------------------------------------

    Y_fake_for_g = discriminator(
        H_hat
    )

    generator_labels = torch.ones_like(
        Y_fake_for_g
    )

    adversarial_loss = bce_loss(
        Y_fake_for_g,
        generator_labels
    )

    # --------------------------------------------------------
    # SUPERVISED TEMPORAL HIDDEN LOSS
    #
    # Compare generated hidden transitions with real
    # embedded hidden transitions.
    # --------------------------------------------------------

    with torch.no_grad():

        H_real_target = embedder(
            real_batch
        )

    supervised_loss_joint = mse_loss(
        H_hat[:, :-1, :],
        H_real_target[:, 1:, :]
    )

    # --------------------------------------------------------
    # RECONSTRUCTION / REALISTIC FEATURE LOSS
    #
    # This does NOT force synthetic data to equal real data.
    # It only keeps the generated representation in a
    # reasonable standardized feature space.
    # --------------------------------------------------------

    generated_reconstruction_loss = torch.mean(
        X_hat ** 2
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stat_loss = statistics_loss(
        real_batch,
        X_hat
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    corr_loss = correlation_loss(
        real_batch,
        X_hat
    )

    # --------------------------------------------------------
    # TEMPORAL
    # --------------------------------------------------------

    temp_loss = temporal_loss(
        real_batch,
        X_hat
    )

    # --------------------------------------------------------
    # TOTAL GENERATOR LOSS
    # --------------------------------------------------------

    generator_loss = (

        ADVERSARIAL_WEIGHT
        * adversarial_loss

        +

        SUPERVISED_WEIGHT
        * supervised_loss_joint

        +

        STATISTICS_WEIGHT
        * stat_loss

        +

        CORRELATION_WEIGHT
        * corr_loss

        +

        TEMPORAL_WEIGHT
        * temp_loss

        +

        RECONSTRUCTION_WEIGHT
        * generated_reconstruction_loss
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not generator_loss.requires_grad:

        raise RuntimeError(
            "\n"
            "================================================\n"
            "GENERATOR GRAPH ERROR\n"
            "================================================\n"
            "generator_loss.requires_grad = False\n"
            "The generator computation graph was detached.\n"
            "================================================"
        )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    generator_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(generator.parameters())
        + list(supervisor.parameters())
        + list(recovery.parameters()),
        GRAD_CLIP
    )

    optimizer_g.step()


    # ========================================================
    # RESTORE ALL GRADIENTS
    # ========================================================

    set_requires_grad(
        embedder,
        True
    )

    set_requires_grad(
        recovery,
        True
    )

    set_requires_grad(
        generator,
        True
    )

    set_requires_grad(
        supervisor,
        True
    )

    set_requires_grad(
        discriminator,
        True
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history.append(
        {
            "stage": "stage3",
            "epoch": epoch,
            "reconstruction_loss": np.nan,
            "supervised_loss":
                float(
                    supervised_loss_joint.item()
                ),
            "generator_loss":
                float(
                    generator_loss.item()
                ),
            "discriminator_loss":
                float(
                    discriminator_loss.item()
                ),
            "statistics_loss":
                float(
                    stat_loss.item()
                ),
            "correlation_loss":
                float(
                    corr_loss.item()
                ),
            "temporal_loss":
                float(
                    temp_loss.item()
                ),
        }
    )


    # ========================================================
    # PRINT
    # ========================================================

    if (
        epoch == 1
        or epoch % 20 == 0
        or epoch == JOINT_EPOCHS
    ):

        print(
            f"Epoch {epoch:3d}/{JOINT_EPOCHS} "
            f"| G={generator_loss.item():.6f} "
            f"| D={discriminator_loss.item():.6f} "
            f"| Stat={stat_loss.item():.6f} "
            f"| Corr={corr_loss.item():.6f} "
            f"| Temp={temp_loss.item():.6f}"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

print()
print("=" * 78)
print("6. SAVING TRAINING HISTORY")
print("=" * 78)

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)

print()
print(
    "Training history saved:"
)

print(
    HISTORY_FILE
)


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

print()
print("=" * 78)
print("7. SAVING V6 CHECKPOINT")
print("=" * 78)


checkpoint = {

    "model_version":
        "TimeGAN_Stock_V6",

    "institution":
        "Stock Market",

    "data_type":
        "REAL",

    "feature_names":
        FEATURE_NAMES,

    "feature_dim":
        FEATURE_DIM,

    "seq_len":
        SEQ_LEN,

    "hidden_dim":
        HIDDEN_DIM,

    "num_layers":
        NUM_LAYERS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "embedder_epochs":
        EMBEDDER_EPOCHS,

    "supervisor_epochs":
        SUPERVISOR_EPOCHS,

    "joint_epochs":
        JOINT_EPOCHS,

    "statistics_weight":
        STATISTICS_WEIGHT,

    "correlation_weight":
        CORRELATION_WEIGHT,

    "temporal_weight":
        TEMPORAL_WEIGHT,

    "adversarial_weight":
        ADVERSARIAL_WEIGHT,

    "supervised_weight":
        SUPERVISED_WEIGHT,

    "reconstruction_weight":
        RECONSTRUCTION_WEIGHT,

    "gradient_clip":
        GRAD_CLIP,

    "scaler_type":
        "StandardScaler",

    "recovery_activation":
        "linear",

    "architecture":
        "GRU",

    "embedder_state_dict":
        embedder.state_dict(),

    "recovery_state_dict":
        recovery.state_dict(),

    "generator_state_dict":
        generator.state_dict(),

    "supervisor_state_dict":
        supervisor.state_dict(),

    "discriminator_state_dict":
        discriminator.state_dict(),

    "total_parameters":
        total_parameters,

    "seed":
        SEED,
}


torch.save(
    checkpoint,
    CHECKPOINT_FILE
)

print()
print(
    "Checkpoint saved:"
)

print(
    CHECKPOINT_FILE
)


# ============================================================
# SAVE CONFIG SUMMARY
# ============================================================

config_summary = f"""
MacroStress-GAN
Stock Market TimeGAN V6

Model Version:
TimeGAN_Stock_V6

Institution:
Stock Market

Data Type:
REAL

Representation:
StandardScaler

Recovery:
Linear

Sequence Length:
{SEQ_LEN}

Feature Dimension:
{FEATURE_DIM}

Hidden Dimension:
{HIDDEN_DIM}

GRU Layers:
{NUM_LAYERS}

Batch Size:
{BATCH_SIZE}

Stage 1 Epochs:
{EMBEDDER_EPOCHS}

Stage 2 Epochs:
{SUPERVISOR_EPOCHS}

Stage 3 Epochs:
{JOINT_EPOCHS}

Learning Rate:
{LEARNING_RATE}

Statistics Weight:
{STATISTICS_WEIGHT}

Correlation Weight:
{CORRELATION_WEIGHT}

Temporal Weight:
{TEMPORAL_WEIGHT}

Adversarial Weight:
{ADVERSARIAL_WEIGHT}

Supervised Weight:
{SUPERVISED_WEIGHT}

Reconstruction Weight:
{RECONSTRUCTION_WEIGHT}

Gradient Clip:
{GRAD_CLIP}

Total Parameters:
{total_parameters}

Features:
{FEATURE_NAMES}
"""

CONFIG_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stock_v6_training_config.txt"
)

with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        config_summary
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("8. V6 TRAINING COMPLETED")
print("=" * 78)

print()
print("Model version : TimeGAN_Stock_V6")
print("Institution   : Stock Market")
print("Data type     : REAL")
print("Scaler        : StandardScaler")
print("Recovery      : Linear")

print()
print(
    "Input shape   :",
    data.shape
)

print(
    "Features      :",
    FEATURE_NAMES
)

print()
print(
    "Checkpoint:"
)

print(
    CHECKPOINT_FILE
)

print()
print(
    "History:"
)

print(
    HISTORY_FILE
)

print()
print("=" * 78)
print("TRAINING FINISHED SUCCESSFULLY")
print("=" * 78)