# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V4
# ============================================================
#
# V4 objective:
#
#   Build from the successful V2 training mechanics.
#
#   Preserve:
#       - TimeGAN architecture
#       - 30-day sequences
#       - 5 financial features
#       - 2-layer GRU
#       - V2 adversarial training
#       - correlation regularization
#       - temporal regularization
#
#   Add gently:
#       - variance matching
#       - quantile matching
#       - tail-aware regularization
#
# IMPORTANT:
#   V1, V2 and V3 files are NOT modified.
#
# Input:
#   data/processed/institutions/stock/
#       stock_timegan_sequences.npy
#
# Output:
#   models/timegan/stock/
#       timegan_stock_v4.pt
#
#   models/timegan/stock/
#       training_history_stock_v4.csv
#
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_PATH = (
    "data/processed/institutions/stock/"
    "stock_timegan_sequences.npy"
)

OUTPUT_DIR = "models/timegan/stock"

MODEL_PATH = (
    "models/timegan/stock/"
    "timegan_stock_v4.pt"
)

HISTORY_PATH = (
    "models/timegan/stock/"
    "training_history_stock_v4.csv"
)

# ------------------------------------------------------------
# TimeGAN configuration
# ------------------------------------------------------------

SEQ_LEN = 30
FEATURE_DIM = 5

HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 64

EMBEDDER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 400

LEARNING_RATE = 0.001

# ------------------------------------------------------------
# V2-compatible regularization
# ------------------------------------------------------------

STATISTICS_WEIGHT = 1.0
CORRELATION_WEIGHT = 2.0
TEMPORAL_WEIGHT = 1.5

# ------------------------------------------------------------
# V4 gentle tail regularization
# ------------------------------------------------------------

TAIL_WEIGHT = 0.25

TAIL_QUANTILES = [
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
]

# ------------------------------------------------------------
# Variance preservation
# ------------------------------------------------------------

VARIANCE_WEIGHT = 1.0

# ------------------------------------------------------------
# Training stabilization
# ------------------------------------------------------------

GRADIENT_CLIP = 1.0

# Number of synthetic samples used for regularization
# within each training batch.
REGULARIZATION_SAMPLE_SIZE = 64

# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

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

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility where practical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL COMPONENTS
# ============================================================

class Embedder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
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
            hidden_dim,
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
        num_layers,
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
            output_dim,
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
        num_layers,
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
            hidden_dim,
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
        num_layers,
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
            hidden_dim,
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        h_supervised, _ = self.gru(h)

        h_supervised = self.fc(
            h_supervised
        )

        h_supervised = self.activation(
            h_supervised
        )

        return h_supervised


class Discriminator(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers,
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
            1,
        )

    def forward(self, h):

        y, _ = self.gru(h)

        y = self.fc(y)

        return y


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(model):

    for module in model.modules():

        if isinstance(
            module,
            (nn.Linear, nn.GRU),
        ):

            for name, parameter in module.named_parameters():

                if "weight" in name:
                    nn.init.xavier_uniform_(
                        parameter
                    )

                elif "bias" in name:
                    nn.init.zeros_(
                        parameter
                    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print()
    print("=" * 78)
    print("LOADING STOCK TIMEGAN DATA")
    print("=" * 78)

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = np.load(DATA_PATH)

    print(f"Data path      : {DATA_PATH}")
    print(f"Shape          : {data.shape}")
    print(f"Min            : {data.min():.8f}")
    print(f"Max            : {data.max():.8f}")
    print(
        f"NaN            : "
        f"{np.isnan(data).sum()}"
    )
    print(
        f"Inf            : "
        f"{np.isinf(data).sum()}"
    )

    if data.ndim != 3:

        raise ValueError(
            "Expected data shape "
            "(samples, sequence_length, features)."
        )

    if data.shape[1] != SEQ_LEN:

        raise ValueError(
            f"Expected sequence length {SEQ_LEN}, "
            f"received {data.shape[1]}."
        )

    if data.shape[2] != FEATURE_DIM:

        raise ValueError(
            f"Expected {FEATURE_DIM} features, "
            f"received {data.shape[2]}."
        )

    if (
        np.isnan(data).any()
        or np.isinf(data).any()
    ):

        raise ValueError(
            "Dataset contains NaN or Inf."
        )

    return torch.tensor(
        data,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# BATCH SAMPLING
# ============================================================

def sample_batch(
    data,
    batch_size,
):

    indices = torch.randint(
        low=0,
        high=data.shape[0],
        size=(batch_size,),
        device=DEVICE,
    )

    return data[indices]


# ============================================================
# NOISE GENERATION
# ============================================================

def generate_noise(
    batch_size,
    seq_len,
    feature_dim,
):

    return torch.rand(
        batch_size,
        seq_len,
        feature_dim,
        device=DEVICE,
    )


# ============================================================
# STATISTICS LOSS
# ============================================================

def statistics_loss(
    real,
    fake,
):

    # Mean
    real_mean = torch.mean(
        real,
        dim=(0, 1),
    )

    fake_mean = torch.mean(
        fake,
        dim=(0, 1),
    )

    mean_loss = torch.mean(
        torch.abs(
            real_mean - fake_mean
        )
    )

    # Standard deviation
    real_std = torch.std(
        real,
        dim=(0, 1),
        unbiased=False,
    )

    fake_std = torch.std(
        fake,
        dim=(0, 1),
        unbiased=False,
    )

    std_loss = torch.mean(
        torch.abs(
            real_std - fake_std
        )
    )

    return mean_loss + std_loss


# ============================================================
# VARIANCE LOSS
# ============================================================

def variance_loss(
    real,
    fake,
):

    real_std = torch.std(
        real,
        dim=(0, 1),
        unbiased=False,
    )

    fake_std = torch.std(
        fake,
        dim=(0, 1),
        unbiased=False,
    )

    # Normalize by real dispersion so that features
    # with larger numerical scale do not dominate.
    denominator = (
        real_std.detach()
        + 1e-6
    )

    loss = torch.mean(
        torch.abs(
            real_std - fake_std
        ) / denominator
    )

    return loss


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_matrix(x):

    # x:
    # [batch, sequence, features]

    flattened = x.reshape(
        -1,
        x.shape[-1],
    )

    centered = (
        flattened -
        torch.mean(
            flattened,
            dim=0,
            keepdim=True,
        )
    )

    covariance = (
        centered.T @ centered
    )

    denominator = (
        centered.shape[0] - 1
    )

    covariance = (
        covariance /
        max(denominator, 1)
    )

    std = torch.sqrt(
        torch.diag(covariance)
        + 1e-8
    )

    corr = covariance / (
        torch.outer(std, std)
        + 1e-8
    )

    return corr


# ============================================================
# CORRELATION LOSS
# ============================================================

def correlation_loss(
    real,
    fake,
):

    real_corr = correlation_matrix(
        real
    )

    fake_corr = correlation_matrix(
        fake
    )

    # Ignore diagonal because diagonal
    # is always approximately 1.
    mask = (
        1.0 -
        torch.eye(
            real_corr.shape[0],
            device=DEVICE,
        )
    )

    difference = (
        torch.abs(
            real_corr -
            fake_corr
        )
        * mask
    )

    normalization = (
        torch.sum(mask)
        + 1e-8
    )

    return (
        torch.sum(difference)
        / normalization
    )


# ============================================================
# TEMPORAL LOSS
# ============================================================

def temporal_loss(
    real,
    fake,
):

    if real.shape[1] < 2:
        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    real_previous = real[:, :-1, :]
    real_next = real[:, 1:, :]

    fake_previous = fake[:, :-1, :]
    fake_next = fake[:, 1:, :]

    real_delta = (
        real_next -
        real_previous
    )

    fake_delta = (
        fake_next -
        fake_previous
    )

    return torch.mean(
        torch.abs(
            real_delta -
            fake_delta
        )
    )


# ============================================================
# QUANTILE LOSS
# ============================================================

def quantile_loss(
    real,
    fake,
    quantiles,
):

    # Flatten time and batch dimensions.
    real_flat = real.reshape(
        -1,
        real.shape[-1],
    )

    fake_flat = fake.reshape(
        -1,
        fake.shape[-1],
    )

    total_loss = torch.tensor(
        0.0,
        device=DEVICE,
    )

    for q in quantiles:

        real_q = torch.quantile(
            real_flat,
            q,
            dim=0,
        )

        fake_q = torch.quantile(
            fake_flat,
            q,
            dim=0,
        )

        total_loss = (
            total_loss +
            torch.mean(
                torch.abs(
                    real_q -
                    fake_q
                )
            )
        )

    return (
        total_loss /
        len(quantiles)
    )


# ============================================================
# TAIL LOSS
# ============================================================

def tail_loss(
    real,
    fake,
):

    real_flat = real.reshape(
        -1,
        real.shape[-1],
    )

    fake_flat = fake.reshape(
        -1,
        fake.shape[-1],
    )

    # Focus on lower and upper tails.
    real_low = torch.quantile(
        real_flat,
        0.05,
        dim=0,
    )

    fake_low = torch.quantile(
        fake_flat,
        0.05,
        dim=0,
    )

    real_high = torch.quantile(
        real_flat,
        0.95,
        dim=0,
    )

    fake_high = torch.quantile(
        fake_flat,
        0.95,
        dim=0,
    )

    lower_loss = torch.mean(
        torch.abs(
            real_low -
            fake_low
        )
    )

    upper_loss = torch.mean(
        torch.abs(
            real_high -
            fake_high
        )
    )

    return (
        lower_loss +
        upper_loss
    ) / 2.0


# ============================================================
# DISCRIMINATOR LOSS
# ============================================================

def discriminator_loss(
    discriminator,
    real_h,
    fake_h,
):

    real_logits = discriminator(
        real_h
    )

    fake_logits = discriminator(
        fake_h
    )

    # BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    real_labels = torch.ones_like(
        real_logits
    )

    fake_labels = torch.zeros_like(
        fake_logits
    )

    real_loss = criterion(
        real_logits,
        real_labels,
    )

    fake_loss = criterion(
        fake_logits,
        fake_labels,
    )

    return (
        real_loss +
        fake_loss
    ) / 2.0


# ============================================================
# GENERATOR ADVERSARIAL LOSS
# ============================================================

def generator_adversarial_loss(
    discriminator,
    fake_h,
):

    fake_logits = discriminator(
        fake_h
    )

    target = torch.ones_like(
        fake_logits
    )

    criterion = nn.BCEWithLogitsLoss()

    return criterion(
        fake_logits,
        target,
    )


# ============================================================
# STAGE 1
# EMBEDDER + RECOVERY
# ============================================================

def train_embedder_recovery(
    data,
    embedder,
    recovery,
):

    print()
    print("=" * 78)
    print("STAGE 1: EMBEDDER + RECOVERY")
    print("=" * 78)

    optimizer = optim.Adam(
        list(
            embedder.parameters()
        )
        +
        list(
            recovery.parameters()
        ),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    for epoch in range(
        1,
        EMBEDDER_EPOCHS + 1,
    ):

        batch = sample_batch(
            data,
            BATCH_SIZE,
        )

        optimizer.zero_grad()

        h = embedder(batch)

        reconstructed = recovery(h)

        loss = criterion(
            reconstructed,
            batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                embedder.parameters()
            )
            +
            list(
                recovery.parameters()
            ),
            GRADIENT_CLIP,
        )

        optimizer.step()

        history.append(
            float(loss.item())
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EMBEDDER_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/"
                f"{EMBEDDER_EPOCHS} | "
                f"Reconstruction Loss: "
                f"{loss.item():.6f}"
            )

    print()
    print(
        f"Stage 1 start loss : "
        f"{history[0]:.6f}"
    )

    print(
        f"Stage 1 final loss : "
        f"{history[-1]:.6f}"
    )

    return history


# ============================================================
# STAGE 2
# SUPERVISOR
# ============================================================

def train_supervisor(
    data,
    embedder,
    supervisor,
):

    print()
    print("=" * 78)
    print("STAGE 2: SUPERVISOR")
    print("=" * 78)

    optimizer = optim.Adam(
        supervisor.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    for epoch in range(
        1,
        SUPERVISOR_EPOCHS + 1,
    ):

        batch = sample_batch(
            data,
            BATCH_SIZE,
        )

        # Embedder is not updated in Stage 2.
        with torch.no_grad():

            h = embedder(batch)

        optimizer.zero_grad()

        h_supervised = supervisor(h)

        if h.shape[1] > 1:

            loss = criterion(
                h_supervised[:, :-1, :],
                h[:, 1:, :],
            )

        else:

            loss = criterion(
                h_supervised,
                h,
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            supervisor.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        history.append(
            float(loss.item())
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == SUPERVISOR_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/"
                f"{SUPERVISOR_EPOCHS} | "
                f"Supervised Loss: "
                f"{loss.item():.6f}"
            )

    print()
    print(
        f"Stage 2 start loss : "
        f"{history[0]:.6f}"
    )

    print(
        f"Stage 2 final loss : "
        f"{history[-1]:.6f}"
    )

    return history


# ============================================================
# STAGE 3
# JOINT TRAINING
# ============================================================

def train_joint(
    data,
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
):

    print()
    print("=" * 78)
    print("STAGE 3: V4 JOINT TRAINING")
    print("=" * 78)

    print()
    print("V4 regularization:")
    print(
        f"  Statistics weight : "
        f"{STATISTICS_WEIGHT}"
    )
    print(
        f"  Correlation weight: "
        f"{CORRELATION_WEIGHT}"
    )
    print(
        f"  Temporal weight   : "
        f"{TEMPORAL_WEIGHT}"
    )
    print(
        f"  Variance weight   : "
        f"{VARIANCE_WEIGHT}"
    )
    print(
        f"  Tail weight      : "
        f"{TAIL_WEIGHT}"
    )
    print(
        f"  Tail quantiles   : "
        f"{TAIL_QUANTILES}"
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    embedder_recovery_optimizer = optim.Adam(
        list(
            embedder.parameters()
        )
        +
        list(
            recovery.parameters()
        ),
        lr=LEARNING_RATE,
    )

    generator_supervisor_optimizer = optim.Adam(
        list(
            generator.parameters()
        )
        +
        list(
            supervisor.parameters()
        ),
        lr=LEARNING_RATE,
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    mse = nn.MSELoss()

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        JOINT_EPOCHS + 1,
    ):

        # ====================================================
        # PART A
        # Train Embedder + Recovery
        # ====================================================

        real_batch = sample_batch(
            data,
            BATCH_SIZE,
        )

        embedder_recovery_optimizer.zero_grad()

        # Fresh forward pass.
        real_h = embedder(
            real_batch
        )

        real_reconstructed = recovery(
            real_h
        )

        reconstruction_loss = mse(
            real_reconstructed,
            real_batch,
        )

        reconstruction_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                embedder.parameters()
            )
            +
            list(
                recovery.parameters()
            ),
            GRADIENT_CLIP,
        )

        embedder_recovery_optimizer.step()

        # ====================================================
        # PART B
        # Train Generator + Supervisor
        # ====================================================

        real_batch_g = sample_batch(
            data,
            BATCH_SIZE,
        )

        generator_supervisor_optimizer.zero_grad()

        # ----------------------------------------------------
        # Real representation for supervised loss.
        # Embedder parameters are not updated through this
        # branch.
        # ----------------------------------------------------

        with torch.no_grad():

            real_h_g = embedder(
                real_batch_g
            )

        z = generate_noise(
            BATCH_SIZE,
            SEQ_LEN,
            FEATURE_DIM,
        )

        generated_h = generator(z)

        supervised_h = supervisor(
            generated_h
        )

        # ----------------------------------------------------
        # Supervised temporal representation loss.
        # ----------------------------------------------------

        if SEQ_LEN > 1:

            supervised_loss = mse(
                supervised_h[:, :-1, :],
                generated_h[:, 1:, :],
            )

        else:

            supervised_loss = mse(
                supervised_h,
                generated_h,
            )

        # ----------------------------------------------------
        # Recovery from supervised generator representation.
        # ----------------------------------------------------

        synthetic_data = recovery(
            supervised_h
        )

        # ----------------------------------------------------
        # Generate statistical target.
        # ----------------------------------------------------

        statistics = statistics_loss(
            real_batch_g,
            synthetic_data,
        )

        # ----------------------------------------------------
        # Correlation.
        # ----------------------------------------------------

        correlation = correlation_loss(
            real_batch_g,
            synthetic_data,
        )

        # ----------------------------------------------------
        # Temporal.
        # ----------------------------------------------------

        temporal = temporal_loss(
            real_batch_g,
            synthetic_data,
        )

        # ----------------------------------------------------
        # Variance preservation.
        # ----------------------------------------------------

        variance = variance_loss(
            real_batch_g,
            synthetic_data,
        )

        # ----------------------------------------------------
        # Quantile / tail matching.
        # ----------------------------------------------------

        quantile = quantile_loss(
            real_batch_g,
            synthetic_data,
            TAIL_QUANTILES,
        )

        tail = tail_loss(
            real_batch_g,
            synthetic_data,
        )

        # ----------------------------------------------------
        # Adversarial loss.
        #
        # IMPORTANT:
        # Use generator -> supervisor -> discriminator through
        # the current graph. This is a fresh forward pass and
        # avoids reusing a graph from another optimizer step.
        # ----------------------------------------------------

        adversarial = (
            generator_adversarial_loss(
                discriminator,
                supervised_h,
            )
        )

        # ----------------------------------------------------
        # Total generator loss.
        # ----------------------------------------------------

        generator_loss = (
            adversarial
            +
            0.1 * supervised_loss
            +
            STATISTICS_WEIGHT * statistics
            +
            CORRELATION_WEIGHT * correlation
            +
            TEMPORAL_WEIGHT * temporal
            +
            VARIANCE_WEIGHT * variance
            +
            TAIL_WEIGHT * quantile
            +
            TAIL_WEIGHT * tail
        )

        generator_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                generator.parameters()
            )
            +
            list(
                supervisor.parameters()
            ),
            GRADIENT_CLIP,
        )

        generator_supervisor_optimizer.step()

        # ====================================================
        # PART C
        # Train Discriminator
        # ====================================================

        discriminator_optimizer.zero_grad()

        # ----------------------------------------------------
        # Fresh real representation.
        # Detach because discriminator must not update
        # embedder.
        # ----------------------------------------------------

        with torch.no_grad():

            real_batch_d = sample_batch(
                data,
                BATCH_SIZE,
            )

            real_h_d = embedder(
                real_batch_d
            )

        # ----------------------------------------------------
        # Fresh fake representation.
        # Detach because discriminator should not update
        # generator/supervisor.
        # ----------------------------------------------------

        with torch.no_grad():

            z_d = generate_noise(
                BATCH_SIZE,
                SEQ_LEN,
                FEATURE_DIM,
            )

            fake_h_d = generator(
                z_d
            )

            fake_h_d = supervisor(
                fake_h_d
            )

        d_loss = discriminator_loss(
            discriminator,
            real_h_d.detach(),
            fake_h_d.detach(),
        )

        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            GRADIENT_CLIP,
        )

        discriminator_optimizer.step()

        # ====================================================
        # Save history
        # ====================================================

        record = {
            "epoch": epoch,

            "generator_loss":
                float(
                    generator_loss.item()
                ),

            "discriminator_loss":
                float(
                    d_loss.item()
                ),

            "reconstruction_loss":
                float(
                    reconstruction_loss.item()
                ),

            "supervised_loss":
                float(
                    supervised_loss.item()
                ),

            "statistics_loss":
                float(
                    statistics.item()
                ),

            "correlation_loss":
                float(
                    correlation.item()
                ),

            "temporal_loss":
                float(
                    temporal.item()
                ),

            "variance_loss":
                float(
                    variance.item()
                ),

            "quantile_loss":
                float(
                    quantile.item()
                ),

            "tail_loss":
                float(
                    tail.item()
                ),

            "adversarial_loss":
                float(
                    adversarial.item()
                ),
        }

        history.append(record)

        # ====================================================
        # Console output
        # ====================================================

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == JOINT_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/"
                f"{JOINT_EPOCHS} | "
                f"G={generator_loss.item():.6f} | "
                f"D={d_loss.item():.6f} | "
                f"Stat={statistics.item():.6f} | "
                f"Corr={correlation.item():.6f} | "
                f"Temp={temporal.item():.6f} | "
                f"Var={variance.item():.6f} | "
                f"Quant={quantile.item():.6f} | "
                f"Tail={tail.item():.6f}"
            )

    return history


# ============================================================
# MODEL SUMMARY
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
    stage1_history,
    stage2_history,
    stage3_history,
):

    print()
    print("=" * 78)
    print("SAVING V4 CHECKPOINT")
    print("=" * 78)

    checkpoint = {

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "model_version":
            "TimeGAN_Stock_V4",

        "institution":
            "Stock Market",

        "data_type":
            "REAL",

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

        "embedder_epochs":
            EMBEDDER_EPOCHS,

        "supervisor_epochs":
            SUPERVISOR_EPOCHS,

        "joint_epochs":
            JOINT_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        # ----------------------------------------------------
        # V2-compatible losses
        # ----------------------------------------------------

        "statistics_weight":
            STATISTICS_WEIGHT,

        "correlation_weight":
            CORRELATION_WEIGHT,

        "temporal_weight":
            TEMPORAL_WEIGHT,

        # ----------------------------------------------------
        # V4 additions
        # ----------------------------------------------------

        "variance_weight":
            VARIANCE_WEIGHT,

        "tail_weight":
            TAIL_WEIGHT,

        "tail_quantiles":
            TAIL_QUANTILES,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed":
            SEED,

        "feature_names":
            FEATURE_NAMES,

        # ----------------------------------------------------
        # Model state
        # ----------------------------------------------------

        "embedder":
            embedder.state_dict(),

        "recovery":
            recovery.state_dict(),

        "generator":
            generator.state_dict(),

        "supervisor":
            supervisor.state_dict(),

        "discriminator":
            discriminator.state_dict(),

        # ----------------------------------------------------
        # Training history
        # ----------------------------------------------------

        "stage1_final_loss":
            stage1_history[-1],

        "stage2_final_loss":
            stage2_history[-1],

        "stage3_final_generator_loss":
            stage3_history[-1][
                "generator_loss"
            ],

        "stage3_final_discriminator_loss":
            stage3_history[-1][
                "discriminator_loss"
            ],
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(
        f"Checkpoint saved to:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    stage1_history,
    stage2_history,
    stage3_history,
):

    rows = []

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    for epoch, loss in enumerate(
        stage1_history,
        start=1,
    ):

        rows.append({
            "stage":
                "Stage1_Embedder_Recovery",

            "epoch":
                epoch,

            "reconstruction_loss":
                loss,

            "supervised_loss":
                np.nan,

            "generator_loss":
                np.nan,

            "discriminator_loss":
                np.nan,

            "statistics_loss":
                np.nan,

            "correlation_loss":
                np.nan,

            "temporal_loss":
                np.nan,

            "variance_loss":
                np.nan,

            "quantile_loss":
                np.nan,

            "tail_loss":
                np.nan,

            "adversarial_loss":
                np.nan,
        })

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    for epoch, loss in enumerate(
        stage2_history,
        start=1,
    ):

        rows.append({
            "stage":
                "Stage2_Supervisor",

            "epoch":
                epoch,

            "reconstruction_loss":
                np.nan,

            "supervised_loss":
                loss,

            "generator_loss":
                np.nan,

            "discriminator_loss":
                np.nan,

            "statistics_loss":
                np.nan,

            "correlation_loss":
                np.nan,

            "temporal_loss":
                np.nan,

            "variance_loss":
                np.nan,

            "quantile_loss":
                np.nan,

            "tail_loss":
                np.nan,

            "adversarial_loss":
                np.nan,
        })

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    for record in stage3_history:

        rows.append({
            "stage":
                "Stage3_Joint",

            "epoch":
                record["epoch"],

            "reconstruction_loss":
                record[
                    "reconstruction_loss"
                ],

            "supervised_loss":
                record[
                    "supervised_loss"
                ],

            "generator_loss":
                record[
                    "generator_loss"
                ],

            "discriminator_loss":
                record[
                    "discriminator_loss"
                ],

            "statistics_loss":
                record[
                    "statistics_loss"
                ],

            "correlation_loss":
                record[
                    "correlation_loss"
                ],

            "temporal_loss":
                record[
                    "temporal_loss"
                ],

            "variance_loss":
                record[
                    "variance_loss"
                ],

            "quantile_loss":
                record[
                    "quantile_loss"
                ],

            "tail_loss":
                record[
                    "tail_loss"
                ],

            "adversarial_loss":
                record[
                    "adversarial_loss"
                ],
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print()
    print(
        f"Training history saved to:\n"
        f"{os.path.abspath(HISTORY_PATH)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V4 TRAINING")
    print("=" * 78)

    print()
    print(f"Device            : {DEVICE}")

    print()
    print("Architecture:")
    print(
        f"  Sequence length : {SEQ_LEN}"
    )
    print(
        f"  Feature dim     : {FEATURE_DIM}"
    )
    print(
        f"  Hidden dim      : {HIDDEN_DIM}"
    )
    print(
        f"  GRU layers      : {NUM_LAYERS}"
    )
    print(
        f"  Batch size      : {BATCH_SIZE}"
    )

    print()
    print("Training:")
    print(
        f"  Stage 1 epochs  : "
        f"{EMBEDDER_EPOCHS}"
    )
    print(
        f"  Stage 2 epochs  : "
        f"{SUPERVISOR_EPOCHS}"
    )
    print(
        f"  Stage 3 epochs  : "
        f"{JOINT_EPOCHS}"
    )
    print(
        f"  Learning rate   : "
        f"{LEARNING_RATE}"
    )

    print()
    print("V4 Regularization:")
    print(
        f"  Statistics      : "
        f"{STATISTICS_WEIGHT}"
    )
    print(
        f"  Correlation     : "
        f"{CORRELATION_WEIGHT}"
    )
    print(
        f"  Temporal        : "
        f"{TEMPORAL_WEIGHT}"
    )
    print(
        f"  Variance        : "
        f"{VARIANCE_WEIGHT}"
    )
    print(
        f"  Tail            : "
        f"{TAIL_WEIGHT}"
    )
    print(
        f"  Quantiles       : "
        f"{TAIL_QUANTILES}"
    )

    print()
    print("V1, V2 and V3 checkpoints will NOT be modified.")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CREATING V4 MODEL COMPONENTS")
    print("=" * 78)

    embedder = Embedder(
        FEATURE_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    recovery = Recovery(
        HIDDEN_DIM,
        FEATURE_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    generator = Generator(
        FEATURE_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    supervisor = Supervisor(
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    discriminator = Discriminator(
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    initialize_weights(embedder)
    initialize_weights(recovery)
    initialize_weights(generator)
    initialize_weights(supervisor)
    initialize_weights(discriminator)

    print("Embedder created.")
    print("Recovery created.")
    print("Generator created.")
    print("Supervisor created.")
    print("Discriminator created.")

    total_parameters = (
        count_parameters(embedder)
        +
        count_parameters(recovery)
        +
        count_parameters(generator)
        +
        count_parameters(supervisor)
        +
        count_parameters(discriminator)
    )

    print()
    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    stage1_history = (
        train_embedder_recovery(
            data,
            embedder,
            recovery,
        )
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    stage2_history = (
        train_supervisor(
            data,
            embedder,
            supervisor,
        )
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    stage3_history = train_joint(
        data=data,
        embedder=embedder,
        recovery=recovery,
        generator=generator,
        supervisor=supervisor,
        discriminator=discriminator,
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        embedder=embedder,
        recovery=recovery,
        generator=generator,
        supervisor=supervisor,
        discriminator=discriminator,
        stage1_history=stage1_history,
        stage2_history=stage2_history,
        stage3_history=stage3_history,
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        stage1_history,
        stage2_history,
        stage3_history,
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("V4 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 78)

    print()
    print("Model:")
    print(
        os.path.abspath(MODEL_PATH)
    )

    print()
    print("History:")
    print(
        os.path.abspath(HISTORY_PATH)
    )

    print()
    print("Final losses:")

    print(
        f"  Stage 1 reconstruction : "
        f"{stage1_history[-1]:.6f}"
    )

    print(
        f"  Stage 2 supervised     : "
        f"{stage2_history[-1]:.6f}"
    )

    final_stage3 = stage3_history[-1]

    print(
        f"  Stage 3 generator      : "
        f"{final_stage3['generator_loss']:.6f}"
    )

    print(
        f"  Stage 3 discriminator   : "
        f"{final_stage3['discriminator_loss']:.6f}"
    )

    print(
        f"  Statistics             : "
        f"{final_stage3['statistics_loss']:.6f}"
    )

    print(
        f"  Correlation            : "
        f"{final_stage3['correlation_loss']:.6f}"
    )

    print(
        f"  Temporal               : "
        f"{final_stage3['temporal_loss']:.6f}"
    )

    print(
        f"  Variance               : "
        f"{final_stage3['variance_loss']:.6f}"
    )

    print(
        f"  Quantile               : "
        f"{final_stage3['quantile_loss']:.6f}"
    )

    print(
        f"  Tail                   : "
        f"{final_stage3['tail_loss']:.6f}"
    )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()