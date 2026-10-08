"""
==============================================================================
MACROSTRESS-GAN
STOCK MARKET TIMEGAN V3
TAIL-AWARE TIMEGAN TRAINING
==============================================================================

Purpose:
    Train a Stock Market-specific TimeGAN V3 using REAL financial data.

V2 showed major improvements in:
    - correlation preservation
    - temporal behavior
    - distribution similarity

V2 still had problems with:
    - extreme-event coverage
    - VIX tails
    - USD/INR tails
    - yield tails
    - crude-oil extreme movements

V3 therefore adds tail-aware statistical losses.

IMPORTANT:
    V1 and V2 checkpoints are NOT modified.

INPUT:
    data/processed/institutions/stock/stock_timegan_sequences.npy

OUTPUT:
    models/timegan/stock/timegan_stock_v3.pt

HISTORY:
    models/timegan/stock/training_history_stock_v3.csv
==============================================================================
"""

import os
import random
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


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

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_sequences.npy"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "timegan_stock_v3.pt"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history_stock_v3.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEQ_LEN = 30
FEATURE_DIM = 5

HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 64

EMBEDDER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 400

LEARNING_RATE = 0.001

SEED = 42

# --------------------------------------------------------------------------
# V3 LOSS WEIGHTS
# --------------------------------------------------------------------------

STATISTICS_WEIGHT = 1.0

CORRELATION_WEIGHT = 2.0

TEMPORAL_WEIGHT = 1.5

TAIL_WEIGHT = 3.0

# Additional emphasis for extreme quantiles.
LOW_TAIL_QUANTILE = 0.05
HIGH_TAIL_QUANTILE = 0.95

# --------------------------------------------------------------------------
# Gradient clipping
# --------------------------------------------------------------------------

GRADIENT_CLIP = 1.0


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# GRU MODULE
# =============================================================================

class GRUNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):

        output, _ = self.gru(x)

        return output


# =============================================================================
# TIMEGAN COMPONENTS
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
# HELPER FUNCTIONS
# =============================================================================

def initialize_weights(model):

    for module in model.modules():

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.GRU
        ):

            for name, parameter in module.named_parameters():

                if "weight_ih" in name:

                    nn.init.xavier_uniform_(
                        parameter
                    )

                elif "weight_hh" in name:

                    nn.init.orthogonal_(
                        parameter
                    )

                elif "bias" in name:

                    nn.init.zeros_(
                        parameter
                    )


def gradient_clip(model):

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRADIENT_CLIP
    )


# =============================================================================
# STATISTICS LOSS
# =============================================================================

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
        dim=(0, 1)
    )

    fake_std = torch.std(
        fake,
        dim=(0, 1)
    )

    mean_loss = torch.mean(
        torch.abs(
            real_mean
            - fake_mean
        )
    )

    std_loss = torch.mean(
        torch.abs(
            real_std
            - fake_std
        )
    )

    return (
        mean_loss
        + std_loss
    )


# =============================================================================
# CORRELATION LOSS
# =============================================================================

def correlation_matrix(
    x
):

    batch_size, seq_len, features = x.shape

    flattened = x.reshape(
        -1,
        features
    )

    centered = (
        flattened
        - flattened.mean(
            dim=0,
            keepdim=True
        )
    )

    covariance = (
        centered.T
        @ centered
    ) / max(
        1,
        centered.shape[0] - 1
    )

    std = torch.sqrt(
        torch.diag(
            covariance
        ).clamp(
            min=1e-6
        )
    )

    denominator = (
        std.unsqueeze(1)
        * std.unsqueeze(0)
    )

    correlation = (
        covariance
        / denominator
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
            real_corr
            - fake_corr
        )
    )


# =============================================================================
# TEMPORAL LOSS
# =============================================================================

def temporal_loss(
    real,
    fake
):

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
            real_diff
            - fake_diff
        )
    )


# =============================================================================
# LAG-1 LOSS
# =============================================================================

def lag1_loss(
    real,
    fake
):

    real_previous = real[
        :, :-1, :
    ]

    real_next = real[
        :, 1:, :
    ]

    fake_previous = fake[
        :, :-1, :
    ]

    fake_next = fake[
        :, 1:, :
    ]

    real_previous = (
        real_previous
        - real_previous.mean(
            dim=1,
            keepdim=True
        )
    )

    real_next = (
        real_next
        - real_next.mean(
            dim=1,
            keepdim=True
        )
    )

    fake_previous = (
        fake_previous
        - fake_previous.mean(
            dim=1,
            keepdim=True
        )
    )

    fake_next = (
        fake_next
        - fake_next.mean(
            dim=1,
            keepdim=True
        )
    )

    real_cov = torch.mean(
        real_previous
        * real_next,
        dim=1
    )

    fake_cov = torch.mean(
        fake_previous
        * fake_next,
        dim=1
    )

    real_std_1 = torch.sqrt(
        torch.mean(
            real_previous ** 2,
            dim=1
        ).clamp(
            min=1e-6
        )
    )

    real_std_2 = torch.sqrt(
        torch.mean(
            real_next ** 2,
            dim=1
        ).clamp(
            min=1e-6
        )
    )

    fake_std_1 = torch.sqrt(
        torch.mean(
            fake_previous ** 2,
            dim=1
        ).clamp(
            min=1e-6
        )
    )

    fake_std_2 = torch.sqrt(
        torch.mean(
            fake_next ** 2,
            dim=1
        ).clamp(
            min=1e-6
        )
    )

    real_corr = (
        real_cov
        / (
            real_std_1
            * real_std_2
            + 1e-6
        )
    )

    fake_corr = (
        fake_cov
        / (
            fake_std_1
            * fake_std_2
            + 1e-6
        )
    )

    return torch.mean(
        torch.abs(
            real_corr
            - fake_corr
        )
    )


# =============================================================================
# TAIL LOSS
# =============================================================================

def tail_quantile_loss(
    real,
    fake
):

    """
    Compare lower and upper quantiles.

    This is performed independently for every feature.

    The objective is NOT to force synthetic data to reproduce every
    individual extreme observation. Instead, it encourages the synthetic
    distribution to retain the shape of the real distribution's tails.
    """

    real_flat = real.reshape(
        -1,
        real.shape[-1]
    )

    fake_flat = fake.reshape(
        -1,
        fake.shape[-1]
    )

    lower_real = torch.quantile(
        real_flat,
        LOW_TAIL_QUANTILE,
        dim=0
    )

    lower_fake = torch.quantile(
        fake_flat,
        LOW_TAIL_QUANTILE,
        dim=0
    )

    upper_real = torch.quantile(
        real_flat,
        HIGH_TAIL_QUANTILE,
        dim=0
    )

    upper_fake = torch.quantile(
        fake_flat,
        HIGH_TAIL_QUANTILE,
        dim=0
    )

    lower_loss = torch.mean(
        torch.abs(
            lower_real
            - lower_fake
        )
    )

    upper_loss = torch.mean(
        torch.abs(
            upper_real
            - upper_fake
        )
    )

    return (
        lower_loss
        + upper_loss
    )


# =============================================================================
# EXTREME WEIGHTED LOSS
# =============================================================================

def extreme_weighted_loss(
    real,
    fake
):

    """
    Give additional importance to observations near the tails.

    Since TimeGAN works in [0,1] scaled space, the extreme regions are
    approximately:

        <= 0.05
        >= 0.95

    The weighting is soft rather than a hard threshold.
    """

    lower_real = torch.relu(
        0.10 - real
    )

    upper_real = torch.relu(
        real - 0.90
    )

    lower_fake = torch.relu(
        0.10 - fake
    )

    upper_fake = torch.relu(
        fake - 0.90
    )

    real_tail_energy = (
        lower_real.mean(
            dim=(0, 1)
        )
        +
        upper_real.mean(
            dim=(0, 1)
        )
    )

    fake_tail_energy = (
        lower_fake.mean(
            dim=(0, 1)
        )
        +
        upper_fake.mean(
            dim=(0, 1)
        )
    )

    return torch.mean(
        torch.abs(
            real_tail_energy
            - fake_tail_energy
        )
    )


# =============================================================================
# FULL V3 TAIL LOSS
# =============================================================================

def tail_loss(
    real,
    fake
):

    quantile_component = tail_quantile_loss(
        real,
        fake
    )

    extreme_component = extreme_weighted_loss(
        real,
        fake
    )

    return (
        quantile_component
        + extreme_component
    )


# =============================================================================
# SAMPLE REAL BATCH
# =============================================================================

def get_real_batch(
    data,
    batch_size
):

    indices = np.random.choice(
        len(data),
        size=batch_size,
        replace=False
    )

    batch = torch.tensor(
        data[indices],
        dtype=torch.float32,
        device=DEVICE
    )

    return batch


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(
        SEED
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V3")
    print("TAIL-AWARE TRAINING")
    print("=" * 78)

    print()
    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Sequence length : {SEQ_LEN}"
    )

    print(
        f"Feature dim     : {FEATURE_DIM}"
    )

    print(
        f"Hidden dim      : {HIDDEN_DIM}"
    )

    print(
        f"GRU layers      : {NUM_LAYERS}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Stage 1 epochs  : {EMBEDDER_EPOCHS}"
    )

    print(
        f"Stage 2 epochs  : {SUPERVISOR_EPOCHS}"
    )

    print(
        f"Stage 3 epochs  : {JOINT_EPOCHS}"
    )

    print()
    print("V3 Loss weights:")
    print(
        f"  Statistics   : {STATISTICS_WEIGHT}"
    )
    print(
        f"  Correlation  : {CORRELATION_WEIGHT}"
    )
    print(
        f"  Temporal     : {TEMPORAL_WEIGHT}"
    )
    print(
        f"  Tail         : {TAIL_WEIGHT}"
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print()
    print("=" * 78)
    print("LOADING TRAINING DATA")
    print("=" * 78)

    print()
    print(
        f"Data path:\n{DATA_PATH}"
    )

    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(
            f"Training data not found:\n{DATA_PATH}"
        )

    data = np.load(
        DATA_PATH
    ).astype(
        np.float32
    )

    print()
    print(
        f"Dataset shape : {data.shape}"
    )

    print(
        f"Min           : {data.min():.8f}"
    )

    print(
        f"Max           : {data.max():.8f}"
    )

    print(
        f"NaN count     : {np.isnan(data).sum()}"
    )

    print(
        f"Inf count     : {np.isinf(data).sum()}"
    )

    if data.ndim != 3:

        raise ValueError(
            "Expected 3D sequence data."
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

    # =========================================================================
    # CREATE MODELS
    # =========================================================================

    print()
    print("=" * 78)
    print("CREATING TIMEGAN V3 COMPONENTS")
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

    initialize_weights(
        embedder
    )

    initialize_weights(
        recovery
    )

    initialize_weights(
        generator
    )

    initialize_weights(
        supervisor
    )

    initialize_weights(
        discriminator
    )

    total_parameters = sum(
        parameter.numel()
        for model in [
            embedder,
            recovery,
            generator,
            supervisor,
            discriminator
        ]
        for parameter in model.parameters()
    )

    print()
    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    # =========================================================================
    # OPTIMIZERS
    # =========================================================================

    optimizer_er = optim.Adam(
        list(
            embedder.parameters()
        )
        +
        list(
            recovery.parameters()
        ),
        lr=LEARNING_RATE
    )

    optimizer_s = optim.Adam(
        supervisor.parameters(),
        lr=LEARNING_RATE
    )

    optimizer_g = optim.Adam(
        list(
            generator.parameters()
        )
        +
        list(
            supervisor.parameters()
        ),
        lr=LEARNING_RATE
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE
    )

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss()

    history = []

    # =========================================================================
    # STAGE 1 — EMBEDDER / RECOVERY
    # =========================================================================

    print()
    print("=" * 78)
    print("STAGE 1 — EMBEDDER + RECOVERY")
    print("=" * 78)

    for epoch in range(
        1,
        EMBEDDER_EPOCHS + 1
    ):

        real_batch = get_real_batch(
            data,
            BATCH_SIZE
        )

        optimizer_er.zero_grad()

        h = embedder(
            real_batch
        )

        reconstructed = recovery(
            h
        )

        reconstruction_loss = mse_loss(
            reconstructed,
            real_batch
        )

        reconstruction_loss.backward()

        gradient_clip(
            embedder
        )

        gradient_clip(
            recovery
        )

        optimizer_er.step()

        history.append({

            "stage":
                "embedder_recovery",

            "epoch":
                epoch,

            "reconstruction_loss":
                reconstruction_loss.item(),

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

            "tail_loss":
                np.nan
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EMBEDDER_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EMBEDDER_EPOCHS} "
                f"| Reconstruction: "
                f"{reconstruction_loss.item():.6f}"
            )

    # =========================================================================
    # STAGE 2 — SUPERVISOR
    # =========================================================================

    print()
    print("=" * 78)
    print("STAGE 2 — SUPERVISOR")
    print("=" * 78)

    for epoch in range(
        1,
        SUPERVISOR_EPOCHS + 1
    ):

        real_batch = get_real_batch(
            data,
            BATCH_SIZE
        )

        optimizer_s.zero_grad()

        with torch.no_grad():

            h = embedder(
                real_batch
            )

        supervised_target = h[
            :,
            1:,
            :
        ]

        supervisor_input = h[
            :,
            :-1,
            :
        ]

        supervised_output = supervisor(
            supervisor_input
        )

        supervised_loss = mse_loss(
            supervised_output,
            supervised_target
        )

        supervised_loss.backward()

        gradient_clip(
            supervisor
        )

        optimizer_s.step()

        history.append({

            "stage":
                "supervisor",

            "epoch":
                epoch,

            "reconstruction_loss":
                np.nan,

            "supervised_loss":
                supervised_loss.item(),

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

            "tail_loss":
                np.nan
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == SUPERVISOR_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{SUPERVISOR_EPOCHS} "
                f"| Supervised: "
                f"{supervised_loss.item():.6f}"
            )

    # =========================================================================
    # STAGE 3 — JOINT ADVERSARIAL TRAINING
    # =========================================================================

    print()
    print("=" * 78)
    print("STAGE 3 — JOINT TAIL-AWARE GAN TRAINING")
    print("=" * 78)

    for epoch in range(
        1,
        JOINT_EPOCHS + 1
    ):

        # =====================================================================
        # REAL BATCH
        # =====================================================================

        real_batch = get_real_batch(
            data,
            BATCH_SIZE
        )

        # =====================================================================
        # GENERATOR UPDATE
        # =====================================================================

        optimizer_g.zero_grad()

        noise = torch.rand(
            BATCH_SIZE,
            SEQ_LEN,
            FEATURE_DIM,
            device=DEVICE
        )

        generated_hidden = generator(
            noise
        )

        supervised_generated = supervisor(
            generated_hidden
        )

        synthetic = recovery(
            supervised_generated
        )

        # ---------------------------------------------------------------------
        # ADVERSARIAL LOSS
        # ---------------------------------------------------------------------

        fake_logits = discriminator(
            supervised_generated
        )

        adversarial_loss = bce_loss(
            fake_logits,
            torch.ones_like(
                fake_logits
            )
        )

        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        stat_loss = statistics_loss(
            real_batch,
            synthetic
        )

        # ---------------------------------------------------------------------
        # CORRELATION
        # ---------------------------------------------------------------------

        corr_loss = correlation_loss(
            real_batch,
            synthetic
        )

        # ---------------------------------------------------------------------
        # TEMPORAL
        # ---------------------------------------------------------------------

        temp_loss = temporal_loss(
            real_batch,
            synthetic
        )

        # ---------------------------------------------------------------------
        # LAG-1
        # ---------------------------------------------------------------------

        temporal_corr_loss = lag1_loss(
            real_batch,
            synthetic
        )

        # ---------------------------------------------------------------------
        # TAIL
        # ---------------------------------------------------------------------

        tail_component = tail_loss(
            real_batch,
            synthetic
        )

        # ---------------------------------------------------------------------
        # COMBINED V3 LOSS
        # ---------------------------------------------------------------------

        generator_loss = (

            adversarial_loss

            + STATISTICS_WEIGHT
            * stat_loss

            + CORRELATION_WEIGHT
            * corr_loss

            + TEMPORAL_WEIGHT
            * (
                temp_loss
                + temporal_corr_loss
            )

            + TAIL_WEIGHT
            * tail_component
        )

        generator_loss.backward()

        gradient_clip(
            generator
        )

        gradient_clip(
            supervisor
        )

        optimizer_g.step()

        # =====================================================================
        # DISCRIMINATOR UPDATE
        # =====================================================================

        optimizer_d.zero_grad()

        # Fresh real forward pass
        real_hidden = embedder(
            real_batch
        )

        real_supervised = supervisor(
            real_hidden
        )

        real_logits = discriminator(
            real_supervised.detach()
        )

        real_labels = torch.ones_like(
            real_logits
        )

        real_loss = bce_loss(
            real_logits,
            real_labels
        )

        # Fresh fake forward pass
        noise_d = torch.rand(
            BATCH_SIZE,
            SEQ_LEN,
            FEATURE_DIM,
            device=DEVICE
        )

        fake_hidden_d = generator(
            noise_d
        )

        fake_supervised_d = supervisor(
            fake_hidden_d
        )

        fake_logits_d = discriminator(
            fake_supervised_d.detach()
        )

        fake_labels = torch.zeros_like(
            fake_logits_d
        )

        fake_loss = bce_loss(
            fake_logits_d,
            fake_labels
        )

        discriminator_loss = (
            real_loss
            + fake_loss
        )

        discriminator_loss.backward()

        gradient_clip(
            discriminator
        )

        optimizer_d.step()

        # =====================================================================
        # SAVE HISTORY
        # =====================================================================

        history.append({

            "stage":
                "joint",

            "epoch":
                epoch,

            "reconstruction_loss":
                np.nan,

            "supervised_loss":
                np.nan,

            "generator_loss":
                generator_loss.item(),

            "discriminator_loss":
                discriminator_loss.item(),

            "statistics_loss":
                stat_loss.item(),

            "correlation_loss":
                corr_loss.item(),

            "temporal_loss":
                (
                    temp_loss.item()
                    +
                    temporal_corr_loss.item()
                ),

            "tail_loss":
                tail_component.item()
        })

        # =====================================================================
        # LOGGING
        # =====================================================================

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == JOINT_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{JOINT_EPOCHS} "
                f"| G={generator_loss.item():.6f} "
                f"| D={discriminator_loss.item():.6f} "
                f"| Stat={stat_loss.item():.6f} "
                f"| Corr={corr_loss.item():.6f} "
                f"| Temp={temp_loss.item() + temporal_corr_loss.item():.6f} "
                f"| Tail={tail_component.item():.6f}"
            )

    # =========================================================================
    # SAVE TRAINING HISTORY
    # =========================================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    print()
    print(
        f"Training history saved:\n"
        f"{HISTORY_PATH}"
    )

    # =========================================================================
    # MODEL CHECKPOINT
    # =========================================================================

    checkpoint = {

        "model_version":
            "TimeGAN_Stock_V3",

        "institution":
            "Stock Market",

        "data_type":
            "REAL",

        "feature_names":
            [
                "NIFTY50_Return",
                "CRUDE_OIL_Return",
                "USD_INR_Return",
                "INDIA_VIX_Change",
                "INDIA_10Y_YIELD_Change"
            ],

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

        "statistics_weight":
            STATISTICS_WEIGHT,

        "correlation_weight":
            CORRELATION_WEIGHT,

        "temporal_weight":
            TEMPORAL_WEIGHT,

        "tail_weight":
            TAIL_WEIGHT,

        "low_tail_quantile":
            LOW_TAIL_QUANTILE,

        "high_tail_quantile":
            HIGH_TAIL_QUANTILE,

        "seed":
            SEED,

        "embedder":
            embedder.state_dict(),

        "recovery":
            recovery.state_dict(),

        "generator":
            generator.state_dict(),

        "supervisor":
            supervisor.state_dict(),

        "discriminator":
            discriminator.state_dict()
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print()
    print("=" * 78)
    print("TIMEGAN V3 MODEL SAVED")
    print("=" * 78)

    print()
    print(
        MODEL_PATH
    )

    print()
    print("=" * 78)
    print("TRAINING COMPLETED")
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()