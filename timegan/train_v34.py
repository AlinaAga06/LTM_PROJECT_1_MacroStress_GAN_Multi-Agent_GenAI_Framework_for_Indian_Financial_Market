"""
MacroStress-GAN V3.4
====================

Regime + Stress Conditioned TimeGAN

Purpose
-------
Generate synthetic Indian financial return/change sequences conditioned on
historical market regime and continuous stress context.

Financial features:
    1. NIFTY50_Return
    2. CRUDE_OIL_Return
    3. USD_INR_Return
    4. INDIA_VIX_Change
    5. INDIA_10Y_YIELD_Change

Conditioning:
    4 categorical regime dimensions
        - Normal
        - Elevated Volatility
        - Market Stress
        - Crisis

    5 continuous stress dimensions
        - Mean_Stress_Score
        - Mean_Volatility_Score
        - Mean_Movement_Score
        - Mean_NIFTY_Drawdown
        - Mean_VIX_Stress

Total conditioning dimensions:
    9

Important:
    This version intentionally does NOT use the V3.3 global extreme losses.
    Extreme behavior is learned through conditional adversarial training and
    the V3.2 tail-aware statistical losses.

Preprocessing:
    QuantileTransformer-based V3.2/V3.3/V3.4 pipeline.

Architecture:
    Embedder:      financial sequence -> latent sequence
    Recovery:      latent sequence -> financial sequence

    Generator:     noise + condition -> latent sequence
    Supervisor:    latent sequence + condition -> latent sequence
    Discriminator: latent sequence + condition -> real/fake probability

Outputs:
    models/timegan/timegan_v34_model.pt
    models/timegan/training_history_v34.csv
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v34"
    / "timegan_v34_sequences.npy"
)

CONDITION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v34"
    / "conditioning_vector.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v34"
    / "conditioning_metadata.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "timegan_v34_model.pt"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "training_history_v34.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQ_LEN = 30
N_FEATURES = 5
COND_DIM = 9
HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 64

STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 100
STAGE3_EPOCHS = 300

LEARNING_RATE = 0.001

# ------------------------------------------------------------
# V3.2 statistical losses
# ------------------------------------------------------------

LAMBDA_ADV = 1.0
LAMBDA_SUPERVISED = 100.0
LAMBDA_MOMENT = 10.0
LAMBDA_CORRELATION = 25.0
LAMBDA_TEMPORAL = 25.0

# ------------------------------------------------------------
# V3.2 tail-aware losses
# ------------------------------------------------------------

LAMBDA_QUANTILE = 25.0
LAMBDA_MAGNITUDE = 10.0
LAMBDA_TAIL = 10.0

TAIL_QUANTILES = [0.01, 0.05, 0.95, 0.99]
TAIL_THRESHOLDS = [1.5, 2.0]

# ------------------------------------------------------------
# Numerical stability
# ------------------------------------------------------------

EPS = 1e-6


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class Embedder(nn.Module):
    """
    Maps the five-dimensional financial sequence into latent space.
    """

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

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.activation(h)

        return h


class Recovery(nn.Module):
    """
    Maps latent sequence back to five financial features.
    """

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

        self.linear = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.activation = nn.Sigmoid()

    def forward(self, h):

        x, _ = self.gru(h)

        x = self.linear(x)

        x = self.activation(x)

        return x


class Generator(nn.Module):
    """
    Conditional generator.

    Input:
        random noise + 9-dimensional conditioning vector

    Output:
        latent sequence
    """

    def __init__(
        self,
        noise_dim,
        cond_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=noise_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        z,
        condition,
    ):

        condition_seq = condition.unsqueeze(1).repeat(
            1,
            z.size(1),
            1,
        )

        generator_input = torch.cat(
            [z, condition_seq],
            dim=2,
        )

        h, _ = self.gru(generator_input)

        h = self.linear(h)

        h = self.activation(h)

        return h


class Supervisor(nn.Module):
    """
    Conditional temporal supervisor.

    Learns next-step latent dynamics while observing
    the regime/stress context.
    """

    def __init__(
        self,
        hidden_dim,
        cond_dim,
        num_layers,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        h,
        condition,
    ):

        condition_seq = condition.unsqueeze(1).repeat(
            1,
            h.size(1),
            1,
        )

        supervisor_input = torch.cat(
            [h, condition_seq],
            dim=2,
        )

        s, _ = self.gru(supervisor_input)

        s = self.linear(s)

        s = self.activation(s)

        return s


class Discriminator(nn.Module):
    """
    Conditional discriminator.

    Determines whether a latent financial sequence is real or synthetic
    while observing the associated regime/stress context.
    """

    def __init__(
        self,
        hidden_dim,
        cond_dim,
        num_layers,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            1,
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.2
        )

    def forward(
        self,
        h,
        condition,
    ):

        condition_seq = condition.unsqueeze(1).repeat(
            1,
            h.size(1),
            1,
        )

        discriminator_input = torch.cat(
            [h, condition_seq],
            dim=2,
        )

        y, _ = self.gru(discriminator_input)

        y = y[:, -1, :]

        y = self.linear1(y)

        y = self.activation(y)

        y = self.linear2(y)

        return y


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def initialize_weights(model):

    for module in model.modules():

        if isinstance(module, nn.Linear):

            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.GRU):

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


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def batch_iterator(
    x,
    condition,
    batch_size,
):

    n = x.size(0)

    indices = torch.randperm(
        n,
        device=x.device,
    )

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        batch_indices = indices[
            start:end
        ]

        yield (
            x[batch_indices],
            condition[batch_indices],
        )


def random_noise(
    batch_size,
    seq_len,
    hidden_dim,
):

    return torch.rand(
        batch_size,
        seq_len,
        hidden_dim,
        device=DEVICE,
    )


# ============================================================
# STATISTICAL LOSSES
# ============================================================

def calculate_moment_loss(
    real,
    fake,
):

    real_mean = real.mean(
        dim=(0, 1)
    )

    fake_mean = fake.mean(
        dim=(0, 1)
    )

    mean_loss = torch.mean(
        torch.abs(
            real_mean - fake_mean
        )
    )

    real_std = torch.sqrt(
        real.var(
            dim=(0, 1),
            unbiased=False,
        )
        + EPS
    )

    fake_std = torch.sqrt(
        fake.var(
            dim=(0, 1),
            unbiased=False,
        )
        + EPS
    )

    std_loss = torch.mean(
        torch.abs(
            real_std - fake_std
        )
    )

    return mean_loss + std_loss


def calculate_correlation_loss(
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

    real_centered = (
        real_flat
        - real_flat.mean(dim=0)
    )

    fake_centered = (
        fake_flat
        - fake_flat.mean(dim=0)
    )

    real_cov = (
        real_centered.T
        @ real_centered
    ) / (
        real_centered.shape[0] - 1
    )

    fake_cov = (
        fake_centered.T
        @ fake_centered
    ) / (
        fake_centered.shape[0] - 1
    )

    real_std = torch.sqrt(
        torch.diag(real_cov)
        + EPS
    )

    fake_std = torch.sqrt(
        torch.diag(fake_cov)
        + EPS
    )

    real_corr = (
        real_cov
        / (
            real_std.unsqueeze(1)
            * real_std.unsqueeze(0)
            + EPS
        )
    )

    fake_corr = (
        fake_cov
        / (
            fake_std.unsqueeze(1)
            * fake_std.unsqueeze(0)
            + EPS
        )
    )

    return torch.mean(
        torch.abs(
            real_corr - fake_corr
        )
    )


def calculate_temporal_loss(
    real,
    fake,
):

    if real.size(1) < 2:
        return torch.tensor(
            0.0,
            device=real.device,
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


def calculate_quantile_loss(
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

    losses = []

    for q in TAIL_QUANTILES:

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

        losses.append(
            torch.mean(
                torch.abs(
                    real_q - fake_q
                )
            )
        )

    return torch.stack(
        losses
    ).mean()


def calculate_magnitude_loss(
    real,
    fake,
):

    real_abs = torch.abs(real)

    fake_abs = torch.abs(fake)

    real_q = torch.quantile(
        real_abs.reshape(
            -1,
            real_abs.shape[-1],
        ),
        0.95,
        dim=0,
    )

    fake_q = torch.quantile(
        fake_abs.reshape(
            -1,
            fake_abs.shape[-1],
        ),
        0.95,
        dim=0,
    )

    return torch.mean(
        torch.abs(
            real_q - fake_q
        )
    )


def calculate_tail_loss(
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

    real_mean = real_flat.mean(
        dim=0
    )

    real_std = torch.sqrt(
        real_flat.var(
            dim=0,
            unbiased=False,
        )
        + EPS
    )

    real_z = (
        real_flat
        - real_mean
    ) / real_std

    fake_z = (
        fake_flat
        - real_mean
    ) / real_std

    losses = []

    for threshold in TAIL_THRESHOLDS:

        real_tail = torch.sigmoid(
            (
                torch.abs(real_z)
                - threshold
            ) / 0.15
        )

        fake_tail = torch.sigmoid(
            (
                torch.abs(fake_z)
                - threshold
            ) / 0.15
        )

        real_rate = real_tail.mean(
            dim=0
        )

        fake_rate = fake_tail.mean(
            dim=0
        )

        losses.append(
            torch.mean(
                torch.abs(
                    real_rate
                    - fake_rate
                )
            )
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# STAGE 1
# ============================================================

def train_embedder_recovery(
    embedder,
    recovery,
    train_data,
):

    print()
    print("=" * 80)
    print("STAGE 1 - EMBEDDER / RECOVERY TRAINING")
    print("=" * 80)

    optimizer = optim.Adam(
        list(embedder.parameters())
        + list(recovery.parameters()),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(
        1,
        STAGE1_EPOCHS + 1,
    ):

        epoch_loss = 0.0
        batches = 0

        for real_batch, _ in batch_iterator(
            train_data,
            torch.zeros(
                train_data.size(0),
                COND_DIM,
                device=DEVICE,
            ),
            BATCH_SIZE,
        ):

            optimizer.zero_grad()

            h = embedder(
                real_batch
            )

            reconstructed = recovery(
                h
            )

            reconstruction_loss = torch.mean(
                torch.abs(
                    real_batch
                    - reconstructed
                )
            )

            reconstruction_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(embedder.parameters())
                + list(recovery.parameters()),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_loss += (
                reconstruction_loss.item()
            )

            batches += 1

        epoch_loss /= max(
            batches,
            1,
        )

        history.append(
            epoch_loss
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == STAGE1_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d}/{STAGE1_EPOCHS} | "
                f"Reconstruction: {epoch_loss:.6f}"
            )

    return history


# ============================================================
# STAGE 2
# ============================================================

def train_supervisor(
    embedder,
    supervisor,
    train_data,
    conditions,
):

    print()
    print("=" * 80)
    print("STAGE 2 - CONDITIONAL SUPERVISOR TRAINING")
    print("=" * 80)

    optimizer = optim.Adam(
        supervisor.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(
        1,
        STAGE2_EPOCHS + 1,
    ):

        epoch_loss = 0.0
        batches = 0

        for real_batch, cond_batch in batch_iterator(
            train_data,
            conditions,
            BATCH_SIZE,
        ):

            optimizer.zero_grad()

            with torch.no_grad():

                h = embedder(
                    real_batch
                )

            supervised_output = supervisor(
                h,
                cond_batch,
            )

            if h.size(1) > 1:

                supervised_loss = torch.mean(
                    torch.abs(
                        supervised_output[:, :-1, :]
                        - h[:, 1:, :]
                    )
                )

            else:

                supervised_loss = torch.mean(
                    torch.abs(
                        supervised_output
                        - h
                    )
                )

            supervised_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                supervisor.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_loss += (
                supervised_loss.item()
            )

            batches += 1

        epoch_loss /= max(
            batches,
            1,
        )

        history.append(
            epoch_loss
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == STAGE2_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d}/{STAGE2_EPOCHS} | "
                f"Supervised: {epoch_loss:.6f}"
            )

    return history


# ============================================================
# STAGE 3
# ============================================================

def train_joint(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
    train_data,
    conditions,
):

    print()
    print("=" * 80)
    print("STAGE 3 - CONDITIONAL JOINT TRAINING")
    print("=" * 80)

    print()
    print("Loss weights:")
    print(
        f"  Adversarial:   {LAMBDA_ADV}"
    )
    print(
        f"  Supervised:    {LAMBDA_SUPERVISED}"
    )
    print(
        f"  Moment:        {LAMBDA_MOMENT}"
    )
    print(
        f"  Correlation:   {LAMBDA_CORRELATION}"
    )
    print(
        f"  Temporal:      {LAMBDA_TEMPORAL}"
    )
    print(
        f"  Quantile:      {LAMBDA_QUANTILE}"
    )
    print(
        f"  Magnitude:     {LAMBDA_MAGNITUDE}"
    )
    print(
        f"  Tail:          {LAMBDA_TAIL}"
    )

    generator_optimizer = optim.Adam(
        list(generator.parameters())
        + list(supervisor.parameters()),
        lr=LEARNING_RATE,
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    embedder_optimizer = optim.Adam(
        embedder.parameters(),
        lr=LEARNING_RATE,
    )

    recovery_optimizer = optim.Adam(
        recovery.parameters(),
        lr=LEARNING_RATE,
    )

    adversarial_loss = nn.BCEWithLogitsLoss()

    history = []

    for epoch in range(
        1,
        STAGE3_EPOCHS + 1,
    ):

        epoch_metrics = {
            "G": 0.0,
            "D": 0.0,
            "E": 0.0,
            "Corr": 0.0,
            "Temp": 0.0,
            "Quant": 0.0,
            "Mag": 0.0,
            "Tail": 0.0,
            "Sup": 0.0,
            "Adv": 0.0,
        }

        batches = 0

        for real_batch, cond_batch in batch_iterator(
            train_data,
            conditions,
            BATCH_SIZE,
        ):

            batch_size = real_batch.size(0)

            # ==================================================
            # REAL LATENT REPRESENTATION
            # ==================================================

            real_h = embedder(
                real_batch
            )

            # ==================================================
            # GENERATOR
            # ==================================================

            z = random_noise(
                batch_size,
                SEQ_LEN,
                HIDDEN_DIM,
            )

            generated_h = generator(
                z,
                cond_batch,
            )

            supervised_h = supervisor(
                generated_h,
                cond_batch,
            )

            fake_data = recovery(
                supervised_h
            )

            # ==================================================
            # GENERATOR / SUPERVISOR UPDATE
            # ==================================================

            generator_optimizer.zero_grad()

            real_for_generator = real_batch.detach()

            # --------------------------------------------------
            # Adversarial
            # --------------------------------------------------

            fake_logits = discriminator(
                supervised_h,
                cond_batch,
            )

            real_labels = torch.ones_like(
                fake_logits
            )

            adversarial = adversarial_loss(
                fake_logits,
                real_labels,
            )

            # --------------------------------------------------
            # Supervised temporal loss
            # --------------------------------------------------

            if real_h.size(1) > 1:

                supervised = torch.mean(
                    torch.abs(
                        supervised_h[:, :-1, :]
                        - real_h[:, 1:, :].detach()
                    )
                )

            else:

                supervised = torch.mean(
                    torch.abs(
                        supervised_h
                        - real_h.detach()
                    )
                )

            # --------------------------------------------------
            # Statistical losses
            # --------------------------------------------------

            moment = calculate_moment_loss(
                real_for_generator,
                fake_data,
            )

            correlation = calculate_correlation_loss(
                real_for_generator,
                fake_data,
            )

            temporal = calculate_temporal_loss(
                real_for_generator,
                fake_data,
            )

            quantile = calculate_quantile_loss(
                real_for_generator,
                fake_data,
            )

            magnitude = calculate_magnitude_loss(
                real_for_generator,
                fake_data,
            )

            tail = calculate_tail_loss(
                real_for_generator,
                fake_data,
            )

            # --------------------------------------------------
            # Total generator loss
            # --------------------------------------------------

            generator_loss = (
                LAMBDA_ADV
                * adversarial
                +
                LAMBDA_SUPERVISED
                * supervised
                +
                LAMBDA_MOMENT
                * moment
                +
                LAMBDA_CORRELATION
                * correlation
                +
                LAMBDA_TEMPORAL
                * temporal
                +
                LAMBDA_QUANTILE
                * quantile
                +
                LAMBDA_MAGNITUDE
                * magnitude
                +
                LAMBDA_TAIL
                * tail
            )

            generator_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters())
                + list(supervisor.parameters()),
                max_norm=5.0,
            )

            generator_optimizer.step()

            # ==================================================
            # DISCRIMINATOR UPDATE
            # ==================================================

            discriminator_optimizer.zero_grad()

            # Real
            real_logits = discriminator(
                real_h.detach(),
                cond_batch,
            )

            real_labels = torch.ones_like(
                real_logits
            )

            real_loss = adversarial_loss(
                real_logits,
                real_labels,
            )

            # Fake
            fake_logits = discriminator(
                supervised_h.detach(),
                cond_batch,
            )

            fake_labels = torch.zeros_like(
                fake_logits
            )

            fake_loss = adversarial_loss(
                fake_logits,
                fake_labels,
            )

            discriminator_loss = (
                0.5
                * (
                    real_loss
                    + fake_loss
                )
            )

            discriminator_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0,
            )

            discriminator_optimizer.step()

            # ==================================================
            # EMBEDDER / RECOVERY UPDATE
            # ==================================================

            embedder_optimizer.zero_grad()
            recovery_optimizer.zero_grad()

            h_reconstruction = embedder(
                real_batch
            )

            reconstructed = recovery(
                h_reconstruction
            )

            reconstruction_loss = torch.mean(
                torch.abs(
                    real_batch
                    - reconstructed
                )
            )

            reconstruction_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                embedder.parameters(),
                max_norm=5.0,
            )

            torch.nn.utils.clip_grad_norm_(
                recovery.parameters(),
                max_norm=5.0,
            )

            embedder_optimizer.step()
            recovery_optimizer.step()

            # ==================================================
            # RECORD
            # ==================================================

            epoch_metrics["G"] += (
                generator_loss.item()
            )

            epoch_metrics["D"] += (
                discriminator_loss.item()
            )

            epoch_metrics["E"] += (
                reconstruction_loss.item()
            )

            epoch_metrics["Corr"] += (
                correlation.item()
            )

            epoch_metrics["Temp"] += (
                temporal.item()
            )

            epoch_metrics["Quant"] += (
                quantile.item()
            )

            epoch_metrics["Mag"] += (
                magnitude.item()
            )

            epoch_metrics["Tail"] += (
                tail.item()
            )

            epoch_metrics["Sup"] += (
                supervised.item()
            )

            epoch_metrics["Adv"] += (
                adversarial.item()
            )

            batches += 1

        # ======================================================
        # AVERAGE
        # ======================================================

        for key in epoch_metrics:

            epoch_metrics[key] /= max(
                batches,
                1,
            )

        epoch_record = {
            "epoch": epoch,
            **epoch_metrics,
        }

        history.append(
            epoch_record
        )

        # ======================================================
        # PRINT
        # ======================================================

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == STAGE3_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d}/{STAGE3_EPOCHS} | "
                f"G {epoch_metrics['G']:.6f} | "
                f"D {epoch_metrics['D']:.6f} | "
                f"E {epoch_metrics['E']:.6f} | "
                f"Corr {epoch_metrics['Corr']:.6f} | "
                f"Temp {epoch_metrics['Temp']:.6f} | "
                f"Quant {epoch_metrics['Quant']:.6f} | "
                f"Mag {epoch_metrics['Mag']:.6f} | "
                f"Tail {epoch_metrics['Tail']:.6f}"
            )

    return history


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
    history,
):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_version": "TimeGAN_V3.4",
        "description": (
            "Regime and stress conditioned TimeGAN for "
            "Indian financial return/change sequences "
            "using QuantileTransformer preprocessing."
        ),
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "cond_dim": COND_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
        "stage3_epochs": STAGE3_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lambda_adv": LAMBDA_ADV,
        "lambda_supervised": LAMBDA_SUPERVISED,
        "lambda_moment": LAMBDA_MOMENT,
        "lambda_correlation": LAMBDA_CORRELATION,
        "lambda_temporal": LAMBDA_TEMPORAL,
        "lambda_quantile": LAMBDA_QUANTILE,
        "lambda_magnitude": LAMBDA_MAGNITUDE,
        "lambda_tail": LAMBDA_TAIL,
        "conditioning_features": [
            "Regime_Normal",
            "Regime_Elevated_Volatility",
            "Regime_Market_Stress",
            "Regime_Crisis",
            "Mean_Stress_Score",
            "Mean_Volatility_Score",
            "Mean_Movement_Score",
            "Mean_NIFTY_Drawdown",
            "Mean_VIX_Stress",
        ],
        "embedder_state_dict": (
            embedder.state_dict()
        ),
        "recovery_state_dict": (
            recovery.state_dict()
        ),
        "generator_state_dict": (
            generator.state_dict()
        ),
        "supervisor_state_dict": (
            supervisor.state_dict()
        ),
        "discriminator_state_dict": (
            discriminator.state_dict()
        ),
        "history": history,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to:\n{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 80)
    print("MacroStress-GAN V3.4")
    print("REGIME + STRESS CONDITIONED TIMEGAN")
    print("=" * 80)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device:       {DEVICE}")
    print(f"Seed:         {SEED}")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("=" * 80)
    print("1. LOADING V3.4 DATA")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sequence data not found:\n{DATA_PATH}"
        )

    if not CONDITION_PATH.exists():
        raise FileNotFoundError(
            f"Conditioning data not found:\n{CONDITION_PATH}"
        )

    sequences = np.load(
        DATA_PATH
    ).astype(np.float32)

    conditions = np.load(
        CONDITION_PATH
    ).astype(np.float32)

    print(
        f"Financial sequences: {sequences.shape}"
    )

    print(
        f"Conditioning vectors: {conditions.shape}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    print()
    print("=" * 80)
    print("2. VALIDATING INPUT DATA")
    print("=" * 80)

    if sequences.shape[1:] != (
        SEQ_LEN,
        N_FEATURES,
    ):
        raise ValueError(
            f"Expected sequence shape "
            f"(N,{SEQ_LEN},{N_FEATURES}), "
            f"got {sequences.shape}"
        )

    if conditions.shape[1] != COND_DIM:
        raise ValueError(
            f"Expected conditioning dimension "
            f"{COND_DIM}, got {conditions.shape}"
        )

    if len(sequences) != len(conditions):
        raise ValueError(
            "Sequence/conditioning count mismatch:\n"
            f"Sequences: {len(sequences)}\n"
            f"Conditions: {len(conditions)}"
        )

    if not np.isfinite(sequences).all():
        raise ValueError(
            "Financial sequences contain NaN or Inf."
        )

    if not np.isfinite(conditions).all():
        raise ValueError(
            "Conditioning vectors contain NaN or Inf."
        )

    print("Sequence shape:       PASS")
    print("Condition shape:      PASS")
    print("Count alignment:      PASS")
    print("Financial finite:     PASS")
    print("Condition finite:     PASS")

    # ========================================================
    # CONDITIONING DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print("3. CONDITIONING DISTRIBUTION")
    print("=" * 80)

    metadata = pd.read_csv(
        METADATA_PATH
    )

    if len(metadata) != len(sequences):
        raise ValueError(
            "Metadata count does not match sequences."
        )

    print(
        metadata["Dominant_Regime"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    train_data = torch.from_numpy(
        sequences
    ).to(DEVICE)

    condition_tensor = torch.from_numpy(
        conditions
    ).to(DEVICE)

    # ========================================================
    # MODELS
    # ========================================================

    print()
    print("=" * 80)
    print("4. INITIALIZING CONDITIONAL TIMEGAN")
    print("=" * 80)

    embedder = Embedder(
        N_FEATURES,
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    recovery = Recovery(
        HIDDEN_DIM,
        N_FEATURES,
        NUM_LAYERS,
    ).to(DEVICE)

    generator = Generator(
        HIDDEN_DIM,
        COND_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    supervisor = Supervisor(
        HIDDEN_DIM,
        COND_DIM,
        NUM_LAYERS,
    ).to(DEVICE)

    discriminator = Discriminator(
        HIDDEN_DIM,
        COND_DIM,
        NUM_LAYERS,
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

    print(
        f"Embedder parameters:      "
        f"{count_parameters(embedder):,}"
    )

    print(
        f"Recovery parameters:      "
        f"{count_parameters(recovery):,}"
    )

    print(
        f"Generator parameters:     "
        f"{count_parameters(generator):,}"
    )

    print(
        f"Supervisor parameters:    "
        f"{count_parameters(supervisor):,}"
    )

    print(
        f"Discriminator parameters:  "
        f"{count_parameters(discriminator):,}"
    )

    total_parameters = sum(
        [
            count_parameters(embedder),
            count_parameters(recovery),
            count_parameters(generator),
            count_parameters(supervisor),
            count_parameters(discriminator),
        ]
    )

    print(
        f"Total parameters:          "
        f"{total_parameters:,}"
    )

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_history = train_embedder_recovery(
        embedder,
        recovery,
        train_data,
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_history = train_supervisor(
        embedder,
        supervisor,
        train_data,
        condition_tensor,
    )

    # ========================================================
    # STAGE 3
    # ========================================================

    stage3_history = train_joint(
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator,
        train_data,
        condition_tensor,
    )

    # ========================================================
    # COMBINE HISTORY
    # ========================================================

    history_rows = []

    for epoch, loss in enumerate(
        stage1_history,
        start=1,
    ):

        history_rows.append(
            {
                "stage": "stage1",
                "epoch": epoch,
                "reconstruction": loss,
            }
        )

    for epoch, loss in enumerate(
        stage2_history,
        start=1,
    ):

        history_rows.append(
            {
                "stage": "stage2",
                "epoch": epoch,
                "supervised": loss,
            }
        )

    for record in stage3_history:

        row = {
            "stage": "stage3",
            **record,
        }

        history_rows.append(
            row
        )

    history_df = pd.DataFrame(
        history_rows
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print()
    print(
        f"Training history saved to:\n"
        f"{HISTORY_PATH}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_checkpoint(
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator,
        history_rows,
    )

    # ========================================================
    # FINAL
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("V3.4 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("Architecture:")
    print(
        f"  Financial features:       {N_FEATURES}"
    )
    print(
        f"  Conditioning features:    {COND_DIM}"
    )
    print(
        f"  Sequence length:          {SEQ_LEN}"
    )
    print(
        f"  Hidden dimension:         {HIDDEN_DIM}"
    )
    print(
        f"  GRU layers:               {NUM_LAYERS}"
    )

    print()
    print("Training:")
    print(
        f"  Stage 1 epochs:           {STAGE1_EPOCHS}"
    )
    print(
        f"  Stage 2 epochs:           {STAGE2_EPOCHS}"
    )
    print(
        f"  Stage 3 epochs:           {STAGE3_EPOCHS}"
    )

    print()
    print("V3.4 conditioning:")
    print(
        "  Categorical regime:       4 dimensions"
    )
    print(
        "  Continuous stress:        5 dimensions"
    )
    print(
        "  Total conditioning:       9 dimensions"
    )

    print()
    print("No V3.3 global extreme-dependence losses were used.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
