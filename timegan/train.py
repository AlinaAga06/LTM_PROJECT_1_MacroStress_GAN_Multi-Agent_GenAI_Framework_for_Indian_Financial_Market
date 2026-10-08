"""
============================================================
MacroStress-GAN - TimeGAN Training
============================================================

Purpose:
    Train TimeGAN on 30-day sequences of Indian financial
    market variables.

Input:
    data/processed/timegan_sequences.npy

Shape:
    (3989, 30, 5)

Features:
    1. NIFTY50
    2. INDIA_VIX
    3. CRUDE_OIL
    4. USD_INR
    5. INDIA_10Y_YIELD

Training stages:
    Stage 1 - Embedder + Recovery pretraining
    Stage 2 - Supervisor pretraining
    Stage 3 - Joint adversarial training

Output:
    models/timegan/timegan_model.pt
    models/timegan/training_history.csv
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import TimeGAN


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_sequences.npy"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "timegan"
)

MODEL_PATH = MODEL_DIR / "timegan_model.pt"

HISTORY_PATH = MODEL_DIR / "training_history.csv"


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 5
HIDDEN_DIM = 24
NOISE_DIM = 5

NUM_LAYERS = 2
SEQUENCE_LENGTH = 30

BATCH_SIZE = 64

# ------------------------------------------------------------
# Training epochs
# ------------------------------------------------------------

EMBEDDER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 300

# ------------------------------------------------------------
# Learning rates
# ------------------------------------------------------------

LEARNING_RATE = 0.001

# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

GAMMA = 1.0

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    """
    Load prepared TimeGAN sequences.
    """

    print("=" * 70)
    print("LOADING TIMEGAN TRAINING DATA")
    print("=" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nTraining data not found:\n{DATA_PATH}\n\n"
            "Please run:\n"
            "python timegan\\data_preparation.py"
        )

    data = np.load(DATA_PATH)

    print(f"\nData shape: {data.shape}")
    print(f"Data type : {data.dtype}")

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(data.shape) != 3:

        raise ValueError(
            f"Expected 3-dimensional data, got {data.shape}"
        )

    if data.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, "
            f"got {data.shape[1]}"
        )

    if data.shape[2] != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"got {data.shape[2]}"
        )

    # --------------------------------------------------------
    # Check NaN / Infinity
    # --------------------------------------------------------

    if np.isnan(data).any():

        raise ValueError(
            "Training data contains NaN values."
        )

    if np.isinf(data).any():

        raise ValueError(
            "Training data contains infinite values."
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    data = data.astype(np.float32)

    # --------------------------------------------------------
    # Check normalized range
    # --------------------------------------------------------

    print(
        f"Minimum value: {data.min():.6f}"
    )

    print(
        f"Maximum value: {data.max():.6f}"
    )

    print(
        f"Number of sequences: {len(data)}"
    )

    print(
        f"Sequence length: {data.shape[1]}"
    )

    print(
        f"Number of features: {data.shape[2]}"
    )

    print("\n✓ Training data validated")

    return data


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(data):
    """
    Create PyTorch DataLoader.
    """

    tensor_data = torch.tensor(
        data,
        dtype=torch.float32
    )

    dataset = TensorDataset(tensor_data)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print("\nDataLoader created")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(loader)}")

    return loader


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def reconstruction_loss(real, recovered):
    """
    Reconstruction loss between original and recovered data.
    """

    return nn.MSELoss()(recovered, real)


def supervised_loss(h, h_supervise):
    """
    Supervisor loss.

    The supervisor learns the temporal dynamics between
    consecutive latent states.
    """

    return nn.MSELoss()(
        h_supervise[:, :-1, :],
        h[:, 1:, :]
    )


# ============================================================
# STAGE 1
# EMBEDDER + RECOVERY PRETRAINING
# ============================================================

def train_embedder(
    model,
    data_loader,
    optimizer,
    epochs
):
    """
    Stage 1:

    Train Embedder and Recovery so that:

        Real Data
            ↓
        Embedder
            ↓
        Latent Representation
            ↓
        Recovery
            ↓
        Reconstructed Data

    Objective:
        reconstructed data ≈ real data
    """

    print("\n")
    print("=" * 70)
    print("STAGE 1: EMBEDDER + RECOVERY PRETRAINING")
    print("=" * 70)

    model.embedder.train()
    model.recovery.train()

    history = []

    for epoch in range(1, epochs + 1):

        epoch_loss = 0.0

        batch_count = 0

        for batch in data_loader:

            real_data = batch[0].to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            h = model.embedder(real_data)

            recovered = model.recovery(h)

            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            loss = reconstruction_loss(
                real_data,
                recovered
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                list(model.embedder.parameters())
                + list(model.recovery.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()

            batch_count += 1

        average_loss = epoch_loss / batch_count

        history.append({
            "stage": "embedder",
            "epoch": epoch,
            "loss": average_loss
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch [{epoch:4d}/{epochs}] "
                f"Reconstruction Loss: "
                f"{average_loss:.6f}"
            )

    print("\n✓ Stage 1 completed")

    return history


# ============================================================
# STAGE 2
# SUPERVISOR PRETRAINING
# ============================================================

def train_supervisor(
    model,
    data_loader,
    optimizer,
    epochs
):
    """
    Stage 2:

    Train Supervisor to learn temporal relationships
    in the latent space.
    """

    print("\n")
    print("=" * 70)
    print("STAGE 2: SUPERVISOR PRETRAINING")
    print("=" * 70)

    model.embedder.eval()

    model.supervisor.train()

    history = []

    for epoch in range(1, epochs + 1):

        epoch_loss = 0.0

        batch_count = 0

        for batch in data_loader:

            real_data = batch[0].to(DEVICE)

            # ------------------------------------------------
            # Embed real data
            # ------------------------------------------------

            with torch.no_grad():

                h = model.embedder(real_data)

            # ------------------------------------------------
            # Supervisor
            # ------------------------------------------------

            h_supervise = model.supervisor(h)

            # ------------------------------------------------
            # Temporal supervised loss
            # ------------------------------------------------

            loss = supervised_loss(
                h,
                h_supervise
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.supervisor.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()

            batch_count += 1

        average_loss = epoch_loss / batch_count

        history.append({
            "stage": "supervisor",
            "epoch": epoch,
            "loss": average_loss
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch [{epoch:4d}/{epochs}] "
                f"Supervisor Loss: "
                f"{average_loss:.6f}"
            )

    print("\n✓ Stage 2 completed")

    return history


# ============================================================
# GENERATOR LOSS
# ============================================================

def generator_loss(
    model,
    real_data,
    noise
):
    """
    Calculate generator losses.
    """

    # --------------------------------------------------------
    # Generate latent representation
    # --------------------------------------------------------

    e_hat = model.generator(noise)

    # --------------------------------------------------------
    # Apply supervisor
    # --------------------------------------------------------

    h_hat = model.supervisor(e_hat)

    # --------------------------------------------------------
    # Generate synthetic data
    # --------------------------------------------------------

    x_hat = model.recovery(h_hat)

    # --------------------------------------------------------
    # Discriminator
    # --------------------------------------------------------

    y_fake = model.discriminator(h_hat)

    # --------------------------------------------------------
    # Adversarial loss
    # --------------------------------------------------------

    adversarial_loss = nn.BCEWithLogitsLoss()(
        y_fake,
        torch.ones_like(y_fake)
    )

    # --------------------------------------------------------
    # Supervised loss
    # --------------------------------------------------------

    with torch.no_grad():

        h_real = model.embedder(real_data)

    h_supervise_real = model.supervisor(h_real)

    g_supervised_loss = supervised_loss(
        h_real,
        h_supervise_real
    )

    # --------------------------------------------------------
    # Moment loss
    # --------------------------------------------------------

    real_mean = torch.mean(
        real_data,
        dim=(0, 1)
    )

    fake_mean = torch.mean(
        x_hat,
        dim=(0, 1)
    )

    real_std = torch.std(
        real_data,
        dim=(0, 1)
    )

    fake_std = torch.std(
        x_hat,
        dim=(0, 1)
    )

    mean_loss = torch.mean(
        torch.abs(real_mean - fake_mean)
    )

    std_loss = torch.mean(
        torch.abs(real_std - fake_std)
    )

    moment_loss = mean_loss + std_loss

    # --------------------------------------------------------
    # Total generator loss
    # --------------------------------------------------------

    total_loss = (
        adversarial_loss
        + GAMMA * g_supervised_loss
        + 100.0 * moment_loss
    )

    return (
        total_loss,
        adversarial_loss,
        g_supervised_loss,
        moment_loss
    )


# ============================================================
# DISCRIMINATOR LOSS
# ============================================================

def discriminator_loss(
    model,
    real_data,
    noise
):
    """
    Calculate discriminator loss.
    """

    # --------------------------------------------------------
    # Real latent representation
    # --------------------------------------------------------

    with torch.no_grad():

        h_real = model.embedder(real_data)

    # --------------------------------------------------------
    # Fake latent representation
    # --------------------------------------------------------

    with torch.no_grad():

        e_hat = model.generator(noise)

        h_fake = model.supervisor(e_hat)

    # --------------------------------------------------------
    # Discriminator predictions
    # --------------------------------------------------------

    y_real = model.discriminator(h_real)

    y_fake = model.discriminator(h_fake)

    # --------------------------------------------------------
    # Binary cross entropy
    # --------------------------------------------------------

    loss_real = nn.BCEWithLogitsLoss()(
        y_real,
        torch.ones_like(y_real)
    )

    loss_fake = nn.BCEWithLogitsLoss()(
        y_fake,
        torch.zeros_like(y_fake)
    )

    loss = loss_real + loss_fake

    return loss


# ============================================================
# STAGE 3
# JOINT ADVERSARIAL TRAINING
# ============================================================

def train_joint(
    model,
    data_loader,
    optimizer_generator,
    optimizer_embedder,
    optimizer_discriminator,
    epochs
):
    """
    Stage 3:

    Joint adversarial training of:

        Generator
        Supervisor
        Embedder
        Recovery
        Discriminator
    """

    print("\n")
    print("=" * 70)
    print("STAGE 3: JOINT ADVERSARIAL TRAINING")
    print("=" * 70)

    history = []

    model.train()

    for epoch in range(1, epochs + 1):

        g_epoch_loss = 0.0

        d_epoch_loss = 0.0

        e_epoch_loss = 0.0

        batch_count = 0

        for batch in data_loader:

            real_data = batch[0].to(DEVICE)

            batch_size = real_data.size(0)

            # ------------------------------------------------
            # Create random noise
            # ------------------------------------------------

            noise = torch.rand(
                batch_size,
                SEQUENCE_LENGTH,
                NOISE_DIM,
                device=DEVICE
            )

            # =================================================
            # GENERATOR + SUPERVISOR
            # =================================================

            optimizer_generator.zero_grad()

            (
                g_loss,
                adv_loss,
                sup_loss,
                mom_loss
            ) = generator_loss(
                model,
                real_data,
                noise
            )

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.generator.parameters())
                + list(model.supervisor.parameters())
                + list(model.recovery.parameters()),
                max_norm=1.0
            )

            optimizer_generator.step()

            # =================================================
            # EMBEDDER + RECOVERY
            # =================================================

            optimizer_embedder.zero_grad()

            h = model.embedder(real_data)

            recovered = model.recovery(h)

            reconstruction = reconstruction_loss(
                real_data,
                recovered
            )

            reconstruction.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.embedder.parameters())
                + list(model.recovery.parameters()),
                max_norm=1.0
            )

            optimizer_embedder.step()

            # =================================================
            # DISCRIMINATOR
            # =================================================

            optimizer_discriminator.zero_grad()

            d_loss = discriminator_loss(
                model,
                real_data,
                noise
            )

            # ------------------------------------------------
            # Update discriminator only when useful
            # ------------------------------------------------

            if d_loss.item() > 0.15:

                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.discriminator.parameters(),
                    max_norm=1.0
                )

                optimizer_discriminator.step()

            # ------------------------------------------------
            # Accumulate losses
            # ------------------------------------------------

            g_epoch_loss += g_loss.item()

            d_epoch_loss += d_loss.item()

            e_epoch_loss += reconstruction.item()

            batch_count += 1

        # ----------------------------------------------------
        # Average losses
        # ----------------------------------------------------

        avg_g = g_epoch_loss / batch_count

        avg_d = d_epoch_loss / batch_count

        avg_e = e_epoch_loss / batch_count

        history.append({
            "stage": "joint",
            "epoch": epoch,
            "generator_loss": avg_g,
            "discriminator_loss": avg_d,
            "embedder_loss": avg_e
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch [{epoch:4d}/{epochs}] "
                f"G Loss: {avg_g:.6f} | "
                f"D Loss: {avg_d:.6f} | "
                f"E Loss: {avg_e:.6f}"
            )

    print("\n✓ Stage 3 completed")

    return history


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save trained TimeGAN model.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "noise_dim":
            NOISE_DIM,

        "num_layers":
            NUM_LAYERS,

        "sequence_length":
            SEQUENCE_LENGTH,

        "features": [
            "NIFTY50",
            "INDIA_VIX",
            "CRUDE_OIL",
            "USD_INR",
            "INDIA_10Y_YIELD"
        ]
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):
    """
    Save training loss history.
    """

    if not history:
        return

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(history)

    df.to_csv(
        HISTORY_PATH,
        index=False
    )

    print("\nTraining history saved:")
    print(HISTORY_PATH)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MACROSTRESS-GAN")
    print("TIMEGAN TRAINING")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU not available - training on CPU"
        )

    print("\nConfiguration:")
    print(f"Input dimensions     : {INPUT_DIM}")
    print(f"Hidden dimensions    : {HIDDEN_DIM}")
    print(f"Noise dimensions     : {NOISE_DIM}")
    print(f"Sequence length      : {SEQUENCE_LENGTH}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Embedder epochs      : {EMBEDDER_EPOCHS}")
    print(f"Supervisor epochs    : {SUPERVISOR_EPOCHS}")
    print(f"Joint epochs         : {JOINT_EPOCHS}")

    # ========================================================
    # SET SEED
    # ========================================================

    set_seed(SEED)

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_training_data()

    # ========================================================
    # DATALOADER
    # ========================================================

    data_loader = create_dataloader(data)

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CREATING TIMEGAN MODEL")
    print("=" * 70)

    model = TimeGAN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        noise_dim=NOISE_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{total_parameters:,}"
    )

    # ========================================================
    # OPTIMIZERS - STAGE 1
    # ========================================================

    embedder_optimizer = optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters()),
        lr=LEARNING_RATE
    )

    # ========================================================
    # STAGE 1
    # ========================================================

    history_1 = train_embedder(
        model,
        data_loader,
        embedder_optimizer,
        EMBEDDER_EPOCHS
    )

    # ========================================================
    # OPTIMIZER - STAGE 2
    # ========================================================

    supervisor_optimizer = optim.Adam(
        model.supervisor.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    history_2 = train_supervisor(
        model,
        data_loader,
        supervisor_optimizer,
        SUPERVISOR_EPOCHS
    )

    # ========================================================
    # OPTIMIZERS - STAGE 3
    # ========================================================

    generator_optimizer = optim.Adam(
        list(model.generator.parameters())
        + list(model.supervisor.parameters())
        + list(model.recovery.parameters()),
        lr=LEARNING_RATE
    )

    embedder_optimizer_joint = optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters()),
        lr=LEARNING_RATE
    )

    discriminator_optimizer = optim.Adam(
        model.discriminator.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # STAGE 3
    # ========================================================

    history_3 = train_joint(
        model,
        data_loader,
        generator_optimizer,
        embedder_optimizer_joint,
        discriminator_optimizer,
        JOINT_EPOCHS
    )

    # ========================================================
    # COMBINE HISTORY
    # ========================================================

    complete_history = (
        history_1
        + history_2
        + history_3
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_model(model)

    save_history(
        complete_history
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TIMEGAN TRAINING COMPLETED")
    print("=" * 70)

    print("\n✓ Stage 1: Embedder + Recovery")
    print("✓ Stage 2: Supervisor")
    print("✓ Stage 3: Joint Adversarial Training")
    print("\n✓ Model saved successfully")
    print("\nNext step:")
    print("Generate synthetic financial market sequences.")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()