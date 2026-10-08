# ============================================================
# TIMEGAN V2 - TRAINING SCRIPT
# MacroStress-GAN
#
# V2 trains on financial RETURNS / CHANGES instead of
# raw market price levels.
#
# Features:
#   1. NIFTY50_Return
#   2. CRUDE_OIL_Return
#   3. USD_INR_Return
#   4. INDIA_VIX_Change
#   5. INDIA_10Y_YIELD_Change
#
# Sequence length: 30 days
#
# Training:
#   Stage 1 -> Embedder + Recovery
#   Stage 2 -> Supervisor
#   Stage 3 -> Joint Adversarial Training
#
# Output:
#   models/timegan/timegan_v2_model.pt
#   models/timegan/training_history_v2.csv
# ============================================================


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "timegan_v2_sequences.npy"

MODEL_DIR = PROJECT_ROOT / "models" / "timegan"

MODEL_PATH = MODEL_DIR / "timegan_v2_model.pt"

HISTORY_PATH = MODEL_DIR / "training_history_v2.csv"


# ============================================================
# TIMEGAN PARAMETERS
# ============================================================

SEQ_LENGTH = 30

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

FEATURE_DIM = len(FEATURES)

HIDDEN_DIM = 24

NUM_LAYERS = 2

BATCH_SIZE = 64

EMBEDDER_EPOCHS = 100

SUPERVISOR_EPOCHS = 100

JOINT_EPOCHS = 300

LEARNING_RATE = 0.001

SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL 1: EMBEDDER
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

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# MODEL 2: RECOVERY
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

        self.linear = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, h):

        x, _ = self.gru(h)

        x = self.linear(x)

        x = self.sigmoid(x)

        return x


# ============================================================
# MODEL 3: GENERATOR
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

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, z):

        h, _ = self.gru(z)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# MODEL 4: SUPERVISOR
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

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, h):

        s, _ = self.gru(h)

        s = self.linear(s)

        s = self.sigmoid(s)

        return s


# ============================================================
# MODEL 5: DISCRIMINATOR
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

        self.linear = nn.Linear(
            hidden_dim,
            1
        )


    def forward(self, h):

        y, _ = self.gru(h)

        # Use the final time step
        y = y[:, -1, :]

        y = self.linear(y)

        return y


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING TIMEGAN V2 DATA")
    print("=" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nTimeGAN V2 data not found:\n{DATA_PATH}\n\n"
            "Run data_preparation_returns.py first."
        )


    data = np.load(DATA_PATH)


    print()
    print(f"Data path : {DATA_PATH}")
    print(f"Shape     : {data.shape}")
    print(f"Dtype     : {data.dtype}")


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if data.ndim != 3:

        raise ValueError(
            f"Expected 3D array, got shape {data.shape}"
        )


    if data.shape[1] != SEQ_LENGTH:

        raise ValueError(
            f"Expected sequence length {SEQ_LENGTH}, "
            f"got {data.shape[1]}"
        )


    if data.shape[2] != FEATURE_DIM:

        raise ValueError(
            f"Expected {FEATURE_DIM} features, "
            f"got {data.shape[2]}"
        )


    if np.isnan(data).any():

        raise ValueError(
            "Dataset contains NaN values."
        )


    if np.isinf(data).any():

        raise ValueError(
            "Dataset contains infinite values."
        )


    # --------------------------------------------------------
    # CONVERT TO TENSOR
    # --------------------------------------------------------

    tensor_data = torch.tensor(
        data,
        dtype=torch.float32
    )


    dataset = TensorDataset(
        tensor_data
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )


    print()
    print(f"Training sequences : {len(dataset)}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Batches per epoch  : {len(loader)}")

    return data, loader


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

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
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)


    supervisor = Supervisor(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)


    discriminator = Discriminator(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)


    return (
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_model_information(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator
):

    print()
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print(
        f"Embedder parameters      : "
        f"{count_parameters(embedder):,}"
    )

    print(
        f"Recovery parameters      : "
        f"{count_parameters(recovery):,}"
    )

    print(
        f"Generator parameters     : "
        f"{count_parameters(generator):,}"
    )

    print(
        f"Supervisor parameters    : "
        f"{count_parameters(supervisor):,}"
    )

    print(
        f"Discriminator parameters : "
        f"{count_parameters(discriminator):,}"
    )


    total = (
        count_parameters(embedder)
        + count_parameters(recovery)
        + count_parameters(generator)
        + count_parameters(supervisor)
        + count_parameters(discriminator)
    )


    print(
        f"Total trainable parameters: "
        f"{total:,}"
    )


# ============================================================
# STAGE 1
# EMBEDDER + RECOVERY
# ============================================================

def train_embedder(
    embedder,
    recovery,
    loader
):

    print()
    print("=" * 70)
    print("STAGE 1: EMBEDDER + RECOVERY")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        list(embedder.parameters())
        + list(recovery.parameters()),
        lr=LEARNING_RATE
    )


    criterion = nn.MSELoss()


    history = []


    for epoch in range(1, EMBEDDER_EPOCHS + 1):

        epoch_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(DEVICE)


            optimizer.zero_grad()


            latent = embedder(
                real_data
            )


            reconstructed = recovery(
                latent
            )


            loss = criterion(
                reconstructed,
                real_data
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                list(embedder.parameters())
                + list(recovery.parameters()),
                max_norm=1.0
            )


            optimizer.step()


            epoch_loss += loss.item()


        epoch_loss /= len(loader)


        history.append({
            "stage": "embedder",
            "epoch": epoch,
            "loss": epoch_loss
        })


        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EMBEDDER_EPOCHS
        ):

            print(
                f"Epoch [{epoch:3d}/{EMBEDDER_EPOCHS}] "
                f"Reconstruction Loss: "
                f"{epoch_loss:.6f}"
            )


    return history


# ============================================================
# STAGE 2
# SUPERVISOR
# ============================================================

def train_supervisor(
    embedder,
    supervisor,
    loader
):

    print()
    print("=" * 70)
    print("STAGE 2: SUPERVISOR")
    print("=" * 70)


    optimizer = torch.optim.Adam(
        supervisor.parameters(),
        lr=LEARNING_RATE
    )


    criterion = nn.MSELoss()


    history = []


    # Freeze embedder during supervisor training

    embedder.eval()


    for epoch in range(1, SUPERVISOR_EPOCHS + 1):

        epoch_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(DEVICE)


            optimizer.zero_grad()


            with torch.no_grad():

                latent = embedder(
                    real_data
                )


            supervised = supervisor(
                latent
            )


            # Supervisor predicts next latent state

            target = latent[:, 1:, :]

            prediction = supervised[:, :-1, :]


            loss = criterion(
                prediction,
                target
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                supervisor.parameters(),
                max_norm=1.0
            )


            optimizer.step()


            epoch_loss += loss.item()


        epoch_loss /= len(loader)


        history.append({
            "stage": "supervisor",
            "epoch": epoch,
            "loss": epoch_loss
        })


        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == SUPERVISOR_EPOCHS
        ):

            print(
                f"Epoch [{epoch:3d}/{SUPERVISOR_EPOCHS}] "
                f"Supervised Loss: "
                f"{epoch_loss:.6f}"
            )


    return history


# ============================================================
# STAGE 3
# JOINT ADVERSARIAL TRAINING
# ============================================================

def train_joint(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
    loader
):

    print()
    print("=" * 70)
    print("STAGE 3: JOINT ADVERSARIAL TRAINING")
    print("=" * 70)


    # --------------------------------------------------------
    # OPTIMIZERS
    # --------------------------------------------------------

    optimizer_generator = torch.optim.Adam(
        list(generator.parameters())
        + list(supervisor.parameters()),
        lr=LEARNING_RATE
    )


    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE
    )


    optimizer_embedder = torch.optim.Adam(
        list(embedder.parameters())
        + list(recovery.parameters()),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # LOSS FUNCTIONS
    # --------------------------------------------------------

    adversarial_loss = nn.BCEWithLogitsLoss()

    mse_loss = nn.MSELoss()


    history = []


    for epoch in range(1, JOINT_EPOCHS + 1):

        epoch_g_loss = 0.0

        epoch_d_loss = 0.0

        epoch_e_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(DEVICE)

            batch_size = real_data.size(0)


            # =================================================
            # 1. TRAIN GENERATOR + SUPERVISOR
            # =================================================

            optimizer_generator.zero_grad()


            # Random noise

            z = torch.rand(
                batch_size,
                SEQ_LENGTH,
                FEATURE_DIM,
                device=DEVICE
            )


            # Generate latent representation

            fake_latent = generator(z)


            # Supervisor improves temporal dynamics

            fake_supervised = supervisor(
                fake_latent
            )


            # Generate real latent representation

            with torch.no_grad():

                real_latent = embedder(
                    real_data
                )


            # -------------------------------------------------
            # Discriminator prediction
            # -------------------------------------------------

            fake_prediction = discriminator(
                fake_supervised
            )


            # Generator wants discriminator
            # to classify generated data as real

            real_labels = torch.ones(
                batch_size,
                1,
                device=DEVICE
            )


            g_adv_loss = adversarial_loss(
                fake_prediction,
                real_labels
            )


            # -------------------------------------------------
            # Supervised temporal loss
            # -------------------------------------------------

            if SEQ_LENGTH > 1:

                supervised_prediction = supervisor(
                    real_latent
                )


                supervised_target = real_latent[
                    :, 1:, :
                ]


                supervised_output = supervised_prediction[
                    :, :-1, :
                ]


                g_supervised_loss = mse_loss(
                    supervised_output,
                    supervised_target
                )

            else:

                g_supervised_loss = torch.tensor(
                    0.0,
                    device=DEVICE
                )


            # -------------------------------------------------
            # Moment matching loss
            # -------------------------------------------------

            fake_mean = fake_supervised.mean(
                dim=(0, 1)
            )


            real_mean = real_latent.mean(
                dim=(0, 1)
            )


            fake_std = fake_supervised.std(
                dim=(0, 1)
            )


            real_std = real_latent.std(
                dim=(0, 1)
            )


            moment_loss = (
                mse_loss(
                    fake_mean,
                    real_mean
                )
                +
                mse_loss(
                    fake_std,
                    real_std
                )
            )


            # -------------------------------------------------
            # TOTAL GENERATOR LOSS
            # -------------------------------------------------

            g_loss = (
                g_adv_loss
                +
                100.0 * g_supervised_loss
                +
                10.0 * moment_loss
            )


            g_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters())
                + list(supervisor.parameters()),
                max_norm=1.0
            )


            optimizer_generator.step()


            # =================================================
            # 2. TRAIN DISCRIMINATOR
            # =================================================

            optimizer_discriminator.zero_grad()


            # Real data

            real_latent_detached = (
                real_latent.detach()
            )


            real_prediction = discriminator(
                real_latent_detached
            )


            real_loss = adversarial_loss(
                real_prediction,
                real_labels
            )


            # Fake data

            fake_latent_detached = (
                fake_supervised.detach()
            )


            fake_prediction = discriminator(
                fake_latent_detached
            )


            fake_labels = torch.zeros(
                batch_size,
                1,
                device=DEVICE
            )


            fake_loss = adversarial_loss(
                fake_prediction,
                fake_labels
            )


            d_loss = (
                real_loss
                + fake_loss
            )


            d_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=1.0
            )


            optimizer_discriminator.step()


            # =================================================
            # 3. TRAIN EMBEDDER + RECOVERY
            # =================================================

            optimizer_embedder.zero_grad()


            latent = embedder(
                real_data
            )


            reconstructed = recovery(
                latent
            )


            reconstruction_loss = mse_loss(
                reconstructed,
                real_data
            )


            reconstruction_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                list(embedder.parameters())
                + list(recovery.parameters()),
                max_norm=1.0
            )


            optimizer_embedder.step()


            # =================================================
            # RECORD LOSSES
            # =================================================

            epoch_g_loss += g_loss.item()

            epoch_d_loss += d_loss.item()

            epoch_e_loss += reconstruction_loss.item()


        # -----------------------------------------------------
        # AVERAGE LOSSES
        # -----------------------------------------------------

        epoch_g_loss /= len(loader)

        epoch_d_loss /= len(loader)

        epoch_e_loss /= len(loader)


        history.append({
            "stage": "joint",
            "epoch": epoch,
            "generator_loss": epoch_g_loss,
            "discriminator_loss": epoch_d_loss,
            "embedder_loss": epoch_e_loss
        })


        # -----------------------------------------------------
        # PRINT PROGRESS
        # -----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == JOINT_EPOCHS
        ):

            print(
                f"Epoch [{epoch:3d}/{JOINT_EPOCHS}] "
                f"G: {epoch_g_loss:.6f} | "
                f"D: {epoch_d_loss:.6f} | "
                f"E: {epoch_e_loss:.6f}"
            )


    return history


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator
):

    checkpoint = {

        "model_version": "TimeGAN_V2",

        "description":
            "TimeGAN trained on financial returns and changes",

        "features": FEATURES,

        "feature_dim": FEATURE_DIM,

        "sequence_length": SEQ_LENGTH,

        "hidden_dim": HIDDEN_DIM,

        "num_layers": NUM_LAYERS,

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

        "seed": SEED
    }


    torch.save(
        checkpoint,
        MODEL_PATH
    )


    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Path: {MODEL_PATH}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    if len(history) == 0:

        return


    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )


    print(
        f"Training history: {HISTORY_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TIMEGAN V2 TRAINING")
    print("=" * 70)


    print()
    print(f"Project root       : {PROJECT_ROOT}")

    print(
        f"Device             : {DEVICE}"
    )

    print(
        f"Sequence length    : {SEQ_LENGTH}"
    )

    print(
        f"Features            : {FEATURES}"
    )

    print(
        f"Batch size         : {BATCH_SIZE}"
    )

    print(
        f"Stage 1 epochs     : {EMBEDDER_EPOCHS}"
    )

    print(
        f"Stage 2 epochs     : {SUPERVISOR_EPOCHS}"
    )

    print(
        f"Stage 3 epochs     : {JOINT_EPOCHS}"
    )

    print(
        f"Total epochs       : "
        f"{EMBEDDER_EPOCHS + SUPERVISOR_EPOCHS + JOINT_EPOCHS}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    data, loader = load_data()


    # ========================================================
    # CREATE MODELS
    # ========================================================

    (
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator
    ) = create_models()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    print_model_information(
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator
    )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    complete_history = []


    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_history = train_embedder(
        embedder,
        recovery,
        loader
    )


    complete_history.extend(
        stage1_history
    )


    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_history = train_supervisor(
        embedder,
        supervisor,
        loader
    )


    complete_history.extend(
        stage2_history
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
        loader
    )


    complete_history.extend(
        stage3_history
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_model(
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    save_history(
        complete_history
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)
    print("TIMEGAN V2 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Outputs:")

    print(
        f"1. Model   : {MODEL_PATH}"
    )

    print(
        f"2. History : {HISTORY_PATH}"
    )

    print()

    print("Next step:")
    print(
        "Generate synthetic financial return/change sequences "
        "using the trained TimeGAN V2 model."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()