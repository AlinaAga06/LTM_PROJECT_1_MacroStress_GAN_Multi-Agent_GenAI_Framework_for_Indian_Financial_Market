# ============================================================
# TIMEGAN V1 - STOCK MARKET TRAINING
# MacroStress-GAN
#
# Institution-specific TimeGAN model
# Focus: Stock Market
#
# REAL DATASET
#   NIFTY50
#   INDIA VIX
#   USD/INR
#   CRUDE OIL
#   INDIA 10Y GOVERNMENT BOND YIELD
#
# TIMEGAN FEATURES
#   1. NIFTY50_Return
#   2. CRUDE_OIL_Return
#   3. USD_INR_Return
#   4. INDIA_VIX_Change
#   5. INDIA_10Y_YIELD_Change
#
# Sequence length : 30 days
# Hidden dimension: 24
# GRU layers      : 2
#
# Training:
#   Stage 1 -> Embedder + Recovery
#   Stage 2 -> Supervisor
#   Stage 3 -> Joint Adversarial Training
#
# OUTPUT:
#   models/timegan/stock/timegan_stock_v1.pt
#   models/timegan/stock/training_history_stock_v1.csv
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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "stock_timegan_sequences.npy"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "stock"
)


MODEL_PATH = (
    MODEL_DIR
    / "timegan_stock_v1.pt"
)


HISTORY_PATH = (
    MODEL_DIR
    / "training_history_stock_v1.csv"
)


# ============================================================
# TIMEGAN CONFIGURATION
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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL 1
# EMBEDDER
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
# MODEL 2
# RECOVERY
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
# MODEL 3
# GENERATOR
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
# MODEL 4
# SUPERVISOR
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
# MODEL 5
# DISCRIMINATOR
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


        # Final timestep

        y = y[:, -1, :]


        y = self.linear(y)


        return y


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING STOCK TIMEGAN DATA")
    print("=" * 70)


    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"""
Stock TimeGAN data not found:

{DATA_PATH}

Run:

python scripts/prepare_stock_timegan.py

first.
"""
        )


    data = np.load(
        DATA_PATH
    )


    print()
    print(f"Data path : {DATA_PATH}")
    print(f"Shape     : {data.shape}")
    print(f"Dtype     : {data.dtype}")


    # ========================================================
    # VALIDATION
    # ========================================================

    if data.ndim != 3:

        raise ValueError(
            f"Expected 3D array, got {data.shape}"
        )


    if data.shape[1] != SEQ_LENGTH:

        raise ValueError(
            f"""
Expected sequence length:

{SEQ_LENGTH}

Got:

{data.shape[1]}
"""
        )


    if data.shape[2] != FEATURE_DIM:

        raise ValueError(
            f"""
Expected feature dimension:

{FEATURE_DIM}

Got:

{data.shape[2]}
"""
        )


    if np.isnan(data).any():

        raise ValueError(
            "Dataset contains NaN values."
        )


    if np.isinf(data).any():

        raise ValueError(
            "Dataset contains infinite values."
        )


    # ========================================================
    # RANGE CHECK
    # ========================================================

    print()
    print("DATA RANGE")
    print("-" * 70)

    print(
        f"Minimum value : {data.min():.6f}"
    )

    print(
        f"Maximum value : {data.max():.6f}"
    )


    # ========================================================
    # CONVERT TO TENSOR
    # ========================================================

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
    print(
        f"Training sequences : {len(dataset)}"
    )

    print(
        f"Batch size         : {BATCH_SIZE}"
    )

    print(
        f"Batches per epoch  : {len(loader)}"
    )


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


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator
):

    print()
    print("=" * 70)
    print("STOCK TIMEGAN MODEL INFORMATION")
    print("=" * 70)


    embedder_params = count_parameters(
        embedder
    )


    recovery_params = count_parameters(
        recovery
    )


    generator_params = count_parameters(
        generator
    )


    supervisor_params = count_parameters(
        supervisor
    )


    discriminator_params = count_parameters(
        discriminator
    )


    print(
        f"Embedder parameters      : "
        f"{embedder_params:,}"
    )


    print(
        f"Recovery parameters      : "
        f"{recovery_params:,}"
    )


    print(
        f"Generator parameters     : "
        f"{generator_params:,}"
    )


    print(
        f"Supervisor parameters    : "
        f"{supervisor_params:,}"
    )


    print(
        f"Discriminator parameters : "
        f"{discriminator_params:,}"
    )


    total = (
        embedder_params
        + recovery_params
        + generator_params
        + supervisor_params
        + discriminator_params
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


    for epoch in range(
        1,
        EMBEDDER_EPOCHS + 1
    ):

        epoch_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(
                DEVICE
            )


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
                f"Epoch [{epoch:3d}/"
                f"{EMBEDDER_EPOCHS}] "
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


    # Freeze embedder

    embedder.eval()


    for epoch in range(
        1,
        SUPERVISOR_EPOCHS + 1
    ):

        epoch_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(
                DEVICE
            )


            optimizer.zero_grad()


            with torch.no_grad():

                latent = embedder(
                    real_data
                )


            supervised = supervisor(
                latent
            )


            # Predict next latent state

            target = latent[:, 1:, :]


            prediction = supervised[
                :, :-1, :
            ]


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
                f"Epoch [{epoch:3d}/"
                f"{SUPERVISOR_EPOCHS}] "
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


    # ========================================================
    # OPTIMIZERS
    # ========================================================

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


    # ========================================================
    # LOSS FUNCTIONS
    # ========================================================

    adversarial_loss = nn.BCEWithLogitsLoss()


    mse_loss = nn.MSELoss()


    history = []


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        JOINT_EPOCHS + 1
    ):

        epoch_g_loss = 0.0

        epoch_d_loss = 0.0

        epoch_e_loss = 0.0


        for batch in loader:

            real_data = batch[0].to(
                DEVICE
            )


            batch_size = (
                real_data.size(0)
            )


            # =================================================
            # 1. GENERATOR + SUPERVISOR
            # =================================================

            optimizer_generator.zero_grad()


            z = torch.rand(
                batch_size,
                SEQ_LENGTH,
                FEATURE_DIM,
                device=DEVICE
            )


            fake_latent = generator(
                z
            )


            fake_supervised = supervisor(
                fake_latent
            )


            # Real latent

            with torch.no_grad():

                real_latent = embedder(
                    real_data
                )


            # Discriminator

            fake_prediction = discriminator(
                fake_supervised
            )


            real_labels = torch.ones(
                batch_size,
                1,
                device=DEVICE
            )


            g_adv_loss = adversarial_loss(
                fake_prediction,
                real_labels
            )


            # =================================================
            # SUPERVISED TEMPORAL LOSS
            # =================================================

            if SEQ_LENGTH > 1:

                supervised_prediction = (
                    supervisor(
                        real_latent
                    )
                )


                supervised_target = (
                    real_latent[:, 1:, :]
                )


                supervised_output = (
                    supervised_prediction[:, :-1, :]
                )


                g_supervised_loss = mse_loss(
                    supervised_output,
                    supervised_target
                )


            else:

                g_supervised_loss = torch.tensor(
                    0.0,
                    device=DEVICE
                )


            # =================================================
            # MOMENT MATCHING
            # =================================================

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


            # =================================================
            # TOTAL GENERATOR LOSS
            # =================================================

            g_loss = (
                g_adv_loss
                + 100.0 * g_supervised_loss
                + 10.0 * moment_loss
            )


            g_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters())
                + list(supervisor.parameters()),
                max_norm=1.0
            )


            optimizer_generator.step()


            # =================================================
            # 2. DISCRIMINATOR
            # =================================================

            optimizer_discriminator.zero_grad()


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
            # 3. EMBEDDER + RECOVERY
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
            # RECORD
            # =================================================

            epoch_g_loss += (
                g_loss.item()
            )


            epoch_d_loss += (
                d_loss.item()
            )


            epoch_e_loss += (
                reconstruction_loss.item()
            )


        # =====================================================
        # AVERAGE
        # =====================================================

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


        # =====================================================
        # PROGRESS
        # =====================================================

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == JOINT_EPOCHS
        ):

            print(
                f"Epoch [{epoch:3d}/"
                f"{JOINT_EPOCHS}] "
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

        "model_version":
            "TimeGAN_Stock_V1",

        "institution":
            "Stock Market",

        "description":
            "TimeGAN trained on real stock-market "
            "financial returns and changes",

        "data_type":
            "REAL",

        "features":
            FEATURES,

        "feature_dim":
            FEATURE_DIM,

        "sequence_length":
            SEQ_LENGTH,

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

        "seed":
            SEED
    }


    torch.save(
        checkpoint,
        MODEL_PATH
    )


    print()
    print("=" * 70)
    print("STOCK TIMEGAN MODEL SAVED")
    print("=" * 70)


    print(
        f"Path: {MODEL_PATH}"
    )


# ============================================================
# SAVE HISTORY
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
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V1 TRAINING")
    print("=" * 70)


    print()
    print(
        f"Project root    : {PROJECT_ROOT}"
    )


    print(
        f"Device          : {DEVICE}"
    )


    print(
        f"Sequence length : {SEQ_LENGTH}"
    )


    print(
        f"Feature count   : {FEATURE_DIM}"
    )


    print(
        f"Hidden dimension: {HIDDEN_DIM}"
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


    print(
        f"Total epochs    : "
        f"{EMBEDDER_EPOCHS + SUPERVISOR_EPOCHS + JOINT_EPOCHS}"
    )


    # ========================================================
    # FEATURES
    # ========================================================

    print()
    print("STOCK MARKET FEATURES")
    print("-" * 70)


    for index, feature in enumerate(
        FEATURES,
        start=1
    ):

        print(
            f"{index}. {feature}"
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
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("STOCK TIMEGAN V1 TRAINING COMPLETED")
    print("=" * 70)


    print()
    print("Outputs:")
    print()


    print(
        f"1. Model:"
    )


    print(
        f"   {MODEL_PATH}"
    )


    print()


    print(
        f"2. Training history:"
    )


    print(
        f"   {HISTORY_PATH}"
    )


    print()


    print(
        "Stock Market TimeGAN V1 is ready for "
        "synthetic sequence generation."
    )


    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()