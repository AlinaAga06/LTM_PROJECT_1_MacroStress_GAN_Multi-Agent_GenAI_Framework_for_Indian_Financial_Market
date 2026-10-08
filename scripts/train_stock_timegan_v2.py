"""
MacroStress-GAN
Stock Market TimeGAN V2

Corrected V2 training.

Important:
    V1 is NOT modified.

Input:
    data/processed/institutions/stock/stock_timegan_sequences.npy

Output:
    models/timegan/stock/timegan_stock_v2.pt
    models/timegan/stock/training_history_stock_v2.csv
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "institutions"
    / "stock"
    / "stock_timegan_sequences.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "stock"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "timegan_stock_v2.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history_stock_v2.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

SEQ_LENGTH = 30
FEATURE_DIM = 5

HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 64

EMBEDDER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 400

LEARNING_RATE = 0.001

NOISE_DIM = FEATURE_DIM


FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]


# ============================================================
# REGULARIZATION WEIGHTS
# ============================================================

STATISTICS_WEIGHT = 1.0
CORRELATION_WEIGHT = 2.0
TEMPORAL_WEIGHT = 1.5


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

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

    def forward(self, x):

        h, _ = self.gru(x)

        h = torch.sigmoid(
            self.fc(h)
        )

        return h


class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim,
        feature_dim,
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
            feature_dim,
        )

    def forward(self, h):

        x, _ = self.gru(h)

        x = torch.sigmoid(
            self.fc(x)
        )

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

    def forward(self, z):

        h, _ = self.gru(z)

        h = torch.sigmoid(
            self.fc(h)
        )

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

    def forward(self, h):

        s, _ = self.gru(h)

        s = torch.sigmoid(
            self.fc(s)
        )

        return s


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

        d, _ = self.gru(h)

        d = self.fc(d)

        return d


# ============================================================
# NOISE
# ============================================================

def sample_noise(
    batch_size,
    seq_length,
    noise_dim,
):

    return torch.rand(
        batch_size,
        seq_length,
        noise_dim,
        device=DEVICE,
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_matrix(x):

    x = x.reshape(
        -1,
        x.shape[-1],
    )

    x = x - x.mean(
        dim=0,
        keepdim=True,
    )

    std = torch.sqrt(
        torch.mean(
            x ** 2,
            dim=0,
        )
        + 1e-6
    )

    normalized = x / std

    corr = (
        normalized.T @ normalized
    ) / max(
        x.shape[0],
        1,
    )

    return corr


# ============================================================
# TEMPORAL LAG-1 CORRELATION
# ============================================================

def temporal_lag1_correlation(x):

    x1 = x[:, :-1, :]
    x2 = x[:, 1:, :]

    x1 = x1.reshape(
        -1,
        x.shape[-1],
    )

    x2 = x2.reshape(
        -1,
        x.shape[-1],
    )

    x1 = x1 - x1.mean(
        dim=0,
        keepdim=True,
    )

    x2 = x2 - x2.mean(
        dim=0,
        keepdim=True,
    )

    numerator = (
        x1 * x2
    ).mean(dim=0)

    denominator = (
        torch.sqrt(
            (x1 ** 2).mean(dim=0)
            + 1e-6
        )
        *
        torch.sqrt(
            (x2 ** 2).mean(dim=0)
            + 1e-6
        )
    )

    return numerator / denominator


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 78)
    print("LOADING STOCK TIMEGAN V2 DATA")
    print("=" * 78)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Training data not found:\n{DATA_PATH}"
        )

    data = np.load(
        DATA_PATH
    )

    print(
        f"\nData path : {DATA_PATH}"
    )

    print(
        f"Shape     : {data.shape}"
    )

    print(
        f"Dtype     : {data.dtype}"
    )

    print(
        f"Minimum   : {data.min():.6f}"
    )

    print(
        f"Maximum   : {data.max():.6f}"
    )

    if data.ndim != 3:

        raise ValueError(
            "Expected 3D sequence data."
        )

    if data.shape[1] != SEQ_LENGTH:

        raise ValueError(
            f"Expected sequence length "
            f"{SEQ_LENGTH}, "
            f"got {data.shape[1]}"
        )

    if data.shape[2] != FEATURE_DIM:

        raise ValueError(
            f"Expected feature dimension "
            f"{FEATURE_DIM}, "
            f"got {data.shape[2]}"
        )

    if not np.isfinite(data).all():

        raise ValueError(
            "Training data contains NaN or Inf."
        )

    return torch.tensor(
        data,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# STAGE 1
# ============================================================

def train_embedder(
    embedder,
    recovery,
    data,
):

    print()
    print("=" * 78)
    print("STAGE 1 — EMBEDDER + RECOVERY")
    print("=" * 78)

    optimizer = optim.Adam(
        list(embedder.parameters())
        +
        list(recovery.parameters()),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    dataset_size = data.shape[0]

    for epoch in range(
        1,
        EMBEDDER_EPOCHS + 1,
    ):

        permutation = torch.randperm(
            dataset_size,
            device=DEVICE,
        )

        epoch_loss = 0.0
        batches = 0

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE,
        ):

            idx = permutation[
                start:start + BATCH_SIZE
            ]

            x = data[idx]

            h = embedder(x)

            reconstructed = recovery(h)

            loss = criterion(
                reconstructed,
                x,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            batches += 1

        epoch_loss /= max(
            batches,
            1,
        )

        history.append({
            "stage": "embedder",
            "epoch": epoch,
            "reconstruction_loss":
                epoch_loss,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EMBEDDER_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"Reconstruction Loss "
                f"{epoch_loss:.6f}"
            )

    return history


# ============================================================
# STAGE 2
# ============================================================

def train_supervisor(
    embedder,
    supervisor,
    data,
):

    print()
    print("=" * 78)
    print("STAGE 2 — SUPERVISOR")
    print("=" * 78)

    optimizer = optim.Adam(
        supervisor.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    dataset_size = data.shape[0]

    for epoch in range(
        1,
        SUPERVISOR_EPOCHS + 1,
    ):

        permutation = torch.randperm(
            dataset_size,
            device=DEVICE,
        )

        epoch_loss = 0.0
        batches = 0

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE,
        ):

            idx = permutation[
                start:start + BATCH_SIZE
            ]

            x = data[idx]

            with torch.no_grad():

                h = embedder(x)

            h_current = h[:, :-1, :]
            h_next = h[:, 1:, :]

            predicted = supervisor(
                h_current
            )

            loss = criterion(
                predicted,
                h_next,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            batches += 1

        epoch_loss /= max(
            batches,
            1,
        )

        history.append({
            "stage": "supervisor",
            "epoch": epoch,
            "supervised_loss":
                epoch_loss,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == SUPERVISOR_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"Supervised Loss "
                f"{epoch_loss:.6f}"
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
    data,
):

    print()
    print("=" * 78)
    print("STAGE 3 — JOINT ADVERSARIAL TRAINING V2")
    print("=" * 78)

    adversarial_loss = nn.BCEWithLogitsLoss()

    mse_loss = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    generator_optimizer = optim.Adam(
        list(generator.parameters())
        +
        list(supervisor.parameters())
        +
        list(recovery.parameters()),
        lr=LEARNING_RATE,
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    embedder_optimizer = optim.Adam(
        list(embedder.parameters())
        +
        list(recovery.parameters()),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Real targets
    # --------------------------------------------------------

    with torch.no_grad():

        real_flat = data.reshape(
            -1,
            FEATURE_DIM,
        )

        real_mean = real_flat.mean(
            dim=0
        )

        real_std = real_flat.std(
            dim=0
        )

        real_corr = correlation_matrix(
            data
        )

        real_lag1 = temporal_lag1_correlation(
            data
        )

    print(
        "\nReal transformed-data targets:"
    )

    print(
        "Mean:",
        real_mean.detach()
        .cpu()
        .numpy()
        .round(4)
    )

    print(
        "Std :",
        real_std.detach()
        .cpu()
        .numpy()
        .round(4)
    )

    print(
        "Lag1:",
        real_lag1.detach()
        .cpu()
        .numpy()
        .round(4)
    )

    history = []

    dataset_size = data.shape[0]

    for epoch in range(
        1,
        JOINT_EPOCHS + 1,
    ):

        permutation = torch.randperm(
            dataset_size,
            device=DEVICE,
        )

        epoch_g = 0.0
        epoch_d = 0.0
        epoch_e = 0.0

        epoch_stat = 0.0
        epoch_corr = 0.0
        epoch_temporal = 0.0

        batches = 0

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE,
        ):

            idx = permutation[
                start:start + BATCH_SIZE
            ]

            real = data[idx]

            batch_size = real.shape[0]

            # =================================================
            # A. GENERATOR UPDATE
            # =================================================

            z = sample_noise(
                batch_size,
                SEQ_LENGTH,
                NOISE_DIM,
            )

            h_fake_initial = generator(z)

            h_fake = supervisor(
                h_fake_initial
            )

            fake = recovery(
                h_fake
            )

            fake_disc = discriminator(
                h_fake
            )

            valid_labels = torch.ones_like(
                fake_disc
            )

            g_adv = adversarial_loss(
                fake_disc,
                valid_labels,
            )

            # -------------------------------------------------
            # Supervised temporal representation loss
            # -------------------------------------------------

            with torch.no_grad():

                real_h = embedder(
                    real
                )

            real_h_current = real_h[:, :-1, :]
            real_h_next = real_h[:, 1:, :]

            predicted_real_next = supervisor(
                real_h_current
            )

            supervised_loss = mse_loss(
                predicted_real_next,
                real_h_next,
            )

            # -------------------------------------------------
            # Feature statistics
            # -------------------------------------------------

            fake_flat = fake.reshape(
                -1,
                FEATURE_DIM,
            )

            fake_mean = fake_flat.mean(
                dim=0
            )

            fake_std = fake_flat.std(
                dim=0
            )

            stat_loss = (
                torch.mean(
                    (
                        fake_mean
                        - real_mean
                    ) ** 2
                )
                +
                torch.mean(
                    (
                        fake_std
                        - real_std
                    ) ** 2
                )
            )

            # -------------------------------------------------
            # Correlation
            # -------------------------------------------------

            fake_corr = correlation_matrix(
                fake
            )

            corr_loss = torch.mean(
                (
                    fake_corr
                    - real_corr
                ) ** 2
            )

            # -------------------------------------------------
            # Temporal lag-1
            # -------------------------------------------------

            fake_lag1 = temporal_lag1_correlation(
                fake
            )

            temporal_loss = torch.mean(
                (
                    fake_lag1
                    - real_lag1
                ) ** 2
            )

            # -------------------------------------------------
            # Total generator loss
            # -------------------------------------------------

            g_loss = (
                g_adv
                +
                supervised_loss
                +
                STATISTICS_WEIGHT
                * stat_loss
                +
                CORRELATION_WEIGHT
                * corr_loss
                +
                TEMPORAL_WEIGHT
                * temporal_loss
            )

            generator_optimizer.zero_grad(
                set_to_none=True
            )

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters())
                +
                list(supervisor.parameters())
                +
                list(recovery.parameters()),
                max_norm=5.0,
            )

            generator_optimizer.step()

            # =================================================
            # B. EMBEDDER / RECOVERY UPDATE
            #
            # IMPORTANT:
            # A completely fresh forward pass is used here.
            # This prevents the "backward through graph twice"
            # error from the previous V2 implementation.
            # =================================================

            h_real_reconstruction = embedder(
                real
            )

            reconstructed_real = recovery(
                h_real_reconstruction
            )

            e_loss = mse_loss(
                reconstructed_real,
                real,
            )

            embedder_optimizer.zero_grad(
                set_to_none=True
            )

            e_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(embedder.parameters())
                +
                list(recovery.parameters()),
                max_norm=5.0,
            )

            embedder_optimizer.step()

            # =================================================
            # C. DISCRIMINATOR UPDATE
            #
            # Completely detached fresh forward pass.
            # =================================================

            with torch.no_grad():

                real_h_detached = embedder(
                    real
                )

                z_d = sample_noise(
                    batch_size,
                    SEQ_LENGTH,
                    NOISE_DIM,
                )

                fake_h_initial_detached = generator(
                    z_d
                )

                fake_h_detached = supervisor(
                    fake_h_initial_detached
                )

            real_disc = discriminator(
                real_h_detached.detach()
            )

            fake_disc = discriminator(
                fake_h_detached.detach()
            )

            real_labels = torch.ones_like(
                real_disc
            )

            fake_labels = torch.zeros_like(
                fake_disc
            )

            d_real_loss = adversarial_loss(
                real_disc,
                real_labels,
            )

            d_fake_loss = adversarial_loss(
                fake_disc,
                fake_labels,
            )

            d_loss = (
                d_real_loss
                +
                d_fake_loss
            ) / 2.0

            discriminator_optimizer.zero_grad(
                set_to_none=True
            )

            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0,
            )

            discriminator_optimizer.step()

            # =================================================
            # TRACK
            # =================================================

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()
            epoch_e += e_loss.item()

            epoch_stat += stat_loss.item()
            epoch_corr += corr_loss.item()
            epoch_temporal += temporal_loss.item()

            batches += 1

        # -----------------------------------------------------
        # Average epoch losses
        # -----------------------------------------------------

        epoch_g /= max(
            batches,
            1,
        )

        epoch_d /= max(
            batches,
            1,
        )

        epoch_e /= max(
            batches,
            1,
        )

        epoch_stat /= max(
            batches,
            1,
        )

        epoch_corr /= max(
            batches,
            1,
        )

        epoch_temporal /= max(
            batches,
            1,
        )

        history.append({
            "stage": "joint",
            "epoch": epoch,
            "generator_loss": epoch_g,
            "discriminator_loss": epoch_d,
            "embedder_loss": epoch_e,
            "statistics_loss": epoch_stat,
            "correlation_loss": epoch_corr,
            "temporal_loss": epoch_temporal,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == JOINT_EPOCHS
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"G {epoch_g:.6f} | "
                f"D {epoch_d:.6f} | "
                f"E {epoch_e:.6f} | "
                f"Stat {epoch_stat:.6f} | "
                f"Corr {epoch_corr:.6f} | "
                f"Lag1 {epoch_temporal:.6f}"
            )

    return history


# ============================================================
# SAVE
# ============================================================

def save_model(
    embedder,
    recovery,
    generator,
    supervisor,
    discriminator,
    history,
):

    checkpoint = {

        "model_version":
            "TimeGAN_Stock_V2",

        "institution":
            "Stock Market",

        "data_type":
            "REAL",

        "features":
            FEATURES,

        "feature_dim":
            FEATURE_DIM,

        "seq_len":
            SEQ_LENGTH,

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
            discriminator.state_dict(),
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("MODEL SAVED")
    print("=" * 78)

    print(
        f"\nModel:\n{MODEL_PATH}"
    )

    print(
        f"\nTraining history:\n{HISTORY_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MACROSTRESS-GAN")
    print("STOCK MARKET TIMEGAN V2")
    print("CORRECTED IMPROVED TRAINING")
    print("=" * 78)

    print(
        f"\nProject root    : {PROJECT_ROOT}"
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

    print()
    print("V2 regularization:")
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

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
        NOISE_DIM,
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

    print()
    print("=" * 78)
    print("MODEL PARAMETERS")
    print("=" * 78)

    total_parameters = 0

    for name, model in [
        ("Embedder", embedder),
        ("Recovery", recovery),
        ("Generator", generator),
        ("Supervisor", supervisor),
        ("Discriminator", discriminator),
    ]:

        count = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total_parameters += count

        print(
            f"{name:<18}: {count:,}"
        )

    print(
        f"Total parameters : "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    history = []

    history.extend(
        train_embedder(
            embedder,
            recovery,
            data,
        )
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    history.extend(
        train_supervisor(
            embedder,
            supervisor,
            data,
        )
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    history.extend(
        train_joint(
            embedder,
            recovery,
            generator,
            supervisor,
            discriminator,
            data,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        embedder,
        recovery,
        generator,
        supervisor,
        discriminator,
        history,
    )

    print()
    print("=" * 78)
    print("STOCK TIMEGAN V2 TRAINING COMPLETED")
    print("=" * 78)

    print(
        "\nV1 remains completely untouched."
    )

    print(
        f"\nV2 model:\n{MODEL_PATH}"
    )

    print(
        f"\nTraining history:\n{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()