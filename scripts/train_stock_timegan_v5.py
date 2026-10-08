# ============================================================
# MACROSTRESS-GAN
# STOCK MARKET TIMEGAN V5 TRAINING
# ============================================================
#
# V5 is based on the successful V2 architecture.
#
# Main goals:
#   - Preserve V2's correlation and temporal behavior
#   - Reduce variance collapse
#   - Improve distribution matching
#   - Improve quantile coverage
#   - Avoid the collapse observed in V3/V4
#
# IMPORTANT:
#   V1, V2, V3 and V4 are NOT modified.
#
# Input:
#   data/processed/institutions/stock/stock_timegan_sequences.npy
#
# Output:
#   models/timegan/stock/timegan_stock_v5.pt
#   models/timegan/stock/training_history_stock_v5.csv
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

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
# V5 REGULARIZATION
# ------------------------------------------------------------
#
# Keep these controlled.
#
# V4 used strong tail/quantile objectives and collapsed.
# V5 uses moderate distribution constraints.
# ------------------------------------------------------------

STATISTICS_WEIGHT = 1.0
CORRELATION_WEIGHT = 1.0
TEMPORAL_WEIGHT = 1.0
VARIANCE_WEIGHT = 0.75
QUANTILE_WEIGHT = 0.50

# Small tail weight.
# This is intentionally NOT aggressive.
TAIL_WEIGHT = 0.10

GRADIENT_CLIP = 1.0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "institutions",
    "stock",
    "stock_timegan_sequences.npy",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "stock",
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "timegan_stock_v5.pt",
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "training_history_stock_v5.csv",
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
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
# PRINT HEADER
# ============================================================

print("\n")
print("=" * 78)
print("MACROSTRESS-GAN")
print("STOCK MARKET TIMEGAN V5 TRAINING")
print("=" * 78)

print(f"\nDevice: {DEVICE}")

print("\nConfiguration:")
print(f"Sequence length       : {SEQ_LEN}")
print(f"Feature dimension     : {FEATURE_DIM}")
print(f"Hidden dimension      : {HIDDEN_DIM}")
print(f"GRU layers            : {NUM_LAYERS}")
print(f"Batch size            : {BATCH_SIZE}")
print(f"Embedder epochs       : {EMBEDDER_EPOCHS}")
print(f"Supervisor epochs     : {SUPERVISOR_EPOCHS}")
print(f"Joint epochs          : {JOINT_EPOCHS}")
print(f"Learning rate         : {LEARNING_RATE}")

print("\nV5 regularization:")
print(f"Statistics weight     : {STATISTICS_WEIGHT}")
print(f"Correlation weight    : {CORRELATION_WEIGHT}")
print(f"Temporal weight      : {TEMPORAL_WEIGHT}")
print(f"Variance weight      : {VARIANCE_WEIGHT}")
print(f"Quantile weight      : {QUANTILE_WEIGHT}")
print(f"Tail weight          : {TAIL_WEIGHT}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n")
print("=" * 78)
print("LOADING STOCK TIMEGAN DATA")
print("=" * 78)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

data = np.load(INPUT_FILE).astype(np.float32)

print(f"\nInput file:")
print(INPUT_FILE)

print(f"\nData shape: {data.shape}")

if data.ndim != 3:
    raise ValueError(
        f"Expected 3D array, received {data.ndim}D."
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

if np.isnan(data).any():
    raise ValueError(
        "Input data contains NaN values."
    )

if np.isinf(data).any():
    raise ValueError(
        "Input data contains Inf values."
    )

print(f"Minimum: {data.min():.8f}")
print(f"Maximum: {data.max():.8f}")
print(f"NaN count: {np.isnan(data).sum()}")
print(f"Inf count: {np.isinf(data).sum()}")


# ============================================================
# DATASET
# ============================================================

dataset = torch.tensor(
    data,
    dtype=torch.float32
)

num_samples = len(dataset)

print(f"\nNumber of sequences: {num_samples}")


# ============================================================
# MODEL COMPONENTS
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
            batch_first=True,
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
# CREATE MODELS
# ============================================================

print("\n")
print("=" * 78)
print("CREATING V5 MODEL COMPONENTS")
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
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


total_params = sum([
    count_parameters(embedder),
    count_parameters(recovery),
    count_parameters(generator),
    count_parameters(supervisor),
    count_parameters(discriminator),
])

print(
    f"\nTotal trainable parameters: {total_params:,}"
)


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_embedder = optim.Adam(
    list(embedder.parameters()) +
    list(recovery.parameters()),
    lr=LEARNING_RATE,
)

optimizer_supervisor = optim.Adam(
    supervisor.parameters(),
    lr=LEARNING_RATE,
)

optimizer_generator = optim.Adam(
    list(generator.parameters()) +
    list(supervisor.parameters()),
    lr=LEARNING_RATE,
)

optimizer_discriminator = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# LOSSES
# ============================================================

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()


# ============================================================
# HELPER: GRADIENT CLIPPING
# ============================================================

def clip_gradients(models):

    parameters = []

    for model in models:

        parameters.extend(
            list(model.parameters())
        )

    torch.nn.utils.clip_grad_norm_(
        parameters,
        GRADIENT_CLIP
    )


# ============================================================
# HELPER: STATISTICS LOSS
# ============================================================

def statistics_loss(real, fake):

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
# HELPER: VARIANCE LOSS
# ============================================================

def variance_loss(real, fake):

    real_var = torch.var(
        real,
        dim=(0, 1),
        unbiased=False
    )

    fake_var = torch.var(
        fake,
        dim=(0, 1),
        unbiased=False
    )

    return torch.mean(
        torch.abs(
            real_var - fake_var
        )
    )


# ============================================================
# HELPER: CORRELATION LOSS
# ============================================================

def correlation_loss(real, fake):

    # Flatten sequence and batch dimensions.
    real_flat = real.reshape(
        -1,
        FEATURE_DIM
    )

    fake_flat = fake.reshape(
        -1,
        FEATURE_DIM
    )

    real_centered = (
        real_flat -
        real_flat.mean(dim=0, keepdim=True)
    )

    fake_centered = (
        fake_flat -
        fake_flat.mean(dim=0, keepdim=True)
    )

    real_std = (
        torch.sqrt(
            torch.mean(
                real_centered ** 2,
                dim=0
            ) + 1e-8
        )
    )

    fake_std = (
        torch.sqrt(
            torch.mean(
                fake_centered ** 2,
                dim=0
            ) + 1e-8
        )
    )

    real_norm = (
        real_centered /
        real_std
    )

    fake_norm = (
        fake_centered /
        fake_std
    )

    real_corr = (
        real_norm.T @ real_norm
    ) / (
        real_norm.shape[0] - 1
    )

    fake_corr = (
        fake_norm.T @ fake_norm
    ) / (
        fake_norm.shape[0] - 1
    )

    return torch.mean(
        torch.abs(
            real_corr - fake_corr
        )
    )


# ============================================================
# HELPER: TEMPORAL LOSS
# ============================================================

def temporal_loss(real, fake):

    real_diff = (
        real[:, 1:, :] -
        real[:, :-1, :]
    )

    fake_diff = (
        fake[:, 1:, :] -
        fake[:, :-1, :]
    )

    real_diff_mean = torch.mean(
        real_diff,
        dim=(0, 1)
    )

    fake_diff_mean = torch.mean(
        fake_diff,
        dim=(0, 1)
    )

    real_diff_std = torch.std(
        real_diff,
        dim=(0, 1)
    )

    fake_diff_std = torch.std(
        fake_diff,
        dim=(0, 1)
    )

    return (
        torch.mean(
            torch.abs(
                real_diff_mean -
                fake_diff_mean
            )
        )
        +
        torch.mean(
            torch.abs(
                real_diff_std -
                fake_diff_std
            )
        )
    )


# ============================================================
# HELPER: QUANTILE LOSS
# ============================================================

def quantile_loss(real, fake):

    quantiles = [
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ]

    real_flat = real.reshape(
        -1,
        FEATURE_DIM
    )

    fake_flat = fake.reshape(
        -1,
        FEATURE_DIM
    )

    total = 0.0

    for q in quantiles:

        real_q = torch.quantile(
            real_flat,
            q,
            dim=0
        )

        fake_q = torch.quantile(
            fake_flat,
            q,
            dim=0
        )

        total = total + torch.mean(
            torch.abs(
                real_q - fake_q
            )
        )

    return total / len(quantiles)


# ============================================================
# HELPER: MODERATE TAIL LOSS
# ============================================================

def tail_loss(real, fake):

    real_flat = real.reshape(
        -1,
        FEATURE_DIM
    )

    fake_flat = fake.reshape(
        -1,
        FEATURE_DIM
    )

    real_low = torch.quantile(
        real_flat,
        0.05,
        dim=0
    )

    real_high = torch.quantile(
        real_flat,
        0.95,
        dim=0
    )

    fake_low = torch.quantile(
        fake_flat,
        0.05,
        dim=0
    )

    fake_high = torch.quantile(
        fake_flat,
        0.95,
        dim=0
    )

    low_loss = torch.mean(
        torch.abs(
            real_low - fake_low
        )
    )

    high_loss = torch.mean(
        torch.abs(
            real_high - fake_high
        )
    )

    return low_loss + high_loss


# ============================================================
# HELPER: REAL BATCH
# ============================================================

def get_batch():

    indices = np.random.randint(
        0,
        num_samples,
        size=BATCH_SIZE
    )

    batch = dataset[
        indices
    ].to(DEVICE)

    return batch


# ============================================================
# STAGE 1
# EMBEDDER + RECOVERY
# ============================================================

print("\n")
print("=" * 78)
print("STAGE 1: EMBEDDER + RECOVERY")
print("=" * 78)

stage1_history = []

for epoch in range(
    1,
    EMBEDDER_EPOCHS + 1
):

    real_batch = get_batch()

    optimizer_embedder.zero_grad()

    hidden = embedder(
        real_batch
    )

    reconstructed = recovery(
        hidden
    )

    loss = mse_loss(
        real_batch,
        reconstructed
    )

    loss.backward()

    clip_gradients([
        embedder,
        recovery
    ])

    optimizer_embedder.step()

    stage1_history.append(
        float(loss.item())
    )

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EMBEDDER_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch:3d}/{EMBEDDER_EPOCHS} "
            f"| Reconstruction Loss: "
            f"{loss.item():.6f}"
        )


# ============================================================
# STAGE 2
# SUPERVISOR
# ============================================================

print("\n")
print("=" * 78)
print("STAGE 2: SUPERVISOR")
print("=" * 78)

stage2_history = []

for epoch in range(
    1,
    SUPERVISOR_EPOCHS + 1
):

    real_batch = get_batch()

    with torch.no_grad():

        hidden = embedder(
            real_batch
        )

    optimizer_supervisor.zero_grad()

    supervised_hidden = supervisor(
        hidden
    )

    loss = mse_loss(
        supervised_hidden[:, :-1, :],
        hidden[:, 1:, :]
    )

    loss.backward()

    clip_gradients([
        supervisor
    ])

    optimizer_supervisor.step()

    stage2_history.append(
        float(loss.item())
    )

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == SUPERVISOR_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch:3d}/{SUPERVISOR_EPOCHS} "
            f"| Supervised Loss: "
            f"{loss.item():.6f}"
        )


# ============================================================
# STAGE 3
# JOINT ADVERSARIAL TRAINING
# ============================================================

print("\n")
print("=" * 78)
print("STAGE 3: JOINT V5 TRAINING")
print("=" * 78)

print("\nRegularization:")
print(
    f"Statistics={STATISTICS_WEIGHT}, "
    f"Correlation={CORRELATION_WEIGHT}, "
    f"Temporal={TEMPORAL_WEIGHT}, "
    f"Variance={VARIANCE_WEIGHT}, "
    f"Quantile={QUANTILE_WEIGHT}, "
    f"Tail={TAIL_WEIGHT}"
)

stage3_history = []


for epoch in range(
    1,
    JOINT_EPOCHS + 1
):

    # ========================================================
    # REAL DATA
    # ========================================================

    real_batch = get_batch()


    # ========================================================
    # GENERATOR UPDATE
    #
    # IMPORTANT:
    # Fresh forward passes are used.
    # This avoids the autograd graph-reuse error encountered
    # in the first V2 implementation.
    # ========================================================

    optimizer_generator.zero_grad()

    z = torch.rand(
        BATCH_SIZE,
        SEQ_LEN,
        FEATURE_DIM,
        device=DEVICE
    )

    fake_hidden = generator(
        z
    )

    fake_supervised = supervisor(
        fake_hidden
    )

    fake_data = recovery(
        fake_supervised
    )

    # --------------------------------------------------------
    # Adversarial generator loss
    # --------------------------------------------------------

    fake_logits = discriminator(
        fake_supervised
    )

    valid_labels = torch.ones_like(
        fake_logits
    )

    adversarial_loss = bce_loss(
        fake_logits,
        valid_labels
    )

    # --------------------------------------------------------
    # Supervised temporal consistency
    # --------------------------------------------------------

    generator_supervised_loss = mse_loss(
        fake_supervised[:, :-1, :],
        fake_hidden[:, 1:, :]
    )

    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    stat_loss = statistics_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    corr_loss = correlation_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    temp_loss = temporal_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Variance
    # --------------------------------------------------------

    var_loss = variance_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    q_loss = quantile_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Moderate tail constraint
    # --------------------------------------------------------

    t_loss = tail_loss(
        real_batch,
        fake_data
    )

    # --------------------------------------------------------
    # Total generator loss
    # --------------------------------------------------------

    total_generator_loss = (
        adversarial_loss
        +
        100.0 *
        generator_supervised_loss
        +
        STATISTICS_WEIGHT *
        stat_loss
        +
        CORRELATION_WEIGHT *
        corr_loss
        +
        TEMPORAL_WEIGHT *
        temp_loss
        +
        VARIANCE_WEIGHT *
        var_loss
        +
        QUANTILE_WEIGHT *
        q_loss
        +
        TAIL_WEIGHT *
        t_loss
    )

    total_generator_loss.backward()

    clip_gradients([
        generator,
        supervisor
    ])

    optimizer_generator.step()


    # ========================================================
    # DISCRIMINATOR UPDATE
    # ========================================================

    optimizer_discriminator.zero_grad()

    # --------------------------------------------------------
    # Real path
    # --------------------------------------------------------

    with torch.no_grad():

        real_hidden = embedder(
            real_batch
        )

        real_supervised = supervisor(
            real_hidden
        )

    real_logits = discriminator(
        real_hidden.detach()
    )

    real_labels = torch.ones_like(
        real_logits
    )

    real_loss = bce_loss(
        real_logits,
        real_labels
    )

    # --------------------------------------------------------
    # Fake path
    # --------------------------------------------------------

    with torch.no_grad():

        z = torch.rand(
            BATCH_SIZE,
            SEQ_LEN,
            FEATURE_DIM,
            device=DEVICE
        )

        fake_hidden_disc = generator(
            z
        )

        fake_supervised_disc = supervisor(
            fake_hidden_disc
        )

    fake_logits_disc = discriminator(
        fake_supervised_disc.detach()
    )

    fake_labels = torch.zeros_like(
        fake_logits_disc
    )

    fake_loss = bce_loss(
        fake_logits_disc,
        fake_labels
    )

    discriminator_loss = (
        real_loss +
        fake_loss
    )

    discriminator_loss.backward()

    clip_gradients([
        discriminator
    ])

    optimizer_discriminator.step()


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    row = {
        "epoch": epoch,

        "generator_loss":
            float(total_generator_loss.item()),

        "discriminator_loss":
            float(discriminator_loss.item()),

        "adversarial_loss":
            float(adversarial_loss.item()),

        "supervised_loss":
            float(generator_supervised_loss.item()),

        "statistics_loss":
            float(stat_loss.item()),

        "correlation_loss":
            float(corr_loss.item()),

        "temporal_loss":
            float(temp_loss.item()),

        "variance_loss":
            float(var_loss.item()),

        "quantile_loss":
            float(q_loss.item()),

        "tail_loss":
            float(t_loss.item()),
    }

    stage3_history.append(row)


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if (
        epoch == 1
        or epoch % 20 == 0
        or epoch == JOINT_EPOCHS
    ):

        print(
            f"Epoch {epoch:3d}/{JOINT_EPOCHS} "
            f"| G={total_generator_loss.item():.6f} "
            f"D={discriminator_loss.item():.6f} "
            f"Stat={stat_loss.item():.6f} "
            f"Corr={corr_loss.item():.6f} "
            f"Temp={temp_loss.item():.6f} "
            f"Var={var_loss.item():.6f} "
            f"Quant={q_loss.item():.6f} "
            f"Tail={t_loss.item():.6f}"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

print("\n")
print("=" * 78)
print("SAVING TRAINING HISTORY")
print("=" * 78)

history_rows = []

# Stage 1
for epoch, loss in enumerate(
    stage1_history,
    start=1
):

    history_rows.append({
        "stage": "embedder_recovery",
        "epoch": epoch,
        "reconstruction_loss": loss,
    })


# Stage 2
for epoch, loss in enumerate(
    stage2_history,
    start=1
):

    history_rows.append({
        "stage": "supervisor",
        "epoch": epoch,
        "supervised_loss": loss,
    })


# Stage 3
for row in stage3_history:

    history_rows.append({
        "stage": "joint",
        **row,
    })


history_df = pd.DataFrame(
    history_rows
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)

print(
    f"\nTraining history saved to:\n"
    f"{HISTORY_FILE}"
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

print("\n")
print("=" * 78)
print("SAVING V5 CHECKPOINT")
print("=" * 78)


checkpoint = {

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    "model_version":
        "TimeGAN_Stock_V5",

    "institution":
        "Stock Market",

    "data_type":
        "REAL",

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    "feature_dim":
        FEATURE_DIM,

    "seq_len":
        SEQ_LEN,

    "hidden_dim":
        HIDDEN_DIM,

    "num_layers":
        NUM_LAYERS,

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # V5 regularization
    # --------------------------------------------------------

    "statistics_weight":
        STATISTICS_WEIGHT,

    "correlation_weight":
        CORRELATION_WEIGHT,

    "temporal_weight":
        TEMPORAL_WEIGHT,

    "variance_weight":
        VARIANCE_WEIGHT,

    "quantile_weight":
        QUANTILE_WEIGHT,

    "tail_weight":
        TAIL_WEIGHT,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    "seed":
        SEED,

    # --------------------------------------------------------
    # Model state dictionaries
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final training values
    # --------------------------------------------------------

    "final_stage1_loss":
        stage1_history[-1],

    "final_stage2_loss":
        stage2_history[-1],

    "final_generator_loss":
        stage3_history[-1][
            "generator_loss"
        ],

    "final_discriminator_loss":
        stage3_history[-1][
            "discriminator_loss"
        ],

    # --------------------------------------------------------
    # Input information
    # --------------------------------------------------------

    "input_file":
        INPUT_FILE,

    "num_training_sequences":
        num_samples,

    "parameter_count":
        total_params,
}


torch.save(
    checkpoint,
    MODEL_FILE
)

print(
    f"\nV5 checkpoint saved to:\n"
    f"{MODEL_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("TIMEGAN STOCK V5 TRAINING COMPLETED")
print("=" * 78)

print("\nModel:")
print(MODEL_FILE)

print("\nHistory:")
print(HISTORY_FILE)

print("\nFinal losses:")

print(
    f"Stage 1 reconstruction : "
    f"{stage1_history[-1]:.6f}"
)

print(
    f"Stage 2 supervised     : "
    f"{stage2_history[-1]:.6f}"
)

print(
    f"Stage 3 generator      : "
    f"{stage3_history[-1]['generator_loss']:.6f}"
)

print(
    f"Stage 3 discriminator   : "
    f"{stage3_history[-1]['discriminator_loss']:.6f}"
)

print("\n")
print("=" * 78)
print("IMPORTANT")
print("=" * 78)

print(
    "\nV1, V2, V3 and V4 checkpoints were NOT modified."
)

print(
    "\nNext step:"
)

print(
    "Generate V5 synthetic stock sequences and "
    "validate them using the same V1-V4 metrics."
)

print("\n")