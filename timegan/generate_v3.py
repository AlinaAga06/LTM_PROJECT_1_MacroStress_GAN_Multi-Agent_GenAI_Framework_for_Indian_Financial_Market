import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_SEQUENCES = 1000
SEQ_LEN = 30
FEATURE_DIM = 5
HIDDEN_DIM = 24
NUM_LAYERS = 2

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

MODEL_PATH = "models/timegan/timegan_v3_model.pt"
OUTPUT_DIR = "outputs/synthetic"

NPY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v3_sequences.npy"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v3_financial_data.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(max(1, os.cpu_count() // 2))


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
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
# SUPERVISOR
# ============================================================

class Supervisor(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
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

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# RECOVERY
# ============================================================

class Recovery(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        output_dim=FEATURE_DIM,
        num_layers=NUM_LAYERS
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

    def forward(self, x):

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MacroStress-GAN V3 - Synthetic Data Generation")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Sequences: {NUM_SEQUENCES}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Features: {FEATURE_DIM}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Layers: {NUM_LAYERS}")

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nV3 model not found:\n{MODEL_PATH}\n\n"
            "Make sure V3 training completed successfully."
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    print("\nCheckpoint loaded successfully.")

    # --------------------------------------------------------
    # INITIALIZE MODELS
    # --------------------------------------------------------

    generator = Generator().to(device)
    supervisor = Supervisor().to(device)
    recovery = Recovery().to(device)

    # --------------------------------------------------------
    # LOAD MODEL WEIGHTS
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        # Expected training checkpoint format
        if "generator_state_dict" in checkpoint:
            generator.load_state_dict(
                checkpoint["generator_state_dict"]
            )

        elif "generator" in checkpoint:
            generator.load_state_dict(
                checkpoint["generator"]
            )

        else:
            raise KeyError(
                "Generator weights were not found in V3 checkpoint."
            )

        if "supervisor_state_dict" in checkpoint:
            supervisor.load_state_dict(
                checkpoint["supervisor_state_dict"]
            )

        elif "supervisor" in checkpoint:
            supervisor.load_state_dict(
                checkpoint["supervisor"]
            )

        else:
            raise KeyError(
                "Supervisor weights were not found in V3 checkpoint."
            )

        if "recovery_state_dict" in checkpoint:
            recovery.load_state_dict(
                checkpoint["recovery_state_dict"]
            )

        elif "recovery" in checkpoint:
            recovery.load_state_dict(
                checkpoint["recovery"]
            )

        else:
            raise KeyError(
                "Recovery weights were not found in V3 checkpoint."
            )

    else:

        raise ValueError(
            "Unexpected V3 checkpoint format."
        )

    generator.eval()
    supervisor.eval()
    recovery.eval()

    print("Generator loaded.")
    print("Supervisor loaded.")
    print("Recovery loaded.")

    # --------------------------------------------------------
    # GENERATE LATENT REPRESENTATIONS
    # --------------------------------------------------------

    print("\nGenerating synthetic sequences...")

    noise = torch.rand(
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM,
        device=device
    )

    with torch.no_grad():

        # Step 1: Random noise
        generated_hidden = generator(noise)

        # Step 2: Temporal supervision
        supervised_hidden = supervisor(
            generated_hidden
        )

        # Step 3: Recover original feature space
        synthetic_scaled = recovery(
            supervised_hidden
        )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    synthetic_scaled = (
        synthetic_scaled
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    print(
        f"Synthetic tensor shape: {synthetic_scaled.shape}"
    )

    # --------------------------------------------------------
    # VALIDATION OF BASIC OUTPUT
    # --------------------------------------------------------

    if np.isnan(synthetic_scaled).any():

        raise ValueError(
            "Generated data contains NaN values."
        )

    if np.isinf(synthetic_scaled).any():

        raise ValueError(
            "Generated data contains infinite values."
        )

    print(
        f"Min value: {synthetic_scaled.min():.6f}"
    )

    print(
        f"Max value: {synthetic_scaled.max():.6f}"
    )

    # --------------------------------------------------------
    # SAVE NUMPY SEQUENCES
    # --------------------------------------------------------

    np.save(
        NPY_OUTPUT,
        synthetic_scaled
    )

    print(
        f"\nSaved sequence file:\n{NPY_OUTPUT}"
    )

    # --------------------------------------------------------
    # CREATE FLAT CSV
    # --------------------------------------------------------

    rows = []

    for sequence_id in range(NUM_SEQUENCES):

        for timestep in range(SEQ_LEN):

            row = {
                "sequence_id": sequence_id,
                "timestep": timestep
            }

            for feature_index, feature_name in enumerate(FEATURES):

                row[feature_name] = synthetic_scaled[
                    sequence_id,
                    timestep,
                    feature_index
                ]

            rows.append(row)

    synthetic_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    synthetic_df.to_csv(
        CSV_OUTPUT,
        index=False
    )

    print(
        f"Saved CSV file:\n{CSV_OUTPUT}"
    )

    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V3 GENERATION SUCCESSFUL")
    print("=" * 70)

    print(
        f"Sequences generated : {NUM_SEQUENCES}"
    )

    print(
        f"Sequence length     : {SEQ_LEN}"
    )

    print(
        f"Features            : {FEATURE_DIM}"
    )

    print(
        f"NumPy shape         : {synthetic_scaled.shape}"
    )

    print(
        f"CSV shape           : {synthetic_df.shape}"
    )

    print(
        f"NaN values          : {synthetic_df.isna().sum().sum()}"
    )

    print(
        f"Inf values          : "
        f"{np.isinf(synthetic_df[FEATURES].values).sum()}"
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(f"  - {feature}")

    print("\nIMPORTANT:")
    print(
        "The generated values are MinMax-scaled values."
    )

    print(
        "Do NOT calculate final financial risk metrics "
        "directly from this CSV."
    )

    print(
        "The V3 validation script must inverse-transform "
        "the values using the V2/V3 scaler."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
