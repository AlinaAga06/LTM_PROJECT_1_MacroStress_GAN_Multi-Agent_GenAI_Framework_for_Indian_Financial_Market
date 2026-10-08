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
COND_DIM = 9
HIDDEN_DIM = 24
NUM_LAYERS = 2

BATCH_SIZE = 128

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

CONDITIONING_FEATURES = [
    "Regime_Normal",
    "Regime_Elevated_Volatility",
    "Regime_Market_Stress",
    "Regime_Crisis",
    "Mean_Stress_Score",
    "Mean_Volatility_Score",
    "Mean_Movement_Score",
    "Mean_NIFTY_Drawdown",
    "Mean_VIX_Stress"
]

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "timegan",
    "timegan_v34_model.pt"
)

CONDITIONING_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v34",
    "conditioning_vector.npy"
)

CONDITIONING_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v34",
    "conditioning_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic"
)

NPY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v34_sequences.npy"
)

CONDITIONING_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v34_conditioning.npy"
)

CONDITIONING_CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v34_conditioning.csv"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_v34_financial_data.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(
    max(1, os.cpu_count() // 2)
)


# ============================================================
# CONDITIONAL GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        noise_dim=HIDDEN_DIM,
        cond_dim=COND_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=noise_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        noise,
        condition
    ):

        # condition:
        # [batch, cond_dim]
        #
        # Expand to:
        # [batch, seq_len, cond_dim]

        condition_sequence = condition.unsqueeze(1).expand(
            -1,
            noise.size(1),
            -1
        )

        x = torch.cat(
            [noise, condition_sequence],
            dim=2
        )

        h, _ = self.gru(x)

        h = self.linear(h)

        h = self.sigmoid(h)

        return h


# ============================================================
# CONDITIONAL SUPERVISOR
# ============================================================

class Supervisor(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        cond_dim=COND_DIM,
        num_layers=NUM_LAYERS
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
        condition
    ):

        condition_sequence = condition.unsqueeze(1).expand(
            -1,
            x.size(1),
            -1
        )

        x_conditioned = torch.cat(
            [x, condition_sequence],
            dim=2
        )

        h, _ = self.gru(x_conditioned)

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
# CONDITIONING SAMPLING
# ============================================================

def sample_conditioning(
    conditioning_vectors,
    conditioning_metadata,
    num_sequences
):
    """
    Sample historical conditioning vectors with replacement.

    This preserves the empirical regime distribution rather than
    artificially forcing crisis conditions into the general dataset.
    """

    total_available = len(conditioning_vectors)

    if total_available == 0:
        raise ValueError(
            "No conditioning vectors available."
        )

    indices = np.random.choice(
        total_available,
        size=num_sequences,
        replace=True
    )

    sampled_vectors = conditioning_vectors[indices]

    sampled_metadata = conditioning_metadata.iloc[
        indices
    ].copy()

    sampled_metadata.insert(
        0,
        "synthetic_sequence_id",
        np.arange(num_sequences)
    )

    return (
        sampled_vectors.astype(np.float32),
        sampled_metadata
    )


# ============================================================
# VALIDATE CONDITIONING
# ============================================================

def validate_conditioning(
    conditioning_vectors
):

    if conditioning_vectors.ndim != 2:

        raise ValueError(
            "Conditioning vectors must be 2-dimensional."
        )

    if conditioning_vectors.shape[1] != COND_DIM:

        raise ValueError(
            f"Expected conditioning dimension {COND_DIM}, "
            f"got {conditioning_vectors.shape[1]}."
        )

    if not np.isfinite(
        conditioning_vectors
    ).all():

        raise ValueError(
            "Conditioning vectors contain NaN or infinite values."
        )

    # First four dimensions are one-hot regime indicators.
    regime_part = conditioning_vectors[:, :4]

    if not np.allclose(
        regime_part.sum(axis=1),
        1.0,
        atol=1e-5
    ):

        raise ValueError(
            "Invalid regime one-hot conditioning."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("MacroStress-GAN V3.4")
    print("CONDITIONAL SYNTHETIC FINANCIAL DATA GENERATION")
    print("=" * 78)

    print(f"\nModel:")
    print(f"  {MODEL_PATH}")

    print(f"\nSequences:")
    print(f"  {NUM_SEQUENCES}")

    print(f"\nSequence length:")
    print(f"  {SEQ_LEN}")

    print(f"\nFinancial features:")
    print(f"  {FEATURE_DIM}")

    print(f"\nConditioning features:")
    print(f"  {COND_DIM}")

    print(f"\nHidden dimension:")
    print(f"  {HIDDEN_DIM}")

    print(f"\nGRU layers:")
    print(f"  {NUM_LAYERS}")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        CONDITIONING_PATH,
        CONDITIONING_METADATA_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # LOAD CONDITIONING
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("1. LOADING CONDITIONING DATA")
    print("-" * 78)

    conditioning_vectors = np.load(
        CONDITIONING_PATH
    )

    conditioning_metadata = pd.read_csv(
        CONDITIONING_METADATA_PATH
    )

    print(
        f"Conditioning vectors: "
        f"{conditioning_vectors.shape}"
    )

    print(
        f"Conditioning metadata: "
        f"{conditioning_metadata.shape}"
    )

    validate_conditioning(
        conditioning_vectors
    )

    if len(conditioning_vectors) != len(
        conditioning_metadata
    ):

        raise ValueError(
            "Conditioning vector count and metadata count "
            "do not match."
        )

    print("Conditioning validation: PASS")

    # --------------------------------------------------------
    # DISPLAY HISTORICAL REGIME DISTRIBUTION
    # --------------------------------------------------------

    print("\nHistorical conditioning distribution:")

    regime_names = [
        "Normal",
        "Elevated Volatility",
        "Market Stress",
        "Crisis"
    ]

    regime_columns = [
        "Regime_Normal",
        "Regime_Elevated_Volatility",
        "Regime_Market_Stress",
        "Regime_Crisis"
    ]

    for name, column in zip(
        regime_names,
        regime_columns
    ):

        if column in conditioning_metadata.columns:

            count = int(
                conditioning_metadata[column].sum()
            )

        else:

            count = int(
                conditioning_vectors[
                    :,
                    regime_columns.index(column)
                ].sum()
            )

        percentage = (
            count /
            len(conditioning_vectors) *
            100
        )

        print(
            f"  {name:<22} "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # SAMPLE CONDITIONS
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("2. SAMPLING CONDITIONING VECTORS")
    print("-" * 78)

    sampled_conditions, sampled_metadata = (
        sample_conditioning(
            conditioning_vectors,
            conditioning_metadata,
            NUM_SEQUENCES
        )
    )

    print(
        f"Sampled conditions: "
        f"{sampled_conditions.shape}"
    )

    # Verify sampled regime distribution

    sampled_regime_ids = np.argmax(
        sampled_conditions[:, :4],
        axis=1
    )

    print("\nSynthetic conditioning distribution:")

    for regime_id, name in enumerate(
        regime_names
    ):

        count = int(
            np.sum(
                sampled_regime_ids == regime_id
            )
        )

        percentage = (
            count /
            NUM_SEQUENCES *
            100
        )

        print(
            f"  {name:<22} "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("3. LOADING V3.4 CHECKPOINT")
    print("-" * 78)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    print("Checkpoint loaded successfully.")

    if not isinstance(
        checkpoint,
        dict
    ):

        raise ValueError(
            "Unexpected checkpoint format."
        )

    model_version = checkpoint.get(
        "model_version"
    )

    print(
        f"Model version: {model_version}"
    )

    if model_version != "TimeGAN_V3.4":

        raise ValueError(
            f"Expected TimeGAN_V3.4 checkpoint, "
            f"got {model_version}"
        )

    checkpoint_cond_dim = checkpoint.get(
        "cond_dim"
    )

    if checkpoint_cond_dim != COND_DIM:

        raise ValueError(
            f"Checkpoint condition dimension "
            f"is {checkpoint_cond_dim}, "
            f"expected {COND_DIM}."
        )

    # --------------------------------------------------------
    # INITIALIZE MODELS
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("4. INITIALIZING CONDITIONAL TIMEGAN")
    print("-" * 78)

    generator = Generator().to(device)

    supervisor = Supervisor().to(device)

    recovery = Recovery().to(device)

    # --------------------------------------------------------
    # LOAD GENERATOR
    # --------------------------------------------------------

    if "generator_state_dict" in checkpoint:

        generator.load_state_dict(
            checkpoint[
                "generator_state_dict"
            ]
        )

    elif "generator" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator"]
        )

    else:

        raise KeyError(
            "Generator weights not found."
        )

    # --------------------------------------------------------
    # LOAD SUPERVISOR
    # --------------------------------------------------------

    if "supervisor_state_dict" in checkpoint:

        supervisor.load_state_dict(
            checkpoint[
                "supervisor_state_dict"
            ]
        )

    elif "supervisor" in checkpoint:

        supervisor.load_state_dict(
            checkpoint["supervisor"]
        )

    else:

        raise KeyError(
            "Supervisor weights not found."
        )

    # --------------------------------------------------------
    # LOAD RECOVERY
    # --------------------------------------------------------

    if "recovery_state_dict" in checkpoint:

        recovery.load_state_dict(
            checkpoint[
                "recovery_state_dict"
            ]
        )

    elif "recovery" in checkpoint:

        recovery.load_state_dict(
            checkpoint["recovery"]
        )

    else:

        raise KeyError(
            "Recovery weights not found."
        )

    generator.eval()
    supervisor.eval()
    recovery.eval()

    print("Generator: PASS")
    print("Supervisor: PASS")
    print("Recovery: PASS")

    print(
        f"Generator input dimension: "
        f"{generator.gru.input_size}"
    )

    expected_generator_input = (
        HIDDEN_DIM + COND_DIM
    )

    if generator.gru.input_size != (
        expected_generator_input
    ):

        raise ValueError(
            "Generator architecture mismatch."
        )

    print(
        f"Expected generator input: "
        f"{expected_generator_input}"
    )

    # --------------------------------------------------------
    # GENERATE IN BATCHES
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("5. GENERATING SYNTHETIC SEQUENCES")
    print("-" * 78)

    all_generated = []

    for start in range(
        0,
        NUM_SEQUENCES,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            NUM_SEQUENCES
        )

        batch_conditions = torch.from_numpy(
            sampled_conditions[start:end]
        ).to(device)

        batch_size = end - start

        # Same noise distribution used during V3.4 training.
        noise = torch.rand(
            batch_size,
            SEQ_LEN,
            HIDDEN_DIM,
            device=device
        )

        with torch.no_grad():

            # ------------------------------------------------
            # STEP 1
            # Noise + condition
            # ------------------------------------------------

            generated_hidden = generator(
                noise,
                batch_conditions
            )

            # ------------------------------------------------
            # STEP 2
            # Conditional temporal supervision
            # ------------------------------------------------

            supervised_hidden = supervisor(
                generated_hidden,
                batch_conditions
            )

            # ------------------------------------------------
            # STEP 3
            # Recover financial features
            # ------------------------------------------------

            synthetic_batch = recovery(
                supervised_hidden
            )

        synthetic_batch = (
            synthetic_batch
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_generated.append(
            synthetic_batch
        )

        print(
            f"Generated "
            f"{end:>4}/{NUM_SEQUENCES} sequences"
        )

    synthetic_scaled = np.concatenate(
        all_generated,
        axis=0
    )

    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("6. VALIDATING GENERATED DATA")
    print("-" * 78)

    print(
        f"Synthetic shape: "
        f"{synthetic_scaled.shape}"
    )

    expected_shape = (
        NUM_SEQUENCES,
        SEQ_LEN,
        FEATURE_DIM
    )

    if synthetic_scaled.shape != (
        expected_shape
    ):

        raise ValueError(
            f"Unexpected generated shape: "
            f"{synthetic_scaled.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(
        synthetic_scaled
    ).all():

        raise ValueError(
            "Synthetic data contains NaN "
            "or infinite values."
        )

    print(
        f"Minimum scaled value: "
        f"{synthetic_scaled.min():.6f}"
    )

    print(
        f"Maximum scaled value: "
        f"{synthetic_scaled.max():.6f}"
    )

    # Recovery uses sigmoid, therefore output should
    # remain within [0, 1].

    if (
        synthetic_scaled.min() < -1e-6
        or synthetic_scaled.max() > 1.000001
    ):

        raise ValueError(
            "Generated scaled values fall outside "
            "the expected [0, 1] range."
        )

    print(
        "Scaled range validation: PASS"
    )

    # --------------------------------------------------------
    # SAVE SYNTHETIC SEQUENCES
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("7. SAVING OUTPUTS")
    print("-" * 78)

    np.save(
        NPY_OUTPUT,
        synthetic_scaled
    )

    print(
        f"Sequences saved:\n"
        f"{NPY_OUTPUT}"
    )

    # --------------------------------------------------------
    # SAVE CONDITIONING
    # --------------------------------------------------------

    np.save(
        CONDITIONING_OUTPUT,
        sampled_conditions
    )

    print(
        f"Conditioning saved:\n"
        f"{CONDITIONING_OUTPUT}"
    )

    sampled_metadata.to_csv(
        CONDITIONING_CSV_OUTPUT,
        index=False
    )

    print(
        f"Conditioning metadata saved:\n"
        f"{CONDITIONING_CSV_OUTPUT}"
    )

    # --------------------------------------------------------
    # CREATE FLAT CSV
    # --------------------------------------------------------

    rows = []

    for sequence_id in range(
        NUM_SEQUENCES
    ):

        for timestep in range(
            SEQ_LEN
        ):

            row = {
                "sequence_id": sequence_id,
                "timestep": timestep
            }

            for feature_index, feature_name in enumerate(
                FEATURES
            ):

                row[feature_name] = (
                    synthetic_scaled[
                        sequence_id,
                        timestep,
                        feature_index
                    ]
                )

            rows.append(row)

    synthetic_df = pd.DataFrame(
        rows
    )

    synthetic_df.to_csv(
        CSV_OUTPUT,
        index=False
    )

    print(
        f"Financial CSV saved:\n"
        f"{CSV_OUTPUT}"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("V3.4 GENERATION SUCCESSFUL")
    print("=" * 78)

    print(
        f"Sequences generated : "
        f"{NUM_SEQUENCES}"
    )

    print(
        f"Sequence length     : "
        f"{SEQ_LEN}"
    )

    print(
        f"Financial features  : "
        f"{FEATURE_DIM}"
    )

    print(
        f"Conditioning dims   : "
        f"{COND_DIM}"
    )

    print(
        f"NumPy shape         : "
        f"{synthetic_scaled.shape}"
    )

    print(
        f"CSV shape           : "
        f"{synthetic_df.shape}"
    )

    print(
        f"NaN values          : "
        f"{synthetic_df.isna().sum().sum()}"
    )

    inf_count = np.isinf(
        synthetic_df[FEATURES].values
    ).sum()

    print(
        f"Inf values          : {inf_count}"
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(f"  - {feature}")

    print("\nConditioning:")

    for feature in CONDITIONING_FEATURES:
        print(f"  - {feature}")

    print("\nIMPORTANT:")
    print(
        "The generated NumPy/CSV financial values are "
        "QuantileTransformer-scaled values."
    )

    print(
        "Do NOT calculate final financial risk metrics "
        "directly from this CSV."
    )

    print(
        "The validation pipeline must inverse-transform "
        "these values using the V2 scaler."
    )

    print(
        "\nV3.4 generation uses historical "
        "regime/stress conditioning."
    )

    print(
        "The general synthetic dataset preserves the "
        "empirical conditioning distribution."
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()