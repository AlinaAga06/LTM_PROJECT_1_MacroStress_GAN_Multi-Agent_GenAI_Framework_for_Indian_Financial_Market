# ============================================================
# TIMEGAN V2 - SYNTHETIC DATA GENERATION
# MacroStress-GAN
#
# IMPORTANT:
# Generator + Supervisor produce latent representations.
# Recovery converts the latent representation back into the
# original 5 financial return/change variables.
#
# Input:
#   models/timegan/timegan_v2_model.pt
#
# Output:
#   outputs/synthetic/synthetic_v2_sequences.npy
#   outputs/synthetic/synthetic_v2_financial_data.csv
#
# Final expected shape:
#   (1000, 30, 5)
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "timegan"
    / "timegan_v2_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
)

SEQUENCES_PATH = (
    OUTPUT_DIR
    / "synthetic_v2_sequences.npy"
)

CSV_PATH = (
    OUTPUT_DIR
    / "synthetic_v2_financial_data.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

FEATURE_DIM = 5

SEQ_LENGTH = 30

NUM_SEQUENCES = 1000

HIDDEN_DIM = 24

NUM_LAYERS = 2

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
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING TIMEGAN V2 MODEL")
    print("=" * 70)


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    print(
        f"Model path: {MODEL_PATH}"
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model_features = checkpoint.get(
        "features",
        FEATURES
    )

    sequence_length = checkpoint.get(
        "sequence_length",
        SEQ_LENGTH
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        HIDDEN_DIM
    )

    num_layers = checkpoint.get(
        "num_layers",
        NUM_LAYERS
    )


    print()
    print("Model information:")

    print(
        f"Version          : "
        f"{checkpoint.get('model_version', 'Unknown')}"
    )

    print(
        f"Sequence length  : {sequence_length}"
    )

    print(
        f"Features         : {model_features}"
    )

    print(
        f"Hidden dimension : {hidden_dim}"
    )

    print(
        f"Number of layers : {num_layers}"
    )


    # ========================================================
    # CREATE MODELS
    # ========================================================

    generator = Generator(
        input_dim=len(model_features),
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(DEVICE)


    supervisor = Supervisor(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(DEVICE)


    recovery = Recovery(
        hidden_dim=hidden_dim,
        output_dim=len(model_features),
        num_layers=num_layers
    ).to(DEVICE)


    # ========================================================
    # LOAD TRAINED WEIGHTS
    # ========================================================

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    supervisor.load_state_dict(
        checkpoint["supervisor_state_dict"]
    )

    recovery.load_state_dict(
        checkpoint["recovery_state_dict"]
    )


    # ========================================================
    # EVALUATION MODE
    # ========================================================

    generator.eval()

    supervisor.eval()

    recovery.eval()


    print()
    print("Generator loaded successfully.")
    print("Supervisor loaded successfully.")
    print("Recovery loaded successfully.")


    return (
        generator,
        supervisor,
        recovery,
        model_features,
        sequence_length
    )


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

def generate_sequences(
    generator,
    supervisor,
    recovery,
    sequence_length,
    feature_dim,
    num_sequences
):

    print()
    print("=" * 70)
    print("GENERATING SYNTHETIC V2 FINANCIAL DATA")
    print("=" * 70)


    print()
    print(
        f"Number of sequences : {num_sequences}"
    )

    print(
        f"Sequence length     : {sequence_length}"
    )

    print(
        f"Latent dimension    : {HIDDEN_DIM}"
    )

    print(
        f"Output features     : {feature_dim}"
    )


    with torch.no_grad():

        # ----------------------------------------------------
        # STEP 1: RANDOM NOISE
        # ----------------------------------------------------

        z = torch.rand(
            num_sequences,
            sequence_length,
            feature_dim,
            device=DEVICE
        )


        print()
        print("Step 1/4: Random noise generated.")


        # ----------------------------------------------------
        # STEP 2: GENERATOR
        # ----------------------------------------------------

        generated_latent = generator(z)


        print(
            f"Step 2/4: Generator output = "
            f"{tuple(generated_latent.shape)}"
        )


        # ----------------------------------------------------
        # STEP 3: SUPERVISOR
        # ----------------------------------------------------

        supervised_latent = supervisor(
            generated_latent
        )


        print(
            f"Step 3/4: Supervisor output = "
            f"{tuple(supervised_latent.shape)}"
        )


        # ----------------------------------------------------
        # STEP 4: RECOVERY
        # ----------------------------------------------------

        synthetic = recovery(
            supervised_latent
        )


        print(
            f"Step 4/4: Recovery output = "
            f"{tuple(synthetic.shape)}"
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        synthetic = (
            synthetic
            .cpu()
            .numpy()
            .astype(np.float32)
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATED DATA VALIDATION")
    print("=" * 70)


    print(
        f"Shape : {synthetic.shape}"
    )


    nan_count = np.isnan(
        synthetic
    ).sum()


    inf_count = np.isinf(
        synthetic
    ).sum()


    print(
        f"NaN count : {nan_count}"
    )

    print(
        f"Inf count : {inf_count}"
    )

    print(
        f"Min       : {synthetic.min():.6f}"
    )

    print(
        f"Max       : {synthetic.max():.6f}"
    )


    # --------------------------------------------------------
    # IMPORTANT SHAPE CHECK
    # --------------------------------------------------------

    expected_shape = (
        num_sequences,
        sequence_length,
        feature_dim
    )


    if synthetic.shape != expected_shape:

        raise ValueError(
            f"\nIncorrect generated shape!\n"
            f"Expected: {expected_shape}\n"
            f"Received: {synthetic.shape}"
        )


    if nan_count > 0:

        raise ValueError(
            "Generated data contains NaN values."
        )


    if inf_count > 0:

        raise ValueError(
            "Generated data contains infinite values."
        )


    return synthetic


# ============================================================
# SAVE NUMPY
# ============================================================

def save_sequences(synthetic):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        SEQUENCES_PATH,
        synthetic
    )


    print()
    print(
        "NumPy sequences saved:"
    )

    print(
        SEQUENCES_PATH
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe(
    synthetic,
    features
):

    print()
    print("=" * 70)
    print("CREATING DATAFRAME")
    print("=" * 70)


    rows = []


    num_sequences = synthetic.shape[0]

    sequence_length = synthetic.shape[1]


    for sequence_id in range(num_sequences):

        for day in range(sequence_length):

            row = {

                "Sequence_ID":
                    sequence_id + 1,

                "Day":
                    day + 1
            }


            for feature_index, feature in enumerate(features):

                row[feature] = float(
                    synthetic[
                        sequence_id,
                        day,
                        feature_index
                    ]
                )


            rows.append(row)


    df = pd.DataFrame(rows)


    return df


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(df):

    df.to_csv(
        CSV_PATH,
        index=False
    )


    print()
    print(
        "CSV saved:"
    )

    print(
        CSV_PATH
    )

    print()
    print(
        f"CSV shape: {df.shape}"
    )


# ============================================================
# DISPLAY STATISTICS
# ============================================================

def display_statistics(
    df,
    features
):

    print()
    print("=" * 70)
    print("SYNTHETIC V2 FINANCIAL STATISTICS")
    print("=" * 70)


    for feature in features:

        values = df[feature]


        print()
        print(feature)

        print(
            f"  Mean : {values.mean():.8f}"
        )

        print(
            f"  Std  : {values.std():.8f}"
        )

        print(
            f"  Min  : {values.min():.8f}"
        )

        print(
            f"  Max  : {values.max():.8f}"
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    df,
    synthetic,
    features
):

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)


    expected_rows = (
        NUM_SEQUENCES
        * SEQ_LENGTH
    )


    print(
        f"Rows           : {len(df):,}"
    )

    print(
        f"Columns        : {len(df.columns)}"
    )

    print(
        f"Expected rows  : {expected_rows:,}"
    )

    print(
        f"Missing values : "
        f"{df[features].isna().sum().sum()}"
    )

    print(
        f"Infinite values: "
        f"{np.isinf(df[features].values).sum()}"
    )


    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if synthetic.shape != (
        NUM_SEQUENCES,
        SEQ_LENGTH,
        FEATURE_DIM
    ):

        raise ValueError(
            "Synthetic sequence shape validation failed."
        )


    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    if len(df) != expected_rows:

        raise ValueError(
            "CSV row count validation failed."
        )


    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    if df[features].isna().sum().sum() != 0:

        raise ValueError(
            "Missing-value validation failed."
        )


    # --------------------------------------------------------
    # Infinite
    # --------------------------------------------------------

    if np.isinf(
        df[features].values
    ).sum() != 0:

        raise ValueError(
            "Infinite-value validation failed."
        )


    print()
    print("=" * 70)
    print("TIMEGAN V2 GENERATION SUCCESSFUL")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TIMEGAN V2 - SYNTHETIC FINANCIAL DATA GENERATION")
    print("=" * 70)


    print()
    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Device       : {DEVICE}"
    )


    # ========================================================
    # LOAD
    # ========================================================

    (
        generator,
        supervisor,
        recovery,
        model_features,
        sequence_length
    ) = load_model()


    # ========================================================
    # GENERATE
    # ========================================================

    synthetic = generate_sequences(
        generator=generator,
        supervisor=supervisor,
        recovery=recovery,
        sequence_length=sequence_length,
        feature_dim=len(model_features),
        num_sequences=NUM_SEQUENCES
    )


    # ========================================================
    # SAVE NUMPY
    # ========================================================

    save_sequences(
        synthetic
    )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = create_dataframe(
        synthetic,
        model_features
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    save_csv(
        df
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    display_statistics(
        df,
        model_features
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_validation(
        df,
        synthetic,
        model_features
    )


    print()
    print("Output files:")

    print(
        f"1. {SEQUENCES_PATH}"
    )

    print(
        f"2. {CSV_PATH}"
    )

    print()
    print(
        "Next step: Validate TimeGAN V2 against the real "
        "returns/change dataset."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()