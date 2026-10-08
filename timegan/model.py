
"""
MacroStress-GAN
============================================================
TimeGAN Model Architecture

Components:
    1. Embedder
    2. Recovery
    3. Generator
    4. Supervisor
    5. Discriminator

Framework:
    PyTorch

Input shape:
    (batch_size, sequence_length, feature_dim)

For this project:
    sequence_length = 30
    feature_dim = 5

Features:
    NIFTY50
    INDIA_VIX
    CRUDE_OIL
    USD_INR
    INDIA_10Y_YIELD
"""

import torch
import torch.nn as nn


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# EMBEDDER
# ============================================================

class Embedder(nn.Module):
    """
    Embedder network.

    Converts real financial time-series data
    into a lower-dimensional latent representation.

    Input:
        X

    Output:
        H

    X → GRU → GRU → Linear → H
    """

    def __init__(
        self,
        input_dim=5,
        hidden_dim=24,
        num_layers=2
    ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear projection
        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Activation
        self.activation = nn.Sigmoid()


    def forward(self, x):

        # GRU processes the complete time sequence
        output, _ = self.gru(x)

        # Convert GRU output to latent representation
        output = self.fc(output)

        # Keep latent values between 0 and 1
        output = self.activation(output)

        return output


# ============================================================
# RECOVERY
# ============================================================

class Recovery(nn.Module):
    """
    Recovery network.

    Converts latent representation back into
    the original financial feature space.

    H → GRU → GRU → Linear → X_hat
    """

    def __init__(
        self,
        hidden_dim=24,
        output_dim=5,
        num_layers=2
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Convert hidden representation
        # back to five financial variables
        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        # Normalize reconstructed values
        self.activation = nn.Sigmoid()


    def forward(self, h):

        # Process latent sequence
        output, _ = self.gru(h)

        # Convert back to original feature dimension
        output = self.fc(output)

        # Keep output between 0 and 1
        output = self.activation(output)

        return output


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):
    """
    Generator network.

    Takes random noise and generates
    synthetic latent representations.

    Z → GRU → GRU → Linear → E
    """

    def __init__(
        self,
        noise_dim=5,
        hidden_dim=24,
        num_layers=2
    ):

        super().__init__()

        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection
        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Activation
        self.activation = nn.Sigmoid()


    def forward(self, z):

        # Process random noise
        output, _ = self.gru(z)

        # Convert to latent representation
        output = self.fc(output)

        # Normalize latent values
        output = self.activation(output)

        return output


# ============================================================
# SUPERVISOR
# ============================================================

class Supervisor(nn.Module):
    """
    Supervisor network.

    Learns temporal dynamics in the latent space.

    It helps the generator learn:

        H_t → H_(t+1)

    Architecture:

        H → GRU → GRU → Linear → H_next
    """

    def __init__(
        self,
        hidden_dim=24,
        num_layers=2
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Activation
        self.activation = nn.Sigmoid()


    def forward(self, h):

        output, _ = self.gru(h)

        output = self.fc(output)

        output = self.activation(output)

        return output


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):
    """
    Discriminator network.

    Attempts to distinguish:

        Real latent sequences
        vs
        Synthetic latent sequences

    H → GRU → Linear → Probability
    """

    def __init__(
        self,
        hidden_dim=24,
        num_layers=2
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification layer
        self.fc = nn.Linear(
            hidden_dim,
            1
        )


    def forward(self, h):

        # Process sequence
        output, _ = self.gru(h)

        # Use final time step
        final_output = output[:, -1, :]

        # Binary classification score
        output = self.fc(
            final_output
        )

        return output


# ============================================================
# COMPLETE TIMEGAN MODEL
# ============================================================

class TimeGAN(nn.Module):
    """
    Complete TimeGAN model.

    Contains:

        Embedder
        Recovery
        Generator
        Supervisor
        Discriminator
    """

    def __init__(
        self,
        input_dim=5,
        hidden_dim=24,
        noise_dim=5,
        num_layers=2
    ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers

        # ----------------------------------------------------
        # Embedder
        # ----------------------------------------------------

        self.embedder = Embedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Recovery
        # ----------------------------------------------------

        self.recovery = Recovery(
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        self.generator = Generator(
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Supervisor
        # ----------------------------------------------------

        self.supervisor = Supervisor(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------

        self.discriminator = Discriminator(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )


# ============================================================
# MODEL CREATION FUNCTION
# ============================================================

def create_timegan_model(
    input_dim=5,
    hidden_dim=24,
    noise_dim=5,
    num_layers=2
):
    """
    Create and initialize TimeGAN model.

    Returns:
        TimeGAN model
    """

    model = TimeGAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        noise_dim=noise_dim,
        num_layers=num_layers
    )

    # Move model to GPU if available
    model = model.to(DEVICE)

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_summary(model):
    """
    Print model architecture and
    number of trainable parameters.
    """

    print()
    print("=" * 70)
    print("TIMEGAN MODEL SUMMARY")
    print("=" * 70)

    print()
    print(model)

    # Count trainable parameters
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print("=" * 70)


# ============================================================
# TEST MODEL
# ============================================================

def test_model():

    print()
    print("=" * 70)
    print("TESTING TIMEGAN MODEL")
    print("=" * 70)

    # Configuration
    batch_size = 8
    sequence_length = 30
    feature_dim = 5
    hidden_dim = 24
    noise_dim = 5

    # Create model
    model = create_timegan_model(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        noise_dim=noise_dim,
        num_layers=2
    )

    # Display model
    print_model_summary(
        model
    )

    # --------------------------------------------------------
    # Create test real data
    # --------------------------------------------------------

    real_data = torch.rand(
        batch_size,
        sequence_length,
        feature_dim
    ).to(DEVICE)

    # --------------------------------------------------------
    # Create random noise
    # --------------------------------------------------------

    noise = torch.rand(
        batch_size,
        sequence_length,
        noise_dim
    ).to(DEVICE)

    # --------------------------------------------------------
    # Embedder
    # --------------------------------------------------------

    embedded = model.embedder(
        real_data
    )

    print()
    print(
        f"Real data shape      : "
        f"{real_data.shape}"
    )

    print(
        f"Embedded data shape  : "
        f"{embedded.shape}"
    )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    recovered = model.recovery(
        embedded
    )

    print(
        f"Recovered data shape : "
        f"{recovered.shape}"
    )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    generated_latent = model.generator(
        noise
    )

    print(
        f"Generated latent     : "
        f"{generated_latent.shape}"
    )

    # --------------------------------------------------------
    # Supervisor
    # --------------------------------------------------------

    supervised_latent = model.supervisor(
        generated_latent
    )

    print(
        f"Supervisor output    : "
        f"{supervised_latent.shape}"
    )

    # --------------------------------------------------------
    # Recovery of generated data
    # --------------------------------------------------------

    generated_data = model.recovery(
        supervised_latent
    )

    print(
        f"Generated data shape : "
        f"{generated_data.shape}"
    )

    # --------------------------------------------------------
    # Discriminator
    # --------------------------------------------------------

    discriminator_output = model.discriminator(
        embedded
    )

    print(
        f"Discriminator output : "
        f"{discriminator_output.shape}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        feature_dim
    )

    if tuple(recovered.shape) != expected_shape:

        raise RuntimeError(
            "Recovery output shape is incorrect."
        )

    if tuple(generated_data.shape) != expected_shape:

        raise RuntimeError(
            "Generated output shape is incorrect."
        )

    print()
    print("✓ Embedder working")
    print("✓ Recovery working")
    print("✓ Generator working")
    print("✓ Supervisor working")
    print("✓ Discriminator working")
    print("✓ Output dimensions verified")

    print()
    print("=" * 70)
    print("TIMEGAN MODEL TEST PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_model()

