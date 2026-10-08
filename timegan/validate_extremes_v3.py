"""
MacroStress-GAN V3.2
Extreme Event / Stress Validation

Purpose:
    Evaluate whether synthetic financial sequences reproduce
    historical extreme events and multi-variable stress conditions.

Inputs:
    Real:
        data/processed/timegan_v2_sequences.npy

    Synthetic:
        outputs/synthetic/synthetic_v3_sequences.npy

    Scaler:
        data/processed/timegan_v2_scaler.pkl

Outputs:
    outputs/validation/v3/extreme/
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_sequences.npy"
)

SYNTHETIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic"
    / "synthetic_v3_sequences.npy"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timegan_v2_scaler.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "v3"
    / "extreme"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change",
]

SEQUENCE_LENGTH = 30

# Standard-deviation thresholds
STD_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]


# ============================================================
# HELPERS
# ============================================================

def load_and_inverse_transform(path, scaler):
    """Load scaled sequences and convert back to original space."""

    data = np.load(path)

    print(f"Loaded: {path}")
    print(f"Shape: {data.shape}")
    print(f"Scaled min: {data.min():.6f}")
    print(f"Scaled max: {data.max():.6f}")

    original_shape = data.shape

    flat = data.reshape(-1, data.shape[-1])

    inverse = scaler.inverse_transform(flat)

    inverse = inverse.reshape(original_shape)

    return inverse


def flatten_sequences(data):
    """Flatten sequence data to observation level."""

    return data.reshape(-1, data.shape[-1])


def safe_rate(count, total):
    """Return percentage rate."""

    if total == 0:
        return 0.0

    return float(count / total)


def zscore_using_real(real_flat, synthetic_flat):
    """
    Standardize both datasets using REAL mean/std.

    This is important because synthetic data must be evaluated
    against the historical reference distribution.
    """

    real_mean = np.mean(real_flat, axis=0)
    real_std = np.std(real_flat, axis=0)

    real_std = np.where(real_std == 0, 1.0, real_std)

    real_z = (real_flat - real_mean) / real_std
    synthetic_z = (synthetic_flat - real_mean) / real_std

    return real_z, synthetic_z, real_mean, real_std


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("MacroStress-GAN V3.2 Extreme Event Validation")
print("=" * 80)

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print(f"Scaler type: {type(scaler).__name__}")

print("\nLoading real sequences...")

real = load_and_inverse_transform(
    REAL_PATH,
    scaler
)

print("\nLoading synthetic sequences...")

synthetic = load_and_inverse_transform(
    SYNTHETIC_PATH,
    scaler
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("BASIC INTEGRITY")
print("=" * 80)

print(f"Real shape:      {real.shape}")
print(f"Synthetic shape: {synthetic.shape}")

print(
    f"Real NaN:        {np.isnan(real).sum()}"
)

print(
    f"Synthetic NaN:   {np.isnan(synthetic).sum()}"
)

print(
    f"Real Inf:        {np.isinf(real).sum()}"
)

print(
    f"Synthetic Inf:   {np.isinf(synthetic).sum()}"
)


# ============================================================
# FLATTEN
# ============================================================

real_flat = flatten_sequences(real)
synthetic_flat = flatten_sequences(synthetic)

real_z, synthetic_z, real_mean, real_std = zscore_using_real(
    real_flat,
    synthetic_flat
)


# ============================================================
# FEATURE EXTREME EVENT COUNTS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE EXTREME EVENT FREQUENCIES")
print("=" * 80)

rows = []

for feature_idx, feature in enumerate(FEATURES):

    print(f"\n--- {feature} ---")

    for threshold in STD_THRESHOLDS:

        # Negative tail
        real_negative = (
            real_z[:, feature_idx] <= -threshold
        )

        synthetic_negative = (
            synthetic_z[:, feature_idx] <= -threshold
        )

        # Positive tail
        real_positive = (
            real_z[:, feature_idx] >= threshold
        )

        synthetic_positive = (
            synthetic_z[:, feature_idx] >= threshold
        )

        rn = int(real_negative.sum())
        sn = int(synthetic_negative.sum())

        rp = int(real_positive.sum())
        sp = int(synthetic_positive.sum())

        total_real = len(real_z[:, feature_idx])
        total_synthetic = len(synthetic_z[:, feature_idx])

        row = {
            "feature": feature,
            "threshold_sigma": threshold,

            "real_negative_count": rn,
            "synthetic_negative_count": sn,

            "real_negative_rate": safe_rate(
                rn,
                total_real
            ),

            "synthetic_negative_rate": safe_rate(
                sn,
                total_synthetic
            ),

            "real_positive_count": rp,
            "synthetic_positive_count": sp,

            "real_positive_rate": safe_rate(
                rp,
                total_real
            ),

            "synthetic_positive_rate": safe_rate(
                sp,
                total_synthetic
            ),
        }

        rows.append(row)

        print(
            f"{threshold:.1f}σ | "
            f"negative: real={rn:,} "
            f"synthetic={sn:,} | "
            f"positive: real={rp:,} "
            f"synthetic={sp:,}"
        )


extreme_df = pd.DataFrame(rows)

extreme_df.to_csv(
    OUTPUT_DIR / "feature_extreme_event_rates.csv",
    index=False
)


# ============================================================
# ABSOLUTE EXTREME EVENTS
# ============================================================

print("\n" + "=" * 80)
print("ABSOLUTE EXTREME VALUES")
print("=" * 80)

absolute_rows = []

for feature_idx, feature in enumerate(FEATURES):

    real_values = real_flat[:, feature_idx]
    synthetic_values = synthetic_flat[:, feature_idx]

    real_min_idx = np.argmin(real_values)
    real_max_idx = np.argmax(real_values)

    synthetic_min_idx = np.argmin(synthetic_values)
    synthetic_max_idx = np.argmax(synthetic_values)

    row = {
        "feature": feature,

        "real_min": float(real_values[real_min_idx]),
        "synthetic_min": float(synthetic_values[synthetic_min_idx]),

        "real_max": float(real_values[real_max_idx]),
        "synthetic_max": float(synthetic_values[synthetic_max_idx]),

        "real_abs_max": float(
            np.max(np.abs(real_values))
        ),

        "synthetic_abs_max": float(
            np.max(np.abs(synthetic_values))
        ),
    }

    absolute_rows.append(row)

    print(f"\n{feature}")
    print(
        f"  Real min:       {row['real_min']:.6f}"
    )
    print(
        f"  Synthetic min:  {row['synthetic_min']:.6f}"
    )
    print(
        f"  Real max:       {row['real_max']:.6f}"
    )
    print(
        f"  Synthetic max:  {row['synthetic_max']:.6f}"
    )


absolute_df = pd.DataFrame(absolute_rows)

absolute_df.to_csv(
    OUTPUT_DIR / "absolute_extremes.csv",
    index=False
)


# ============================================================
# JOINT STRESS EVENTS
# ============================================================

print("\n" + "=" * 80)
print("JOINT MARKET STRESS EVENTS")
print("=" * 80)


def event_results(name, real_mask, synthetic_mask):

    real_count = int(real_mask.sum())
    synthetic_count = int(synthetic_mask.sum())

    result = {
        "event": name,

        "real_count": real_count,
        "synthetic_count": synthetic_count,

        "real_rate": safe_rate(
            real_count,
            len(real_mask)
        ),

        "synthetic_rate": safe_rate(
            synthetic_count,
            len(synthetic_mask)
        ),

        "rate_difference": (
            safe_rate(
                synthetic_count,
                len(synthetic_mask)
            )
            -
            safe_rate(
                real_count,
                len(real_mask)
            )
        ),
    }

    print(
        f"{name}: "
        f"real={real_count:,} "
        f"({result['real_rate']:.6%}) | "
        f"synthetic={synthetic_count:,} "
        f"({result['synthetic_rate']:.6%})"
    )

    return result


# Feature indexes
NIFTY = FEATURES.index("NIFTY50_Return")
CRUDE = FEATURES.index("CRUDE_OIL_Return")
USD = FEATURES.index("USD_INR_Return")
VIX = FEATURES.index("INDIA_VIX_Change")
YIELD = FEATURES.index("INDIA_10Y_YIELD_Change")


joint_rows = []


# ------------------------------------------------------------
# NIFTY CRASH + VIX SPIKE
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND VIX >= +2σ",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (real_z[:, VIX] >= 2.0)
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (synthetic_z[:, VIX] >= 2.0)
        )
    )
)


# ------------------------------------------------------------
# NIFTY CRASH + USD STRESS
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND USD_INR >= +2σ",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (real_z[:, USD] >= 2.0)
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (synthetic_z[:, USD] >= 2.0)
        )
    )
)


# ------------------------------------------------------------
# NIFTY CRASH + VIX + USD
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND VIX >= +2σ AND USD_INR >= +2σ",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (real_z[:, VIX] >= 2.0)
            &
            (real_z[:, USD] >= 2.0)
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (synthetic_z[:, VIX] >= 2.0)
            &
            (synthetic_z[:, USD] >= 2.0)
        )
    )
)


# ------------------------------------------------------------
# NIFTY CRASH + VIX OR USD
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND (VIX >= +2σ OR USD_INR >= +2σ)",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (
                (real_z[:, VIX] >= 2.0)
                |
                (real_z[:, USD] >= 2.0)
            )
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (
                (synthetic_z[:, VIX] >= 2.0)
                |
                (synthetic_z[:, USD] >= 2.0)
            )
        )
    )
)


# ------------------------------------------------------------
# NIFTY CRASH + CRUDE SHOCK
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND CRUDE <= -2σ",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (real_z[:, CRUDE] <= -2.0)
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (synthetic_z[:, CRUDE] <= -2.0)
        )
    )
)


# ------------------------------------------------------------
# NIFTY CRASH + YIELD SHOCK
# ------------------------------------------------------------

joint_rows.append(
    event_results(
        "NIFTY <= -2σ AND 10Y <= -2σ",

        (
            (real_z[:, NIFTY] <= -2.0)
            &
            (real_z[:, YIELD] <= -2.0)
        ),

        (
            (synthetic_z[:, NIFTY] <= -2.0)
            &
            (synthetic_z[:, YIELD] <= -2.0)
        )
    )
)


joint_df = pd.DataFrame(joint_rows)

joint_df.to_csv(
    OUTPUT_DIR / "joint_stress_events.csv",
    index=False
)


# ============================================================
# SEQUENCE-LEVEL EXTREME EVENTS
# ============================================================

print("\n" + "=" * 80)
print("SEQUENCE-LEVEL EXTREME EVENTS")
print("=" * 80)


sequence_rows = []


def sequence_event(
    name,
    real_sequence_mask,
    synthetic_sequence_mask
):

    real_count = int(real_sequence_mask.sum())
    synthetic_count = int(synthetic_sequence_mask.sum())

    result = {
        "event": name,

        "real_sequences": real_count,
        "synthetic_sequences": synthetic_count,

        "real_rate": safe_rate(
            real_count,
            len(real_sequence_mask)
        ),

        "synthetic_rate": safe_rate(
            synthetic_count,
            len(synthetic_sequence_mask)
        ),

        "rate_difference": (
            safe_rate(
                synthetic_count,
                len(synthetic_sequence_mask)
            )
            -
            safe_rate(
                real_count,
                len(real_sequence_mask)
            )
        ),
    }

    print(
        f"{name}: "
        f"real={real_count:,} "
        f"({result['real_rate']:.6%}) | "
        f"synthetic={synthetic_count:,} "
        f"({result['synthetic_rate']:.6%})"
    )

    return result


# NIFTY <= -2σ somewhere in sequence
real_nifty_crash = np.any(
    real[:, :, NIFTY] <= -2.0 * real_std[NIFTY],
    axis=1
)

synthetic_nifty_crash = np.any(
    synthetic[:, :, NIFTY] <= -2.0 * real_std[NIFTY],
    axis=1
)

sequence_rows.append(
    sequence_event(
        "Sequence contains NIFTY <= -2σ",
        real_nifty_crash,
        synthetic_nifty_crash
    )
)


# VIX >= +2σ
real_vix_spike = np.any(
    real[:, :, VIX] >= (
        real_mean[VIX] + 2.0 * real_std[VIX]
    ),
    axis=1
)

synthetic_vix_spike = np.any(
    synthetic[:, :, VIX] >= (
        real_mean[VIX] + 2.0 * real_std[VIX]
    ),
    axis=1
)

sequence_rows.append(
    sequence_event(
        "Sequence contains VIX >= +2σ",
        real_vix_spike,
        synthetic_vix_spike
    )
)


# NIFTY crash AND VIX spike in same sequence
real_joint_sequence = (
    real_nifty_crash
    &
    real_vix_spike
)

synthetic_joint_sequence = (
    synthetic_nifty_crash
    &
    synthetic_vix_spike
)

sequence_rows.append(
    sequence_event(
        "Sequence contains NIFTY crash + VIX spike",
        real_joint_sequence,
        synthetic_joint_sequence
    )
)


sequence_df = pd.DataFrame(sequence_rows)

sequence_df.to_csv(
    OUTPUT_DIR / "sequence_extreme_events.csv",
    index=False
)


# ============================================================
# EXTREME COUNTS PER SEQUENCE
# ============================================================

print("\n" + "=" * 80)
print("EXTREME EVENT CLUSTERING")
print("=" * 80)


def sequence_extreme_counts(data, feature_idx, threshold):

    return np.sum(
        (
            np.abs(
                (
                    data[:, :, feature_idx]
                    - real_mean[feature_idx]
                )
                / real_std[feature_idx]
            )
            >= threshold
        ),
        axis=1
    )


cluster_rows = []

for feature_idx, feature in enumerate(FEATURES):

    for threshold in [2.0, 3.0]:

        real_counts = sequence_extreme_counts(
            real,
            feature_idx,
            threshold
        )

        synthetic_counts = sequence_extreme_counts(
            synthetic,
            feature_idx,
            threshold
        )

        row = {
            "feature": feature,
            "threshold_sigma": threshold,

            "real_mean_extremes_per_sequence":
                float(np.mean(real_counts)),

            "synthetic_mean_extremes_per_sequence":
                float(np.mean(synthetic_counts)),

            "real_max_extremes_in_sequence":
                int(np.max(real_counts)),

            "synthetic_max_extremes_in_sequence":
                int(np.max(synthetic_counts)),

            "real_sequences_with_extremes":
                int(np.sum(real_counts > 0)),

            "synthetic_sequences_with_extremes":
                int(np.sum(synthetic_counts > 0)),
        }

        cluster_rows.append(row)

        print(
            f"{feature} | {threshold:.1f}σ | "
            f"mean per sequence: "
            f"real={row['real_mean_extremes_per_sequence']:.4f}, "
            f"synthetic={row['synthetic_mean_extremes_per_sequence']:.4f}"
        )


cluster_df = pd.DataFrame(cluster_rows)

cluster_df.to_csv(
    OUTPUT_DIR / "extreme_event_clustering.csv",
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "model": "MacroStress-GAN V3.2",
    "validation_type": "Extreme Event / Stress Validation",

    "real_shape": list(real.shape),
    "synthetic_shape": list(synthetic.shape),

    "features": FEATURES,

    "real_global_min": float(real.min()),
    "real_global_max": float(real.max()),

    "synthetic_global_min": float(synthetic.min()),
    "synthetic_global_max": float(synthetic.max()),

    "output_directory": str(OUTPUT_DIR),

    "files": [
        "feature_extreme_event_rates.csv",
        "absolute_extremes.csv",
        "joint_stress_events.csv",
        "sequence_extreme_events.csv",
        "extreme_event_clustering.csv",
    ],
}


with open(
    OUTPUT_DIR / "extreme_validation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("EXTREME EVENT VALIDATION COMPLETE")
print("=" * 80)

print(f"\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    if file.is_file():
        print(f"  - {file.name}")

print("\nNo model or preprocessing files were modified.")