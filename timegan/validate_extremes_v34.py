import os
import json
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REAL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v34",
    "timegan_v34_sequences.npy"
)

SYNTHETIC_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "synthetic",
    "synthetic_v34_sequences.npy"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "timegan_v2_scaler.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "validation",
    "v34",
    "extreme"
)

FEATURES = [
    "NIFTY50_Return",
    "CRUDE_OIL_Return",
    "USD_INR_Return",
    "INDIA_VIX_Change",
    "INDIA_10Y_YIELD_Change"
]

NIFTY = 0
CRUDE = 1
USD_INR = 2
VIX = 3
YIELD_10Y = 4

SIGMA_LEVELS = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]


# ============================================================
# HELPER
# ============================================================

def check_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def inverse_transform_sequences(
    sequences,
    scaler
):

    original_shape = sequences.shape

    flat = sequences.reshape(
        -1,
        original_shape[-1]
    )

    inverse = scaler.inverse_transform(
        flat
    )

    return inverse.reshape(
        original_shape
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MacroStress-GAN V3.4")
    print("EXTREME EVENT / STRESS VALIDATION")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("\n1. CHECKING INPUT FILES")
    print("-" * 80)

    for path in [
        REAL_PATH,
        SYNTHETIC_PATH,
        SCALER_PATH
    ]:

        check_file(path)

        print(
            f"PASS: {path}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n2. LOADING DATA")
    print("-" * 80)

    real_scaled = np.load(
        REAL_PATH
    )

    synthetic_scaled = np.load(
        SYNTHETIC_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"Real scaled shape: "
        f"{real_scaled.shape}"
    )

    print(
        f"Synthetic scaled shape: "
        f"{synthetic_scaled.shape}"
    )

    # --------------------------------------------------------
    # SHAPE VALIDATION
    # --------------------------------------------------------

    if real_scaled.ndim != 3:

        raise ValueError(
            "Real data must be 3-dimensional."
        )

    if synthetic_scaled.ndim != 3:

        raise ValueError(
            "Synthetic data must be 3-dimensional."
        )

    if real_scaled.shape[1:] != (
        synthetic_scaled.shape[1:]
    ):

        raise ValueError(
            "Real and synthetic sequence dimensions "
            "do not match."
        )

    if real_scaled.shape[2] != len(FEATURES):

        raise ValueError(
            "Unexpected number of financial features."
        )

    # --------------------------------------------------------
    # INVERSE TRANSFORM
    # --------------------------------------------------------

    print("\n3. INVERSE TRANSFORM")
    print("-" * 80)

    real = inverse_transform_sequences(
        real_scaled,
        scaler
    )

    synthetic = inverse_transform_sequences(
        synthetic_scaled,
        scaler
    )

    print(
        f"Real original shape: "
        f"{real.shape}"
    )

    print(
        f"Synthetic original shape: "
        f"{synthetic.shape}"
    )

    if not np.isfinite(real).all():

        raise ValueError(
            "Real data contains NaN or infinite values."
        )

    if not np.isfinite(synthetic).all():

        raise ValueError(
            "Synthetic data contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # GLOBAL STATISTICS
    # --------------------------------------------------------

    print("\n4. GLOBAL STATISTICS")
    print("-" * 80)

    real_flat = real.reshape(
        -1,
        len(FEATURES)
    )

    synthetic_flat = synthetic.reshape(
        -1,
        len(FEATURES)
    )

    real_mean = real_flat.mean(
        axis=0
    )

    real_std = real_flat.std(
        axis=0
    )

    print("\nFeature means:")

    for i, feature in enumerate(FEATURES):

        print(
            f"{feature:<25} "
            f"real={real_mean[i]: .8f}"
        )

    print("\nFeature standard deviations:")

    for i, feature in enumerate(FEATURES):

        print(
            f"{feature:<25} "
            f"real={real_std[i]: .8f}"
        )

    # --------------------------------------------------------
    # Z-SCORES
    # --------------------------------------------------------

    real_z = (
        real_flat - real_mean
    ) / real_std

    synthetic_z = (
        synthetic_flat - real_mean
    ) / real_std

    # --------------------------------------------------------
    # FEATURE EXTREME EVENT RATES
    # --------------------------------------------------------

    print("\n5. FEATURE EXTREME EVENT RATES")
    print("-" * 80)

    feature_rows = []

    for feature_index, feature in enumerate(
        FEATURES
    ):

        for sigma in SIGMA_LEVELS:

            negative_threshold = -sigma
            positive_threshold = sigma

            real_negative = np.sum(
                real_z[:, feature_index]
                <= negative_threshold
            )

            synthetic_negative = np.sum(
                synthetic_z[:, feature_index]
                <= negative_threshold
            )

            real_positive = np.sum(
                real_z[:, feature_index]
                >= positive_threshold
            )

            synthetic_positive = np.sum(
                synthetic_z[:, feature_index]
                >= positive_threshold
            )

            feature_rows.append({
                "Feature": feature,
                "Sigma": sigma,

                "Real_Negative_Count":
                    int(real_negative),

                "Synthetic_Negative_Count":
                    int(synthetic_negative),

                "Real_Negative_Rate":
                    float(
                        real_negative /
                        len(real_z)
                    ),

                "Synthetic_Negative_Rate":
                    float(
                        synthetic_negative /
                        len(synthetic_z)
                    ),

                "Real_Positive_Count":
                    int(real_positive),

                "Synthetic_Positive_Count":
                    int(synthetic_positive),

                "Real_Positive_Rate":
                    float(
                        real_positive /
                        len(real_z)
                    ),

                "Synthetic_Positive_Rate":
                    float(
                        synthetic_positive /
                        len(synthetic_z)
                    )
            })

    feature_extremes = pd.DataFrame(
        feature_rows
    )

    feature_extremes.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_extreme_event_rates.csv"
        ),
        index=False
    )

    print(
        feature_extremes.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # ABSOLUTE EXTREMES
    # --------------------------------------------------------

    print("\n6. ABSOLUTE EXTREMES")
    print("-" * 80)

    absolute_rows = []

    for feature_index, feature in enumerate(
        FEATURES
    ):

        real_values = real_flat[
            :, feature_index
        ]

        synthetic_values = synthetic_flat[
            :, feature_index
        ]

        row = {
            "Feature": feature,

            "Real_Min":
                float(real_values.min()),

            "Synthetic_Min":
                float(synthetic_values.min()),

            "Real_Max":
                float(real_values.max()),

            "Synthetic_Max":
                float(synthetic_values.max()),

            "Real_Abs_Max":
                float(
                    np.max(
                        np.abs(real_values)
                    )
                ),

            "Synthetic_Abs_Max":
                float(
                    np.max(
                        np.abs(synthetic_values)
                    )
                )
        }

        absolute_rows.append(row)

        print(
            f"{feature:<25} "
            f"Real Min={row['Real_Min']: .8f} | "
            f"Synth Min={row['Synthetic_Min']: .8f} | "
            f"Real Max={row['Real_Max']: .8f} | "
            f"Synth Max={row['Synthetic_Max']: .8f}"
        )

    absolute_df = pd.DataFrame(
        absolute_rows
    )

    absolute_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "absolute_extremes.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # JOINT STRESS EVENTS
    # --------------------------------------------------------

    print("\n7. JOINT STRESS EVENTS")
    print("-" * 80)

    real_z_seq = (
        real - real_mean
    ) / real_std

    synthetic_z_seq = (
        synthetic - real_mean
    ) / real_std

    def count_event(
        data,
        conditions
    ):

        mask = conditions[0]

        for condition in conditions[1:]:

            mask = mask & condition

        return int(
            np.sum(mask)
        )

    joint_definitions = [

        (
            "NIFTY <= -2σ AND VIX >= +2σ",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                real_z_seq[:, :, VIX] >= 2
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                synthetic_z_seq[:, :, VIX] >= 2
            ]
        ),

        (
            "NIFTY <= -2σ AND USD_INR >= +2σ",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                real_z_seq[:, :, USD_INR] >= 2
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                synthetic_z_seq[:, :, USD_INR] >= 2
            ]
        ),

        (
            "NIFTY <= -2σ AND VIX >= +2σ AND USD_INR >= +2σ",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                real_z_seq[:, :, VIX] >= 2,
                real_z_seq[:, :, USD_INR] >= 2
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                synthetic_z_seq[:, :, VIX] >= 2,
                synthetic_z_seq[:, :, USD_INR] >= 2
            ]
        ),

        (
            "NIFTY <= -2σ AND CRUDE <= -2σ",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                real_z_seq[:, :, CRUDE] <= -2
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                synthetic_z_seq[:, :, CRUDE] <= -2
            ]
        ),

        (
            "NIFTY <= -2σ AND 10Y <= -2σ",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                real_z_seq[:, :, YIELD_10Y] <= -2
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                synthetic_z_seq[:, :, YIELD_10Y] <= -2
            ]
        ),

        (
            "NIFTY <= -2σ AND (VIX >= +2σ OR USD_INR >= +2σ)",
            [
                real_z_seq[:, :, NIFTY] <= -2,
                (
                    (real_z_seq[:, :, VIX] >= 2)
                    |
                    (real_z_seq[:, :, USD_INR] >= 2)
                )
            ],
            [
                synthetic_z_seq[:, :, NIFTY] <= -2,
                (
                    (synthetic_z_seq[:, :, VIX] >= 2)
                    |
                    (synthetic_z_seq[:, :, USD_INR] >= 2)
                )
            ]
        )
    ]

    joint_rows = []

    for name, real_conditions, synthetic_conditions in joint_definitions:

        real_count = count_event(
            real_flat
            if False else real_z_seq,
            real_conditions
        )

        synthetic_count = count_event(
            synthetic_z_seq,
            synthetic_conditions
        )

        row = {
            "Event": name,
            "Real_Count": real_count,
            "Synthetic_Count": synthetic_count
        }

        joint_rows.append(row)

        print(
            f"{name:<65} "
            f"Real={real_count:<6} "
            f"Synthetic={synthetic_count}"
        )

    joint_df = pd.DataFrame(
        joint_rows
    )

    joint_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "joint_stress_events.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SEQUENCE LEVEL EXTREMES
    # --------------------------------------------------------

    print("\n8. SEQUENCE-LEVEL EXTREME EVENTS")
    print("-" * 80)

    # IMPORTANT:
    # Thresholds include the real mean.
    # This fixes the threshold bug in the previous V3 validator.

    nifty_crash_real = np.any(
        real[:, :, NIFTY]
        <= (
            real_mean[NIFTY]
            - 2 * real_std[NIFTY]
        ),
        axis=1
    )

    nifty_crash_synthetic = np.any(
        synthetic[:, :, NIFTY]
        <= (
            real_mean[NIFTY]
            - 2 * real_std[NIFTY]
        ),
        axis=1
    )

    vix_spike_real = np.any(
        real[:, :, VIX]
        >= (
            real_mean[VIX]
            + 2 * real_std[VIX]
        ),
        axis=1
    )

    vix_spike_synthetic = np.any(
        synthetic[:, :, VIX]
        >= (
            real_mean[VIX]
            + 2 * real_std[VIX]
        ),
        axis=1
    )

    joint_crisis_real = (
        nifty_crash_real
        & vix_spike_real
    )

    joint_crisis_synthetic = (
        nifty_crash_synthetic
        & vix_spike_synthetic
    )

    sequence_rows = [

        {
            "Event":
                "Sequence contains NIFTY <= -2σ",

            "Real_Count":
                int(nifty_crash_real.sum()),

            "Synthetic_Count":
                int(nifty_crash_synthetic.sum())
        },

        {
            "Event":
                "Sequence contains VIX >= +2σ",

            "Real_Count":
                int(vix_spike_real.sum()),

            "Synthetic_Count":
                int(vix_spike_synthetic.sum())
        },

        {
            "Event":
                "Sequence contains NIFTY crash + VIX spike",

            "Real_Count":
                int(joint_crisis_real.sum()),

            "Synthetic_Count":
                int(joint_crisis_synthetic.sum())
        }
    ]

    sequence_df = pd.DataFrame(
        sequence_rows
    )

    print(
        sequence_df.to_string(
            index=False
        )
    )

    sequence_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sequence_extreme_events.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # EXTREME EVENT CLUSTERING
    # --------------------------------------------------------

    print("\n9. EXTREME EVENT CLUSTERING")
    print("-" * 80)

    clustering_rows = []

    for feature_index, feature in enumerate(
        FEATURES
    ):

        for sigma in [
            2.0,
            3.0
        ]:

            negative_counts_real = np.sum(
                real_z_seq[:, :, feature_index]
                <= -sigma,
                axis=1
            )

            negative_counts_synthetic = np.sum(
                synthetic_z_seq[:, :, feature_index]
                <= -sigma,
                axis=1
            )

            real_nonzero = (
                negative_counts_real > 0
            )

            synthetic_nonzero = (
                negative_counts_synthetic > 0
            )

            clustering_rows.append({

                "Feature":
                    feature,

                "Sigma":
                    sigma,

                "Real_Mean_Extremes_Per_Sequence":
                    float(
                        negative_counts_real.mean()
                    ),

                "Synthetic_Mean_Extremes_Per_Sequence":
                    float(
                        negative_counts_synthetic.mean()
                    ),

                "Real_Max_Extremes_Per_Sequence":
                    int(
                        negative_counts_real.max()
                    ),

                "Synthetic_Max_Extremes_Per_Sequence":
                    int(
                        negative_counts_synthetic.max()
                    ),

                "Real_Sequences_With_Extremes":
                    int(
                        real_nonzero.sum()
                    ),

                "Synthetic_Sequences_With_Extremes":
                    int(
                        synthetic_nonzero.sum()
                    )
            })

    clustering_df = pd.DataFrame(
        clustering_rows
    )

    print(
        clustering_df.to_string(
            index=False
        )
    )

    clustering_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "extreme_event_clustering.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n10. SAVING SUMMARY")
    print("-" * 80)

    summary = {

        "model":
            "MacroStress-GAN V3.4",

        "validation_type":
            "Extreme Event / Stress Validation",

        "real_shape":
            list(real.shape),

        "synthetic_shape":
            list(synthetic.shape),

        "features":
            FEATURES,

        "real_global_min":
            float(real.min()),

        "real_global_max":
            float(real.max()),

        "synthetic_global_min":
            float(synthetic.min()),

        "synthetic_global_max":
            float(synthetic.max()),

        "nifty_sequence_crashes_real":
            int(nifty_crash_real.sum()),

        "nifty_sequence_crashes_synthetic":
            int(nifty_crash_synthetic.sum()),

        "vix_sequence_spikes_real":
            int(vix_spike_real.sum()),

        "vix_sequence_spikes_synthetic":
            int(vix_spike_synthetic.sum()),

        "joint_nifty_vix_real":
            int(joint_crisis_real.sum()),

        "joint_nifty_vix_synthetic":
            int(joint_crisis_synthetic.sum())
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "extreme_validation_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"Summary saved:\n{summary_path}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("V3.4 EXTREME VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f"  - {filename}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()