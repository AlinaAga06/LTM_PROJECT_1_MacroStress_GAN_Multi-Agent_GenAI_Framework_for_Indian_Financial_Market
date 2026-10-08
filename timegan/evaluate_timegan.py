import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import wasserstein_distance


# ============================================================
# 1. DISCRIMINATIVE SCORE
# ============================================================

def discriminative_score(real_data, synthetic_data):

    # Flatten each time-series sample
    real_flat = real_data.reshape(real_data.shape[0], -1)
    synthetic_flat = synthetic_data.reshape(synthetic_data.shape[0], -1)

    X = np.vstack([real_flat, synthetic_flat])

    # Real = 0, Synthetic = 1
    y = np.concatenate([
        np.zeros(len(real_flat)),
        np.ones(len(synthetic_flat))
    ])

    # Train discriminator
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    # Ideal discriminator accuracy = 50%
    score = abs(accuracy - 0.5)

    return accuracy, score


# ============================================================
# 2. PREDICTIVE SCORE
# ============================================================

def predictive_score(real_data, synthetic_data):

    # Use synthetic data for training
    X_train = synthetic_data[:, :-1, :]
    y_train = synthetic_data[:, 1:, :]

    # Flatten
    X_train = X_train.reshape(
        X_train.shape[0] * X_train.shape[1],
        X_train.shape[2]
    )

    y_train = y_train.reshape(
        y_train.shape[0] * y_train.shape[1],
        y_train.shape[2]
    )

    # Train predictor
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test on real data
    X_test = real_data[:, :-1, :]
    y_test = real_data[:, 1:, :]

    X_test = X_test.reshape(
        X_test.shape[0] * X_test.shape[1],
        X_test.shape[2]
    )

    y_test = y_test.reshape(
        y_test.shape[0] * y_test.shape[1],
        y_test.shape[2]
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    return mae


# ============================================================
# 3. CORRELATION SIMILARITY
# ============================================================

def correlation_similarity(real_data, synthetic_data):

    # Combine samples and time steps
    real_flat = real_data.reshape(
        -1,
        real_data.shape[-1]
    )

    synthetic_flat = synthetic_data.reshape(
        -1,
        synthetic_data.shape[-1]
    )

    # Correlation matrices
    real_corr = np.corrcoef(
        real_flat,
        rowvar=False
    )

    synthetic_corr = np.corrcoef(
        synthetic_flat,
        rowvar=False
    )

    # Mean absolute difference
    difference = np.mean(
        np.abs(real_corr - synthetic_corr)
    )

    similarity = 1 - difference

    return similarity, real_corr, synthetic_corr


# ============================================================
# 4. DISTRIBUTION SIMILARITY
# ============================================================

def distribution_similarity(real_data, synthetic_data):

    real_flat = real_data.reshape(
        -1,
        real_data.shape[-1]
    )

    synthetic_flat = synthetic_data.reshape(
        -1,
        synthetic_data.shape[-1]
    )

    distances = []

    for feature in range(real_flat.shape[1]):

        distance = wasserstein_distance(
            real_flat[:, feature],
            synthetic_flat[:, feature]
        )

        distances.append(distance)

    return np.mean(distances), distances


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Replace these with your actual TimeGAN arrays
    # --------------------------------------------------------

    np.random.seed(42)

    real_data = np.random.rand(
        100,
        30,
        5
    )

    synthetic_data = np.random.rand(
        100,
        30,
        5
    )

    # --------------------------------------------------------
    # Discriminative Score
    # --------------------------------------------------------

    discriminator_accuracy, ds = discriminative_score(
        real_data,
        synthetic_data
    )

    # --------------------------------------------------------
    # Predictive Score
    # --------------------------------------------------------

    ps = predictive_score(
        real_data,
        synthetic_data
    )

    # --------------------------------------------------------
    # Correlation Similarity
    # --------------------------------------------------------

    cs, real_corr, synthetic_corr = correlation_similarity(
        real_data,
        synthetic_data
    )

    # --------------------------------------------------------
    # Distribution Similarity
    # --------------------------------------------------------

    wd, feature_distances = distribution_similarity(
        real_data,
        synthetic_data
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("=" * 60)
    print("TIMEGAN EVALUATION")
    print("=" * 60)

    print(f"\nDiscriminator Accuracy : {discriminator_accuracy:.4f}")
    print(f"Discriminative Score   : {ds:.4f}")

    print(f"\nPredictive Score       : {ps:.4f}")

    print(f"\nCorrelation Similarity : {cs:.4f}")

    print(f"\nWasserstein Distance   : {wd:.4f}")

    print("\nFeature-wise Wasserstein Distances:")

    feature_names = [
        "NIFTY50",
        "CRUDE_OIL",
        "USD_INR",
        "INDIA_VIX",
        "INDIA_10Y_YIELD"
    ]

    for name, distance in zip(
        feature_names,
        feature_distances
    ):
        print(
            f"  {name:<20} : {distance:.4f}"
        )

    print("\n" + "=" * 60)