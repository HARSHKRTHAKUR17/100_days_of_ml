"""
Titanic — Feature Discretization with KBinsDiscretizer

This script demonstrates:
1. A baseline Decision Tree using Age and Fare.
2. Quantile-based discretization of Age and Fare.
3. Inspecting the learned bin edges and comparing original vs. discretized values.
4. Evaluating discretization correctly with cross-validation.
5. Comparing different numbers of bins and strategies.

Dataset expected:
    train.csv

Required columns:
    Age, Fare, Survived

The original notebook contained a cross-validation mistake: after creating
X_trf, it cross-validated a DecisionTreeClassifier on the original X instead
of the transformed features. Here the transformed data is evaluated through
a Pipeline so that the bin boundaries are learned separately inside each
cross-validation fold, avoiding data leakage.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10


def build_discretization_pipeline(
    bins: int,
    strategy: str,
) -> Pipeline:
    """Build a leakage-safe discretization + Decision Tree pipeline."""
    discretizer = ColumnTransformer(
        transformers=[
            (
                "age",
                KBinsDiscretizer(
                    n_bins=bins,
                    encode="ordinal",
                    strategy=strategy,
                    random_state=RANDOM_STATE if strategy == "kmeans" else None,
                ),
                ["Age"],
            ),
            (
                "fare",
                KBinsDiscretizer(
                    n_bins=bins,
                    encode="ordinal",
                    strategy=strategy,
                    random_state=RANDOM_STATE if strategy == "kmeans" else None,
                ),
                ["Fare"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("discretizer", discretizer),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def plot_before_after(
    original: pd.Series,
    transformed: np.ndarray,
    feature_name: str,
) -> None:
    """Plot a feature before and after discretization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(original, bins=30)
    axes[0].set_title(f"{feature_name} — Before")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(transformed, bins=np.arange(-0.5, transformed.max() + 1.5, 1))
    axes[1].set_title(f"{feature_name} — After")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    plt.show()


def evaluate_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    bins: int,
    strategy: str,
) -> float:
    """Evaluate a discretization strategy using leakage-safe CV."""
    pipeline = build_discretization_pipeline(bins=bins, strategy=strategy)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    mean_score = scores.mean()

    print(
        f"bins={bins:>2}, strategy={strategy:<8} "
        f"-> mean CV accuracy: {mean_score:.4f}"
    )

    return mean_score


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load and inspect the Titanic data
    # -------------------------------------------------------------------------
    df = pd.read_csv(
        "train.csv",
        usecols=["Age", "Fare", "Survived"],
    )

    # The notebook drops rows with missing values for this demonstration.
    df = df.dropna().copy()

    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    # -------------------------------------------------------------------------
    # 2. Separate features and target
    # -------------------------------------------------------------------------
    X = df[["Age", "Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nFirst two training rows:")
    print(X_train.head(2))

    # -------------------------------------------------------------------------
    # 3. Baseline Decision Tree on the original features
    # -------------------------------------------------------------------------
    baseline_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_clf.fit(X_train, y_train)

    baseline_pred = baseline_clf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    print(f"\nBaseline test accuracy: {baseline_accuracy:.4f}")

    baseline_cv_scores = cross_val_score(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(f"Baseline mean CV accuracy: {baseline_cv_scores.mean():.4f}")

    # -------------------------------------------------------------------------
    # 4. Fit 15-bin quantile discretization on the training set
    # -------------------------------------------------------------------------
    discretizer = ColumnTransformer(
        transformers=[
            (
                "age",
                KBinsDiscretizer(
                    n_bins=15,
                    encode="ordinal",
                    strategy="quantile",
                ),
                ["Age"],
            ),
            (
                "fare",
                KBinsDiscretizer(
                    n_bins=15,
                    encode="ordinal",
                    strategy="quantile",
                ),
                ["Fare"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_trf = discretizer.fit_transform(X_train)
    X_test_trf = discretizer.transform(X_test)

    print("\nAge bin edges:")
    print(discretizer.named_transformers_["age"].bin_edges_[0])

    print("\nFare bin edges:")
    print(discretizer.named_transformers_["fare"].bin_edges_[0])

    # -------------------------------------------------------------------------
    # 5. Compare original values with their bin numbers
    # -------------------------------------------------------------------------
    output = pd.DataFrame(
        {
            "age": X_train["Age"],
            "age_trf": X_train_trf[:, 0],
            "fare": X_train["Fare"],
            "fare_trf": X_train_trf[:, 1],
        },
        index=X_train.index,
    )

    age_edges = discretizer.named_transformers_["age"].bin_edges_[0]
    fare_edges = discretizer.named_transformers_["fare"].bin_edges_[0]

    # Include the rightmost value in the final interval.
    output["age_labels"] = pd.cut(
        X_train["Age"],
        bins=age_edges,
        include_lowest=True,
    )
    output["fare_labels"] = pd.cut(
        X_train["Fare"],
        bins=fare_edges,
        include_lowest=True,
    )

    print("\nOriginal vs. discretized values:")
    print(output.sample(5, random_state=RANDOM_STATE))

    # -------------------------------------------------------------------------
    # 6. Train a Decision Tree on the discretized features
    # -------------------------------------------------------------------------
    discretized_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    discretized_clf.fit(X_train_trf, y_train)

    discretized_pred = discretized_clf.predict(X_test_trf)
    discretized_accuracy = accuracy_score(y_test, discretized_pred)

    print(f"\n15-bin quantile test accuracy: {discretized_accuracy:.4f}")

    # -------------------------------------------------------------------------
    # 7. Correct cross-validation for discretization
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # The notebook originally transformed the complete dataset and then
    # cross-validated the model on X rather than X_trf. The code below fixes
    # that by putting discretization and the classifier inside one Pipeline.
    pipeline = build_discretization_pipeline(
        bins=15,
        strategy="quantile",
    )

    transformed_cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(
        "\n15-bin quantile mean CV accuracy "
        f"(leakage-safe): {transformed_cv_scores.mean():.4f}"
    )

    # -------------------------------------------------------------------------
    # 8. Visualize discretization
    # -------------------------------------------------------------------------
    fitted_transformed = discretizer.transform(X)

    plot_before_after(
        X["Age"],
        fitted_transformed[:, 0],
        "Age",
    )

    plot_before_after(
        X["Fare"],
        fitted_transformed[:, 1],
        "Fare",
    )

    # -------------------------------------------------------------------------
    # 9. Compare bin counts and strategies
    # -------------------------------------------------------------------------
    print("\nDiscretization strategy comparison:")

    results = []

    for strategy in ("uniform", "quantile", "kmeans"):
        for bins in (5, 10, 15, 20):
            score = evaluate_strategy(
                X,
                y,
                bins=bins,
                strategy=strategy,
            )
            results.append(
                {
                    "strategy": strategy,
                    "bins": bins,
                    "mean_cv_accuracy": score,
                }
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values("mean_cv_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    print("\nBest configurations:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
