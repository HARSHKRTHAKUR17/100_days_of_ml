"""
Random-Sample Imputation
------------------------

Demonstrates random-sample imputation for:

1. Numerical missing values using the Titanic dataset.
2. Categorical missing values using the House Prices dataset.

Random-sample imputation replaces each missing value with a randomly
selected observed value from the same feature.

Important:
- The replacement values are sampled from the training data only.
- The test set is transformed using values sampled from training data.
- A fixed random_state is used so the results are reproducible.
- The original notebook used deprecated chained assignment and
  ``sns.distplot``; both have been updated.
- The notebook's final ``observation['Fare']`` example referenced an
  undefined variable, so it is replaced with a valid reproducible example.

Datasets expected:
    train.csv
    house-train.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


RANDOM_STATE = 2


def random_sample_impute(
    train: pd.Series,
    other: pd.Series,
    random_state: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Random-sample impute missing values in two Series.

    Observed values are sampled from ``train`` only. This prevents
    information from the test set from influencing the imputation.

    Parameters
    ----------
    train:
        Training feature containing missing values.
    other:
        Another feature split, typically the test feature.
    random_state:
        Seed for reproducible sampling.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Imputed training and other Series.
    """
    rng = np.random.default_rng(random_state)

    train_imputed = train.copy()
    other_imputed = other.copy()

    observed_values = train.dropna().to_numpy()

    if len(observed_values) == 0:
        raise ValueError("Training feature contains no observed values.")

    train_missing = train_imputed.isna()
    other_missing = other_imputed.isna()

    train_imputed.loc[train_missing] = rng.choice(
        observed_values,
        size=train_missing.sum(),
        replace=True,
    )

    other_imputed.loc[other_missing] = rng.choice(
        observed_values,
        size=other_missing.sum(),
        replace=True,
    )

    return train_imputed, other_imputed


def demonstrate_titanic_random_imputation() -> None:
    """Random-sample imputation of the Titanic Age feature."""
    df = pd.read_csv(
        "train.csv",
        usecols=["Age", "Fare", "Survived"],
    )

    print("=" * 70)
    print("TITANIC — RANDOM-SAMPLE IMPUTATION")
    print("=" * 70)

    print("
First five rows:")
    print(df.head())

    print("
Missing-value percentage:")
    print(df.isna().mean().mul(100))

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Make explicit copies so that adding engineered columns does not
    # trigger SettingWithCopyWarning.
    X_train = X_train.copy()
    X_test = X_test.copy()

    print("
Training-set shape:", X_train.shape)
    print("Test-set shape:", X_test.shape)
    print("Missing Age values in training:", X_train["Age"].isna().sum())
    print("Missing Age values in test:", X_test["Age"].isna().sum())

    # Keep the original Age and create a separate imputed feature.
    X_train["Age_imputed"], X_test["Age_imputed"] = random_sample_impute(
        X_train["Age"],
        X_test["Age"],
        random_state=RANDOM_STATE,
    )

    print("
Titanic training data after random-sample imputation:")
    print(X_train.head())

    print("
Remaining missing Age_imputed values:")
    print("Train:", X_train["Age_imputed"].isna().sum())
    print("Test:", X_test["Age_imputed"].isna().sum())

    # ---------------------------------------------------------
    # Distribution comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.kdeplot(
        data=X_train,
        x="Age",
        label="Original",
        warn_singular=False,
    )
    sns.kdeplot(
        data=X_train,
        x="Age_imputed",
        label="Imputed",
        warn_singular=False,
    )

    plt.title("Age: Original vs Random-Sample Imputed")
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Variance comparison
    # ---------------------------------------------------------
    print("
Variance comparison:")
    print(f"Original Age variance: {X_train['Age'].var():.4f}")
    print(
        "Random-sample imputed Age variance: "
        f"{X_train['Age_imputed'].var():.4f}"
    )

    # ---------------------------------------------------------
    # Covariance comparison
    # ---------------------------------------------------------
    print("
Covariance matrix:")
    print(X_train[["Fare", "Age", "Age_imputed"]].cov())

    # ---------------------------------------------------------
    # Boxplot comparison
    # ---------------------------------------------------------
    X_train[["Age", "Age_imputed"]].boxplot(figsize=(7, 5))
    plt.title("Age: Original vs Random-Sample Imputed")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Reproducible random sample example
    # ---------------------------------------------------------
    sampled_value = X_train["Age"].dropna().sample(
        1,
        random_state=RANDOM_STATE,
    )

    print("
Example randomly sampled Age value:")
    print(sampled_value.to_numpy())


def demonstrate_categorical_random_imputation() -> None:
    """Random-sample imputation of categorical House Prices features."""
    data = pd.read_csv(
        "house-train.csv",
        usecols=["GarageQual", "FireplaceQu", "SalePrice"],
    )

    print("
" + "=" * 70)
    print("HOUSE PRICES — RANDOM-SAMPLE IMPUTATION")
    print("=" * 70)

    print("
First five rows:")
    print(data.head())

    print("
Missing-value percentage:")
    print(data.isna().mean().mul(100))

    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Keep original columns and create separate imputed columns.
    X_train["GarageQual_imputed"], X_test["GarageQual_imputed"] = (
        random_sample_impute(
            X_train["GarageQual"],
            X_test["GarageQual"],
            random_state=RANDOM_STATE,
        )
    )

    X_train["FireplaceQu_imputed"], X_test["FireplaceQu_imputed"] = (
        random_sample_impute(
            X_train["FireplaceQu"],
            X_test["FireplaceQu"],
            random_state=RANDOM_STATE + 1,
        )
    )

    print("
Sample of training data:")
    print(X_train.sample(5, random_state=RANDOM_STATE))

    print("
Remaining missing values:")
    print(
        X_train[
            ["GarageQual_imputed", "FireplaceQu_imputed"]
        ].isna().sum()
    )

    # ---------------------------------------------------------
    # Compare GarageQual category proportions
    # ---------------------------------------------------------
    garage_original = (
        X_train["GarageQual"]
        .value_counts(normalize=True)
        .rename("original")
    )

    garage_imputed = (
        X_train["GarageQual_imputed"]
        .value_counts(normalize=True)
        .rename("imputed")
    )

    garage_comparison = pd.concat(
        [garage_original, garage_imputed],
        axis=1,
    )

    print("
GarageQual category proportions:")
    print(garage_comparison)

    # ---------------------------------------------------------
    # Compare FireplaceQu category proportions
    # ---------------------------------------------------------
    fireplace_original = (
        X_train["FireplaceQu"]
        .value_counts(normalize=True)
        .rename("original")
    )

    fireplace_imputed = (
        X_train["FireplaceQu_imputed"]
        .value_counts(normalize=True)
        .rename("imputed")
    )

    fireplace_comparison = pd.concat(
        [fireplace_original, fireplace_imputed],
        axis=1,
    )

    print("
FireplaceQu category proportions:")
    print(fireplace_comparison)

    # ---------------------------------------------------------
    # SalePrice distribution by original FireplaceQu category
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    for category in X_train["FireplaceQu"].dropna().unique():
        sns.kdeplot(
            X_train.loc[
                X_train["FireplaceQu"] == category,
                "SalePrice",
            ],
            label=str(category),
            warn_singular=False,
        )

    plt.title("SalePrice Distribution by Original FireplaceQu")
    plt.xlabel("SalePrice")
    plt.ylabel("Density")
    plt.legend(title="FireplaceQu")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # SalePrice distribution by imputed FireplaceQu category
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    for category in X_train["FireplaceQu_imputed"].dropna().unique():
        sns.kdeplot(
            X_train.loc[
                X_train["FireplaceQu_imputed"] == category,
                "SalePrice",
            ],
            label=str(category),
            warn_singular=False,
        )

    plt.title("SalePrice Distribution by Imputed FireplaceQu")
    plt.xlabel("SalePrice")
    plt.ylabel("Density")
    plt.legend(title="FireplaceQu")
    plt.tight_layout()
    plt.show()


def main() -> None:
    demonstrate_titanic_random_imputation()
    demonstrate_categorical_random_imputation()


if __name__ == "__main__":
    main()
