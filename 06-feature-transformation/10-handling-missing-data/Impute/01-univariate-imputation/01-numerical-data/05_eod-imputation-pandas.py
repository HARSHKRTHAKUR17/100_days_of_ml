"""
End-of-Distribution (EOD) Imputation
-------------------------------------

End-of-distribution imputation replaces missing values with a value
far away from the normal distribution of the feature.

For a normally distributed feature, a common choice is:

    mean + 3 * standard deviation

This creates an artificial value at the tail/end of the distribution.

Dataset expected:
    titanic_toy.csv

Columns expected:
    Age
    Fare
    Family
    Survived
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the dataset
    # ---------------------------------------------------------
    df = pd.read_csv("titanic_toy.csv")

    print("First five rows:")
    print(df.head())

    print("\nMissing-value fractions:")
    print(df.isna().mean())

    # ---------------------------------------------------------
    # 2. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # ---------------------------------------------------------
    # 3. Train-test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2,
    )

    # Make independent copies so that we can safely add columns.
    X_train = X_train.copy()
    X_test = X_test.copy()

    # ---------------------------------------------------------
    # 4. Calculate statistics from TRAINING data only
    # ---------------------------------------------------------
    age_mean = X_train["Age"].mean()
    age_std = X_train["Age"].std()

    fare_mean = X_train["Fare"].mean()
    fare_std = X_train["Fare"].std()

    print("\nTraining statistics:")
    print(f"Age mean: {age_mean:.4f}")
    print(f"Age std:  {age_std:.4f}")
    print(f"Fare mean: {fare_mean:.4f}")
    print(f"Fare std:  {fare_std:.4f}")

    # ---------------------------------------------------------
    # 5. Calculate end-of-distribution values
    #
    # We use:
    #
    #     mean + 3 * std
    #
    # These values lie far into the right tail of the
    # distribution.
    # ---------------------------------------------------------
    age_eod = age_mean + 3 * age_std
    fare_eod = fare_mean + 3 * fare_std

    print("\nEnd-of-distribution values:")
    print(f"Age EOD value:  {age_eod:.4f}")
    print(f"Fare EOD value: {fare_eod:.4f}")

    # ---------------------------------------------------------
    # 6. Impute missing values
    # ---------------------------------------------------------
    X_train["Age_eod"] = X_train["Age"].fillna(age_eod)
    X_train["Fare_eod"] = X_train["Fare"].fillna(fare_eod)

    # IMPORTANT:
    # Use exactly the same values calculated from X_train
    # when transforming X_test.
    X_test["Age_eod"] = X_test["Age"].fillna(age_eod)
    X_test["Fare_eod"] = X_test["Fare"].fillna(fare_eod)

    # ---------------------------------------------------------
    # 7. Compare variances
    # ---------------------------------------------------------
    print("\nVariance comparison:")

    print(f"Original Age:       {X_train['Age'].var():.4f}")
    print(f"Age after EOD:      {X_train['Age_eod'].var():.4f}")

    print(f"Original Fare:      {X_train['Fare'].var():.4f}")
    print(f"Fare after EOD:     {X_train['Fare_eod'].var():.4f}")

    # ---------------------------------------------------------
    # 8. Compare distributions visually
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(X_train["Age"].dropna(), bins=20)
    axes[0, 0].set_title("Original Age")

    axes[0, 1].hist(X_train["Age_eod"], bins=20)
    axes[0, 1].set_title("Age after EOD Imputation")

    axes[1, 0].hist(X_train["Fare"].dropna(), bins=20)
    axes[1, 0].set_title("Original Fare")

    axes[1, 1].hist(X_train["Fare_eod"], bins=20)
    axes[1, 1].set_title("Fare after EOD Imputation")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 9. Inspect the result
    # ---------------------------------------------------------
    print("\nTraining data after EOD imputation:")
    print(
        X_train[
            [
                "Age",
                "Age_eod",
                "Fare",
                "Fare_eod",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()