"""
Missing Value Imputation — Pandas
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv("titanic_toy.csv")

    print("First five rows:")
    print(df.head())

    print("\nMissing-value fractions:")
    print(df.isna().mean())

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Calculate statistics from training data only.
    mean_age = X_train["Age"].mean()
    median_age = X_train["Age"].median()
    mean_fare = X_train["Fare"].mean()
    median_fare = X_train["Fare"].median()

    # Avoid SettingWithCopyWarning.
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Arbitrary-value imputation.
    X_train["Age_99"] = X_train["Age"].fillna(99)
    X_train["Age_minus1"] = X_train["Age"].fillna(-1)
    X_train["Fare_999"] = X_train["Fare"].fillna(999)
    X_train["Fare_minus1"] = X_train["Fare"].fillna(-1)

    print("\nVariance after arbitrary-value imputation:")
    print(f"Original Age: {X_train['Age'].var():.4f}")
    print(f"Age after 99: {X_train['Age_99'].var():.4f}")
    print(f"Age after -1: {X_train['Age_minus1'].var():.4f}")
    print(f"Original Fare: {X_train['Fare'].var():.4f}")
    print(f"Fare after 999: {X_train['Fare_999'].var():.4f}")
    print(f"Fare after -1: {X_train['Fare_minus1'].var():.4f}")

    # Mean/median imputation.
    X_train["Age_median"] = X_train["Age"].fillna(median_age)
    X_train["Age_mean"] = X_train["Age"].fillna(mean_age)
    X_train["Fare_median"] = X_train["Fare"].fillna(median_fare)
    X_train["Fare_mean"] = X_train["Fare"].fillna(mean_fare)

    # Use training statistics on the test set.
    X_test["Age_median"] = X_test["Age"].fillna(median_age)
    X_test["Age_mean"] = X_test["Age"].fillna(mean_age)
    X_test["Fare_median"] = X_test["Fare"].fillna(median_fare)
    X_test["Fare_mean"] = X_test["Fare"].fillna(mean_fare)

    print("\nTraining data after imputation:")
    print(X_train.head())


if __name__ == "__main__":
    main()
