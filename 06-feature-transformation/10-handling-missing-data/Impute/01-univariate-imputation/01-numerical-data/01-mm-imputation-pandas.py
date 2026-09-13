"""
Missing Value Imputation using Pandas

Demonstrates:
- Mean imputation
- Median imputation
- Calculating statistics only from training data
- Applying the learned values to missing observations

Dataset expected:
    titanic_toy.csv

Columns:
    Age
    Fare
    Family
    Survived
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    # Load dataset
    df = pd.read_csv("titanic_toy.csv")

    print("First five rows:")
    print(df.head())

    print("\nMissing-value percentage:")
    print(df.isna().mean())

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split before calculating imputation values.
    # This prevents data leakage from the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2,
    )

    # Calculate statistics using ONLY the training data
    age_mean = X_train["Age"].mean()
    age_median = X_train["Age"].median()

    fare_mean = X_train["Fare"].mean()
    fare_median = X_train["Fare"].median()

    print("\nTraining-set imputation values:")
    print(f"Age mean: {age_mean:.4f}")
    print(f"Age median: {age_median:.4f}")
    print(f"Fare mean: {fare_mean:.4f}")
    print(f"Fare median: {fare_median:.4f}")

    # Make explicit copies to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Create mean- and median-imputed versions
    X_train["Age_median"] = X_train["Age"].fillna(age_median)
    X_train["Age_mean"] = X_train["Age"].fillna(age_mean)

    X_train["Fare_median"] = X_train["Fare"].fillna(fare_median)
    X_train["Fare_mean"] = X_train["Fare"].fillna(fare_mean)

    # Apply the SAME training statistics to the test set
    X_test["Age_median"] = X_test["Age"].fillna(age_median)
    X_test["Age_mean"] = X_test["Age"].fillna(age_mean)

    X_test["Fare_median"] = X_test["Fare"].fillna(fare_median)
    X_test["Fare_mean"] = X_test["Fare"].fillna(fare_mean)

    print("\nTraining data after imputation:")
    print(
        X_train[
            [
                "Age",
                "Fare",
                "Family",
                "Age_median",
                "Age_mean",
                "Fare_median",
                "Fare_mean",
            ]
        ].head()
    )

    # Compare variance before and after imputation
    print("\nVariance comparison:")

    print(f"Original Age variance: {X_train['Age'].var():.4f}")
    print(
        f"Age variance after median imputation: "
        f"{X_train['Age_median'].var():.4f}"
    )
    print(
        f"Age variance after mean imputation: "
        f"{X_train['Age_mean'].var():.4f}"
    )

    print(f"\nOriginal Fare variance: {X_train['Fare'].var():.4f}")
    print(
        f"Fare variance after median imputation: "
        f"{X_train['Fare_median'].var():.4f}"
    )
    print(
        f"Fare variance after mean imputation: "
        f"{X_train['Fare_mean'].var():.4f}"
    )


if __name__ == "__main__":
    main()