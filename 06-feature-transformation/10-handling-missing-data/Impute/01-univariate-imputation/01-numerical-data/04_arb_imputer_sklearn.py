"""
Missing Value Imputation — Scikit-learn
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

    # Median for Age, mean for Fare.
    age_imputer = SimpleImputer(strategy="median")
    fare_imputer = SimpleImputer(strategy="mean")

    transformer = ColumnTransformer(
        transformers=[
            ("age_imputer", age_imputer, ["Age"]),
            ("fare_imputer", fare_imputer, ["Fare"]),
        ],
        remainder="passthrough",
    )

    # Fit only on training data to avoid data leakage.
    transformer.fit(X_train)

    print("\nLearned imputation statistics:")
    print(
        "Age median:",
        transformer.named_transformers_["age_imputer"].statistics_[0],
    )
    print(
        "Fare mean:",
        transformer.named_transformers_["fare_imputer"].statistics_[0],
    )

    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    print("\nTransformed training data:")
    print(X_train_transformed[:5])

    print("\nTransformed test data:")
    print(X_test_transformed[:5])

    print("\nShapes:")
    print("Original X_train:", X_train.shape)
    print("Transformed X_train:", X_train_transformed.shape)
    print("Original X_test:", X_test.shape)
    print("Transformed X_test:", X_test_transformed.shape)


if __name__ == "__main__":
    main()
