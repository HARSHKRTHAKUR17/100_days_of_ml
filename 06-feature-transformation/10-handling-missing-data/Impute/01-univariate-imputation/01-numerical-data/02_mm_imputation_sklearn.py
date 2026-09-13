"""
Missing Value Imputation using Scikit-learn

Demonstrates:
- SimpleImputer
- Mean imputation
- Median imputation
- ColumnTransformer
- Fitting imputers only on training data
- Transforming train and test data consistently

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2,
    )

    # Create separate imputers for different columns
    median_imputer = SimpleImputer(strategy="median")
    mean_imputer = SimpleImputer(strategy="mean")

    # Apply:
    # - Median imputation to Age
    # - Mean imputation to Fare
    # - Keep Family unchanged
    transformer = ColumnTransformer(
        transformers=[
            ("age_imputer", median_imputer, ["Age"]),
            ("fare_imputer", mean_imputer, ["Fare"]),
        ],
        remainder="passthrough",
    )

    # Fit ONLY on training data
    transformer.fit(X_train)

    # Inspect learned imputation values
    print("\nLearned imputation values:")

    print(
        "Age median:",
        transformer.named_transformers_["age_imputer"].statistics_[0],
    )

    print(
        "Fare mean:",
        transformer.named_transformers_["fare_imputer"].statistics_[0],
    )

    # Transform training and test data
    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    print("\nTransformed training data:")
    print(X_train_transformed[:5])

    print("\nTransformed test data:")
    print(X_test_transformed[:5])

    print("\nShapes:")
    print("Original X_train:", X_train.shape)
    print("Transformed X_train:", X_train_transformed.shape)

    print("\nOriginal X_test:", X_test.shape)
    print("Transformed X_test:", X_test_transformed.shape)


if __name__ == "__main__":
    main()