"""
Titanic Feature Construction — Family Features

Demonstrates:
- Creating a new feature from existing features
- Family size = SibSp + Parch + 1
- Converting family size into categorical groups
- Evaluating the feature construction with Logistic Regression

Dataset expected:
    train.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42


def family_type(family_size: int) -> int:
    """Encode family size as 0=alone, 1=small, 2=large."""
    if family_size == 1:
        return 0
    if 2 <= family_size <= 4:
        return 1
    return 2


def main() -> None:
    df = pd.read_csv(
        "train.csv",
        usecols=["Age", "Pclass", "SibSp", "Parch", "Survived"],
    )

    print("First five rows:")
    print(df.head())

    # The notebook removes rows with missing values before feature construction.
    df = df.dropna().copy()

    X = df.drop(columns=["Survived"]).copy()
    y = df["Survived"]

    baseline_score = np.mean(
        cross_val_score(
            LogisticRegression(max_iter=1000),
            X,
            y,
            scoring="accuracy",
            cv=20,
        )
    )
    print(f"\nBaseline 20-fold CV accuracy: {baseline_score:.4f}")

    # Feature construction: total number of people in the passenger's family.
    X["Family_size"] = X["SibSp"] + X["Parch"] + 1

    print("\nAfter creating Family_size:")
    print(X.head())

    # Convert Family_size into three categories.
    X["Family_type"] = X["Family_size"].apply(family_type)

    print("\nAfter creating Family_type:")
    print(X.head())

    # Keep the constructed categorical feature and remove its source features.
    X = X.drop(columns=["SibSp", "Parch", "Family_size"])

    print("\nFinal feature set:")
    print(X.head())

    feature_score = np.mean(
        cross_val_score(
            LogisticRegression(max_iter=1000),
            X,
            y,
            scoring="accuracy",
            cv=20,
        )
    )
    print(f"\nFeature-engineered 20-fold CV accuracy: {feature_score:.4f}")


if __name__ == "__main__":
    main()