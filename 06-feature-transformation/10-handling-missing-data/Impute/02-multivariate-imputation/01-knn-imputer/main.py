"""
KNN Missing-Value Imputation — Titanic Dataset

Demonstrates:
- KNN-based numerical imputation
- Distance-weighted nearest neighbors
- Comparison with mean imputation
- Logistic Regression after imputation

Dataset expected:
    train.csv

Columns used:
    Age
    Pclass
    Fare
    Survived

Important:
- Imputers are fitted only on the training data to avoid data leakage.
- KNNImputer is distance-based, so feature scale can affect the
  imputation result. This script follows the notebook's original
  workflow for learning purposes.
"""

from __future__ import annotations

import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 2


def evaluate_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    description: str,
) -> float:
    """Train Logistic Regression and report test accuracy."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{description}: {accuracy:.4f}")
    return accuracy


def main() -> None:
    df = pd.read_csv("train.csv")[["Age", "Pclass", "Fare", "Survived"]]

    print("First five rows:")
    print(df.head())

    print("\nMissing-value percentage:")
    print((df.isna().mean() * 100).round(2))

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("\nTraining features:")
    print(X_train.head())

    # KNN imputation.
    knn_imputer = KNNImputer(
        n_neighbors=3,
        weights="distance",
    )

    X_train_knn = knn_imputer.fit_transform(X_train)
    X_test_knn = knn_imputer.transform(X_test)

    print("\nKNN-imputed training data:")
    print(X_train_knn[:5])

    evaluate_logistic_regression(
        X_train_knn,
        X_test_knn,
        y_train,
        y_test,
        "Accuracy with KNN imputation",
    )

    # Comparison with mean imputation.
    mean_imputer = SimpleImputer(strategy="mean")

    X_train_mean = mean_imputer.fit_transform(X_train)
    X_test_mean = mean_imputer.transform(X_test)

    print("\nLearned mean values:")
    print(dict(zip(X_train.columns, mean_imputer.statistics_)))

    evaluate_logistic_regression(
        X_train_mean,
        X_test_mean,
        y_train,
        y_test,
        "Accuracy with mean imputation",
    )


if __name__ == "__main__":
    main()
