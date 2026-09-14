"""
Missing-Value Indicators
------------------------

Demonstrates:
1. SimpleImputer for numerical missing values.
2. Logistic Regression after imputation.
3. MissingIndicator as a separate feature.
4. Adding the missing-value indicator to the original data.
5. SimpleImputer(add_indicator=True) as the convenient combined approach.

Dataset expected:
    train.csv

Columns used:
    Age
    Fare
    Survived

Important:
- Imputers and indicators are fitted on the training data only.
- The original notebook's final section accidentally trained on the
  earlier transformed data instead of the newly created data containing
  the missing-value indicator. That bug is fixed here.
- DataFrame copies are used before adding Age_NA to avoid
  SettingWithCopyWarning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 2


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    description: str,
) -> float:
    """Train Logistic Regression and print test accuracy."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{description}: {accuracy:.4f}")
    return accuracy


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the Titanic dataset
    # ---------------------------------------------------------
    df = pd.read_csv(
        "train.csv",
        usecols=["Age", "Fare", "Survived"],
    )

    print("First five rows:")
    print(df.head())

    # ---------------------------------------------------------
    # 2. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("
Training data:")
    print(X_train.head())

    # ---------------------------------------------------------
    # 3. SimpleImputer — baseline
    # ---------------------------------------------------------
    # By default, SimpleImputer uses the mean strategy for numerical
    # columns.
    simple_imputer = SimpleImputer()

    X_train_imputed = simple_imputer.fit_transform(X_train)
    X_test_imputed = simple_imputer.transform(X_test)

    print("
Imputation statistics:")
    print(dict(zip(X_train.columns, simple_imputer.statistics_)))

    print("
Transformed training data:")
    print(X_train_imputed[:5])

    train_and_evaluate(
        X_train_imputed,
        X_test_imputed,
        y_train,
        y_test,
        "Accuracy after simple imputation",
    )

    # ---------------------------------------------------------
    # 4. MissingIndicator — identify where values were missing
    # ---------------------------------------------------------
    indicator = MissingIndicator()

    # Fit only on training data.
    indicator.fit(X_train)

    print("
Columns containing missing values:")
    print(indicator.features_)

    X_train_missing = indicator.transform(X_train)
    X_test_missing = indicator.transform(X_test)

    print("
Training missing-value indicator:")
    print(X_train_missing[:20])

    print("
Test missing-value indicator:")
    print(X_test_missing[:20])

    # ---------------------------------------------------------
    # 5. Add the indicator to the original DataFrames
    # ---------------------------------------------------------
    # MissingIndicator returns one column for Age because Age is the
    # feature containing missing values in this dataset.
    X_train_with_indicator = X_train.copy()
    X_test_with_indicator = X_test.copy()

    indicator_column_names = [
        f"{X_train.columns[i]}_NA"
        for i in indicator.features_
    ]

    X_train_with_indicator[indicator_column_names] = X_train_missing
    X_test_with_indicator[indicator_column_names] = X_test_missing

    print("
Training data with missing-value indicator:")
    print(X_train_with_indicator.head())

    print("
Test data with missing-value indicator:")
    print(X_test_with_indicator.head())

    # ---------------------------------------------------------
    # 6. Impute the data containing the indicator
    # ---------------------------------------------------------
    # The indicator itself is boolean, so SimpleImputer leaves it
    # unchanged because it contains no missing values.
    imputer_with_manual_indicator = SimpleImputer()

    X_train_manual_indicator = imputer_with_manual_indicator.fit_transform(
        X_train_with_indicator
    )
    X_test_manual_indicator = imputer_with_manual_indicator.transform(
        X_test_with_indicator
    )

    train_and_evaluate(
        X_train_manual_indicator,
        X_test_manual_indicator,
        y_train,
        y_test,
        "Accuracy with manual missing indicator",
    )

    # ---------------------------------------------------------
    # 7. SimpleImputer(add_indicator=True)
    # ---------------------------------------------------------
    # This combines imputation and missing-indicator creation in one
    # transformer.
    imputer_with_indicator = SimpleImputer(add_indicator=True)

    X_train_combined = imputer_with_indicator.fit_transform(X_train)
    X_test_combined = imputer_with_indicator.transform(X_test)

    print("
Shape after imputation + missing indicators:")
    print("Original X_train:", X_train.shape)
    print("Transformed X_train:", X_train_combined.shape)
    print("Original X_test:", X_test.shape)
    print("Transformed X_test:", X_test_combined.shape)

    train_and_evaluate(
        X_train_combined,
        X_test_combined,
        y_train,
        y_test,
        "Accuracy with SimpleImputer(add_indicator=True)",
    )


if __name__ == "__main__":
    main()