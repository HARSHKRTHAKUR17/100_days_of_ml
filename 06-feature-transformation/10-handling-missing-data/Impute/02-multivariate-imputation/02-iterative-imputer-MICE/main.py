"""
Iterative Missing-Value Imputation (MICE-style intuition)
-----------------------------------------------------------

Demonstrates the intuition behind iterative/model-based imputation:

1. Start with simple mean imputation.
2. Temporarily remove one imputed value.
3. Train a regression model using the other features.
4. Predict the missing value.
5. Repeat for the other columns.
6. Repeat the cycle for multiple iterations.

Dataset expected:
    50_Startups.csv

The notebook uses a tiny five-row sample from the 50 Startups dataset
so the iterative process can be inspected manually.

Note:
    This is a teaching implementation of iterative imputation, not a
    production MICE implementation. In real projects, use a validated
    iterative-imputation approach inside a leakage-safe Pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


RANDOM_STATE = 9
SCALE_FACTOR = 10_000
FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]


def create_toy_dataset() -> pd.DataFrame:
    """Create the same five-row toy dataset used by the notebook."""
    df = pd.read_csv("50_Startups.csv")[FEATURES + ["Profit"]]
    df = np.round(df / SCALE_FACTOR)

    # Equivalent to the notebook's np.random.seed(9); df.sample(5).
    return df.sample(5, random_state=RANDOM_STATE).drop(columns="Profit")


def introduce_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce one missing value into each feature column."""
    result = df.copy()
    result.iloc[1, 0] = np.nan
    result.iloc[3, 1] = np.nan
    result.iloc[-1, -1] = np.nan
    return result


def mean_initialize(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize missing values using each column's mean."""
    return df.fillna(df.mean())


def model_impute(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    row_index,
) -> float:
    """Predict one missing value using Linear Regression."""
    train_data = df.dropna(subset=[target_column] + feature_columns)

    model = LinearRegression()
    model.fit(train_data[feature_columns], train_data[target_column])

    prediction = model.predict(df.loc[[row_index], feature_columns])[0]
    return float(prediction)


def run_iteration(df: pd.DataFrame, iteration_name: str) -> pd.DataFrame:
    """Run one complete model-based imputation cycle."""
    result = df.copy()

    # R&D Spend <- Administration + Marketing Spend
    rd_index = result.index[1]
    result.loc[rd_index, "R&D Spend"] = np.nan
    rd_prediction = model_impute(
        result,
        "R&D Spend",
        ["Administration", "Marketing Spend"],
        rd_index,
    )
    result.loc[rd_index, "R&D Spend"] = rd_prediction

    # Administration <- R&D Spend + Marketing Spend
    administration_index = result.index[3]
    result.loc[administration_index, "Administration"] = np.nan
    administration_prediction = model_impute(
        result,
        "Administration",
        ["R&D Spend", "Marketing Spend"],
        administration_index,
    )
    result.loc[administration_index, "Administration"] = administration_prediction

    # Marketing Spend <- R&D Spend + Administration
    marketing_index = result.index[-1]
    result.loc[marketing_index, "Marketing Spend"] = np.nan
    marketing_prediction = model_impute(
        result,
        "Marketing Spend",
        ["R&D Spend", "Administration"],
        marketing_index,
    )
    result.loc[marketing_index, "Marketing Spend"] = marketing_prediction

    print(f"\n{iteration_name}:")
    print(result)

    return result


def main() -> None:
    original = create_toy_dataset()
    missing_data = introduce_missing_values(original)

    print("Original toy dataset:")
    print(original)

    print("\nDataset with missing values:")
    print(missing_data)

    # 0th iteration: simple mean initialization.
    df0 = mean_initialize(missing_data)
    print("\n0th iteration — mean initialization:")
    print(df0)

    # 1st iteration: replace each initial mean with a regression estimate.
    df1 = run_iteration(df0, "1st iteration")
    print("\nChange from 0th to 1st iteration:")
    print(df1 - df0)

    # 2nd iteration.
    df2 = run_iteration(df1, "2nd iteration")
    print("\nChange from 1st to 2nd iteration:")
    print(df2 - df1)

    # 3rd iteration.
    df3 = run_iteration(df2, "3rd iteration")
    print("\nChange from 2nd to 3rd iteration:")
    print(df3 - df2)

    print("\nFinal imputed dataset:")
    print(df3)


if __name__ == "__main__":
    main()