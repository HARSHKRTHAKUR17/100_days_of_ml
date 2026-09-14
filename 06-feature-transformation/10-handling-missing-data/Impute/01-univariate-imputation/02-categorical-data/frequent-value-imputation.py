"""
Categorical Missing-Value Imputation
------------------------------------

Demonstrates:
1. Mode imputation using Pandas
2. Checking whether missingness is informative using SalePrice
3. Most-frequent imputation using Scikit-learn
4. Avoiding data leakage by fitting the imputer on training data only

Dataset expected:
    train.csv

Columns used:
    GarageQual
    FireplaceQu
    SalePrice
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def plot_missingness_effect(
    df: pd.DataFrame,
    feature: str,
    category: str,
    target: str = "SalePrice",
) -> None:
    """Compare the target distribution for a category and missing values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    df.loc[df[feature] == category, target].plot(
        kind="kde",
        ax=ax,
        label=f"Houses with {category}",
    )

    df.loc[df[feature].isna(), target].plot(
        kind="kde",
        ax=ax,
        label="Houses with missing value",
    )

    ax.set_title(f"{feature}: category vs missing values")
    ax.set_xlabel(target)
    ax.legend()
    plt.tight_layout()
    plt.show()


def pandas_mode_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Impute categorical missing values with the mode."""
    result = df.copy()

    # The notebook finds the modes:
    # GarageQual -> TA
    # FireplaceQu -> Gd
    #
    # We calculate them instead of hard-coding the values so that
    # the code remains valid if the dataset changes.
    garage_mode = result["GarageQual"].mode().iloc[0]
    fireplace_mode = result["FireplaceQu"].mode().iloc[0]

    print("\nPandas mode values:")
    print(f"GarageQual mode: {garage_mode}")
    print(f"FireplaceQu mode: {fireplace_mode}")

    # Avoid the deprecated/chained inplace style:
    # df["GarageQual"].fillna("TA", inplace=True)
    result["GarageQual"] = result["GarageQual"].fillna(garage_mode)
    result["FireplaceQu"] = result["FireplaceQu"].fillna(fireplace_mode)

    return result


def sklearn_mode_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute categorical columns using Scikit-learn."""
    imputer = SimpleImputer(strategy="most_frequent")

    # Learn the most frequent category from training data only.
    X_train_transformed = imputer.fit_transform(X_train)
    X_test_transformed = imputer.transform(X_test)

    # Convert NumPy arrays back to DataFrames for readability.
    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=X_test.columns,
        index=X_test.index,
    )

    print("\nScikit-learn learned values:")
    print(dict(zip(X_train.columns, imputer.statistics_)))

    return X_train_transformed, X_test_transformed


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the dataset
    # ---------------------------------------------------------
    df = pd.read_csv(
        "train.csv",
        usecols=["GarageQual", "FireplaceQu", "SalePrice"],
    )

    print("First five rows:")
    print(df.head())

    # ---------------------------------------------------------
    # 2. Inspect missing values
    # ---------------------------------------------------------
    print("\nMissing-value percentage:")
    print(df.isna().mean().mul(100))

    print("\nGarageQual value counts:")
    print(df["GarageQual"].value_counts(dropna=False))

    print("\nFireplaceQu value counts:")
    print(df["FireplaceQu"].value_counts(dropna=False))

    # ---------------------------------------------------------
    # 3. Check whether missingness may be informative
    # ---------------------------------------------------------
    plot_missingness_effect(df, "GarageQual", "TA")
    plot_missingness_effect(df, "FireplaceQu", "Gd")

    # ---------------------------------------------------------
    # 4. Pandas: mode imputation
    # ---------------------------------------------------------
    df_pandas = pandas_mode_imputation(df)

    print("\nAfter Pandas mode imputation:")
    print(df_pandas[["GarageQual", "FireplaceQu"]].head())

    print("\nMissing values after Pandas imputation:")
    print(df_pandas[["GarageQual", "FireplaceQu"]].isna().sum())

    # ---------------------------------------------------------
    # 5. Scikit-learn: most-frequent imputation
    #
    # IMPORTANT:
    # Split BEFORE fitting the imputer to prevent data leakage.
    # ---------------------------------------------------------
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_imputed, X_test_imputed = sklearn_mode_imputation(
        X_train,
        X_test,
    )

    print("\nTraining data after Scikit-learn imputation:")
    print(X_train_imputed.head())

    print("\nTest data after Scikit-learn imputation:")
    print(X_test_imputed.head())

    print("\nMissing values after Scikit-learn imputation:")
    print(X_train_imputed.isna().sum())
    print(X_test_imputed.isna().sum())


if __name__ == "__main__":
    main()
