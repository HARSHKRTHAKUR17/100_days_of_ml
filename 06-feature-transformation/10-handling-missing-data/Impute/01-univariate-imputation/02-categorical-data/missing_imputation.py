"""
Categorical Missing-Value Imputation — House Prices

Demonstrates two approaches for categorical missing values:

1. Arbitrary/category-label imputation with Pandas
   - Replace missing values with a dedicated category such as "Missing".
   - This preserves the information that the original value was missing.

2. Constant imputation with Scikit-learn
   - SimpleImputer(strategy="constant", fill_value="Missing")

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

    print("\nMissing-value percentage:")
    print(df.isna().mean().mul(100))

    # ---------------------------------------------------------
    # 2. Inspect the categorical distribution
    # ---------------------------------------------------------
    print("\nGarageQual value counts before imputation:")
    print(df["GarageQual"].value_counts(dropna=False))

    # Visualize the distribution.
    df["GarageQual"].value_counts(dropna=False).plot.bar()
    plt.xlabel("GarageQual")
    plt.ylabel("Number of houses")
    plt.title("GarageQual before imputation")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3. Pandas: arbitrary-value/category imputation
    # ---------------------------------------------------------
    df_pandas = df.copy()

    # Use a dedicated category for missing values.
    # Avoid the deprecated/chained style:
    # df["GarageQual"].fillna("Missing", inplace=True)
    df_pandas["GarageQual"] = df_pandas["GarageQual"].fillna("Missing")

    print("\nGarageQual value counts after Pandas imputation:")
    print(df_pandas["GarageQual"].value_counts(dropna=False))

    df_pandas["GarageQual"].value_counts().plot.bar()
    plt.xlabel("GarageQual")
    plt.ylabel("Number of houses")
    plt.title("GarageQual after 'Missing' imputation")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. Scikit-learn: constant imputation
    # ---------------------------------------------------------
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Split before fitting the imputer so the preprocessing step
    # learns only from the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    imputer = SimpleImputer(
        strategy="constant",
        fill_value="Missing",
    )

    # Fit on training data only.
    X_train_transformed = imputer.fit_transform(X_train)

    # Transform test data using the same learned preprocessing.
    X_test_transformed = imputer.transform(X_test)

    print("\nScikit-learn imputer statistics:")
    print(imputer.statistics_)

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
