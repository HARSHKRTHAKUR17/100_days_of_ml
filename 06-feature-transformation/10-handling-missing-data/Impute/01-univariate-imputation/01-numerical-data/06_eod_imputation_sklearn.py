"""
End-of-Distribution Imputation using Feature-engine
----------------------------------------------------

End-of-distribution imputation replaces missing values with a value
at the end of the feature's distribution.

For normally distributed variables, Feature-engine commonly uses:

    mean + 3 * standard deviation

For skewed variables, it can use a different method such as
median + 3 * IQR.

Install:
    pip install feature-engine pandas scikit-learn
"""

from __future__ import annotations

import pandas as pd
from feature_engine.imputation import EndTailImputer
from sklearn.model_selection import train_test_split


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    df = pd.read_csv("titanic_toy.csv")

    print("First five rows:")
    print(df.head())

    print("\nMissing-value fractions:")
    print(df.isna().mean())

    # ---------------------------------------------------------
    # 2. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # ---------------------------------------------------------
    # 3. Train-test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2,
    )

    # ---------------------------------------------------------
    # 4. Create End-of-Distribution imputer
    #
    # end_tail="right" means:
    #
    #     mean + 3 * std
    #
    # for normally distributed variables.
    # ---------------------------------------------------------
    imputer = EndTailImputer(
        variables=["Age", "Fare"],
        fold=3,
        tail="right",
        distribution="gaussian",
    )

    # ---------------------------------------------------------
    # 5. Fit ONLY on training data
    # ---------------------------------------------------------
    imputer.fit(X_train)

    print("\nLearned imputation values:")
    print(imputer.imputer_dict_)

    # ---------------------------------------------------------
    # 6. Transform train and test data
    # ---------------------------------------------------------
    X_train_transformed = imputer.transform(X_train)
    X_test_transformed = imputer.transform(X_test)

    # ---------------------------------------------------------
    # 7. Compare original and imputed values
    # ---------------------------------------------------------
    print("\nOriginal training data:")
    print(X_train.head(10))

    print("\nTraining data after EOD imputation:")
    print(X_train_transformed.head(10))

    # ---------------------------------------------------------
    # 8. Verify that missing values were removed
    # ---------------------------------------------------------
    print("\nMissing values before:")
    print(X_train[["Age", "Fare"]].isna().sum())

    print("\nMissing values after:")
    print(X_train_transformed[["Age", "Fare"]].isna().sum())


if __name__ == "__main__":
    main()