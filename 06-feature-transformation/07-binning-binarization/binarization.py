"""
Titanic — Feature Engineering: Family Size and Binarization

This script demonstrates:
1. Creating a `family` feature from SibSp + Parch.
2. Removing the original SibSp and Parch features.
3. Training a baseline Decision Tree.
4. Binarizing the family feature with sklearn's Binarizer.
5. Comparing model performance before and after binarization.
6. Evaluating the transformed feature with leakage-safe cross-validation.

Dataset expected:
    train.csv

Required columns:
    Age, Fare, SibSp, Parch, Survived

Note:
The original notebook fits the binarization transformer on the full dataset
before cross-validation. This version puts preprocessing and modeling into a
Pipeline so each CV fold learns its transformation independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10


def build_binarization_pipeline() -> Pipeline:
    """Build a pipeline that binarizes family size and trains a Decision Tree."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("family_binarizer", Binarizer(), ["family"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load and prepare the dataset
    # -------------------------------------------------------------------------
    df = pd.read_csv(
        "train.csv",
        usecols=["Age", "Fare", "SibSp", "Parch", "Survived"],
    )

    # The original notebook removes rows containing missing values.
    df = df.dropna().copy()

    print("First five rows:")
    print(df.head())

    # -------------------------------------------------------------------------
    # 2. Create the family-size feature
    # -------------------------------------------------------------------------
    # Family size here means the number of accompanying family members,
    # because SibSp and Parch are summed without adding 1 for the passenger.
    df["family"] = df["SibSp"] + df["Parch"]

    print("\nAfter creating the family feature:")
    print(df.head())

    # The original features are now redundant for this demonstration.
    df = df.drop(columns=["SibSp", "Parch"])

    print("\nFinal modeling dataframe:")
    print(df.head())

    # -------------------------------------------------------------------------
    # 3. Separate features and target
    # -------------------------------------------------------------------------
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nFirst five training rows:")
    print(X_train.head())

    # -------------------------------------------------------------------------
    # 4. Baseline Decision Tree — without binarization
    # -------------------------------------------------------------------------
    baseline_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_clf.fit(X_train, y_train)

    y_pred = baseline_clf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nBaseline test accuracy: {baseline_accuracy:.4f}")

    baseline_cv_scores = cross_val_score(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(f"Baseline mean CV accuracy: {baseline_cv_scores.mean():.4f}")

    # -------------------------------------------------------------------------
    # 5. Binarize the family feature
    # -------------------------------------------------------------------------
    #
    # Binarizer's default threshold is 0:
    #
    #     family <= 0  -> 0
    #     family >  0  -> 1
    #
    # Therefore, this effectively converts:
    #     no accompanying family members -> 0
    #     at least one family member      -> 1
    #
    # ColumnTransformer keeps Age and Fare unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("family_binarizer", Binarizer(), ["family"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_trf = preprocessor.fit_transform(X_train)
    X_test_trf = preprocessor.transform(X_test)

    transformed_columns = ["family", "Age", "Fare"]

    transformed_df = pd.DataFrame(
        X_train_trf,
        columns=transformed_columns,
        index=X_train.index,
    )

    print("\nTransformed training data:")
    print(transformed_df)

    # -------------------------------------------------------------------------
    # 6. Train a Decision Tree on the transformed features
    # -------------------------------------------------------------------------
    transformed_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    transformed_clf.fit(X_train_trf, y_train)

    y_pred_transformed = transformed_clf.predict(X_test_trf)
    transformed_accuracy = accuracy_score(y_test, y_pred_transformed)

    print(
        f"\nBinarized family test accuracy: "
        f"{transformed_accuracy:.4f}"
    )

    # -------------------------------------------------------------------------
    # 7. Leakage-safe cross-validation after binarization
    # -------------------------------------------------------------------------
    #
    # The original notebook does:
    #
    #     X_trf = trf.fit_transform(X)
    #     cross_val_score(..., X_trf, y, ...)
    #
    # This learns the preprocessing using all samples before CV. Although
    # Binarizer's default threshold is fixed at 0, using a Pipeline is the
    # correct general pattern and remains safe if the preprocessing changes.
    #
    pipeline = build_binarization_pipeline()

    transformed_cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(
        "\nBinarized family mean CV accuracy "
        f"(pipeline): {transformed_cv_scores.mean():.4f}"
    )

    # -------------------------------------------------------------------------
    # 8. Performance comparison
    # -------------------------------------------------------------------------
    comparison = pd.DataFrame(
        {
            "model": [
                "Decision Tree — original family",
                "Decision Tree — binarized family",
            ],
            "test_accuracy": [
                baseline_accuracy,
                transformed_accuracy,
            ],
            "mean_cv_accuracy": [
                baseline_cv_scores.mean(),
                transformed_cv_scores.mean(),
            ],
        }
    )

    print("\nPerformance comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
