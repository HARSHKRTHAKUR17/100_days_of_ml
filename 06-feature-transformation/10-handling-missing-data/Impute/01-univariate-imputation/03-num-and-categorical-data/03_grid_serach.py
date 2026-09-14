"""
Titanic Logistic Regression Pipeline + Grid Search
---------------------------------------------------

Demonstrates:
- Separating features and target
- Train/test splitting
- Numerical preprocessing with imputation + standardization
- Categorical preprocessing with imputation + one-hot encoding
- Combining preprocessing with ColumnTransformer
- Building an end-to-end Pipeline
- Hyperparameter tuning with GridSearchCV

Dataset expected:
    train.csv
"""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 2


def build_pipeline() -> Pipeline:
    """Build the complete Titanic preprocessing + classification pipeline."""

    numerical_features = ["Age", "Fare"]
    categorical_features = ["Embarked", "Sex"]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    df = pd.read_csv("train.csv")

    print("First five rows:")
    print(df.head())

    # Drop identifiers / high-cardinality columns used in this exercise.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    print("\nData after dropping unused columns:")
    print(df.head())

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

    pipeline = build_pipeline()

    # Display the structure of the pipeline.
    print("\nPipeline:")
    print(pipeline)

    # Hyperparameter grid.
    param_grid = {
        "preprocessor__num__imputer__strategy": ["mean", "median"],
        "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],
        "classifier__C": [0.1, 1.0, 10, 100],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nBest internal CV score: {grid_search.best_score_:.3f}")

    # Evaluate the best model on the held-out test set.
    test_score = grid_search.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.3f}")

    # Show the best configurations.
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values("mean_test_score", ascending=False)

    columns_to_show = [
        "param_classifier__C",
        "param_preprocessor__cat__imputer__strategy",
        "param_preprocessor__num__imputer__strategy",
        "mean_test_score",
    ]

    print("\nTop GridSearchCV results:")
    print(cv_results[columns_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
