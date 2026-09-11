"""
Concrete Strength Regression — Power Transformations


Concepts covered:
1. Basic dataset inspection
2. Linear regression without transformation
3. Cross-validation
4. Distribution and Q-Q plots
5. Box-Cox transformation
6. Linear regression after Box-Cox transformation
7. Yeo-Johnson transformation
8. Before/after distribution comparison
9. Comparison of learned transformation lambdas

Notes:
- The original notebook uses deprecated seaborn.distplot(); this script uses
  seaborn.histplot(..., kde=True).
- Box-Cox requires strictly positive values. The concrete dataset contains
  zero-valued features, so a very small positive offset is added before the
  Box-Cox transformation, preserving the original notebook's approach.
- Cross-validation is implemented with sklearn Pipelines so that the
  transformation is fitted separately inside each training fold. This avoids
  data leakage.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


RANDOM_STATE = 42
TEST_SIZE = 0.20
BOX_COX_EPSILON = 1e-6


class AddConstant(BaseEstimator, TransformerMixin):
    """Add a small constant before applying a Box-Cox transformation."""

    def __init__(self, constant=1e-6):
        self.constant = constant

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X + self.constant


def plot_distributions(data, title_prefix=""):
    """Plot histograms with KDE and normal Q-Q plots for each feature."""
    for col in data.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(data[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{title_prefix}{col}")

        stats.probplot(data[col], dist="norm", plot=axes[1])
        axes[1].set_title(f"{title_prefix}{col} — Q-Q Plot")

        plt.tight_layout()
        plt.show()


def plot_before_after(original, transformed, title_prefix=""):
    """Compare feature distributions before and after transformation."""
    for col in original.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(original[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{title_prefix}Before — {col}")

        sns.histplot(transformed[col], kde=True, ax=axes[1])
        axes[1].set_title(f"{title_prefix}After — {col}")

        plt.tight_layout()
        plt.show()


def mean_cv_r2(model, X, y):
    """Return mean 5-fold cross-validated R²."""
    scores = cross_val_score(model, X, y, scoring="r2", cv=5)
    return scores.mean(), scores


def main():
    # -------------------------------------------------------------------------
    # 1. Load and inspect the dataset
    # -------------------------------------------------------------------------
    df = pd.read_csv("concrete_data.csv")

    print("First five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    # -------------------------------------------------------------------------
    # 2. Separate features and target
    # -------------------------------------------------------------------------
    X = df.drop(columns=["Strength"])
    y = df["Strength"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # 3. Linear regression without any transformation
    # -------------------------------------------------------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    baseline_r2 = r2_score(y_test, y_pred)

    print(f"\nBaseline test R²: {baseline_r2:.4f}")

    baseline_cv_mean, baseline_cv_scores = mean_cv_r2(lr, X, y)

    print(f"Baseline 5-fold CV mean R²: {baseline_cv_mean:.4f}")
    print(f"Baseline CV scores: {baseline_cv_scores}")

    # -------------------------------------------------------------------------
    # 4. Plot distributions before transformation
    # -------------------------------------------------------------------------
    plot_distributions(X_train, title_prefix="Original — ")

    # -------------------------------------------------------------------------
    # 5. Apply Box-Cox transformation
    #
    # Box-Cox requires strictly positive values. Some concrete features contain
    # zeros, so add a tiny constant before fitting the transformer.
    # -------------------------------------------------------------------------
    box_cox_transformer = PowerTransformer(
        method="box-cox",
        standardize=True,
    )

    X_train_box_cox = box_cox_transformer.fit_transform(
        X_train + BOX_COX_EPSILON
    )
    X_test_box_cox = box_cox_transformer.transform(
        X_test + BOX_COX_EPSILON
    )

    X_train_box_cox = pd.DataFrame(
        X_train_box_cox,
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_box_cox = pd.DataFrame(
        X_test_box_cox,
        columns=X_test.columns,
        index=X_test.index,
    )

    box_cox_lambdas = pd.DataFrame(
        {
            "feature": X_train.columns,
            "box_cox_lambda": box_cox_transformer.lambdas_,
        }
    )

    print("\nBox-Cox transformation lambdas:")
    print(box_cox_lambdas)

    # -------------------------------------------------------------------------
    # 6. Linear regression after Box-Cox transformation
    # -------------------------------------------------------------------------
    lr_box_cox = LinearRegression()
    lr_box_cox.fit(X_train_box_cox, y_train)

    y_pred_box_cox = lr_box_cox.predict(X_test_box_cox)
    box_cox_r2 = r2_score(y_test, y_pred_box_cox)

    print(f"\nBox-Cox test R²: {box_cox_r2:.4f}")

    # IMPORTANT:
    # Fit the transformer inside each CV fold instead of transforming all X
    # before cross-validation. This prevents information leakage.
    box_cox_pipeline = Pipeline(
        steps=[
            ("add_constant", AddConstant(BOX_COX_EPSILON)),
            (
                "power_transform",
                PowerTransformer(method="box-cox", standardize=True),
            ),
            ("regressor", LinearRegression()),
        ]
    )

    box_cox_cv_mean, box_cox_cv_scores = mean_cv_r2(
        box_cox_pipeline,
        X,
        y,
    )

    print(f"Box-Cox 5-fold CV mean R²: {box_cox_cv_mean:.4f}")
    print(f"Box-Cox CV scores: {box_cox_cv_scores}")

    # -------------------------------------------------------------------------
    # 7. Compare original and Box-Cox distributions
    # -------------------------------------------------------------------------
    plot_before_after(
        X_train,
        X_train_box_cox,
        title_prefix="Box-Cox — ",
    )

    # -------------------------------------------------------------------------
    # 8. Apply Yeo-Johnson transformation
    #
    # Unlike Box-Cox, Yeo-Johnson can directly handle zero and negative values.
    # -------------------------------------------------------------------------
    yeo_johnson_transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=True,
    )

    X_train_yeo = yeo_johnson_transformer.fit_transform(X_train)
    X_test_yeo = yeo_johnson_transformer.transform(X_test)

    X_train_yeo = pd.DataFrame(
        X_train_yeo,
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_yeo = pd.DataFrame(
        X_test_yeo,
        columns=X_test.columns,
        index=X_test.index,
    )

    lr_yeo = LinearRegression()
    lr_yeo.fit(X_train_yeo, y_train)

    y_pred_yeo = lr_yeo.predict(X_test_yeo)
    yeo_r2 = r2_score(y_test, y_pred_yeo)

    print(f"\nYeo-Johnson test R²: {yeo_r2:.4f}")

    yeo_cv_pipeline = Pipeline(
        steps=[
            (
                "power_transform",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
            ("regressor", LinearRegression()),
        ]
    )

    yeo_cv_mean, yeo_cv_scores = mean_cv_r2(
        yeo_cv_pipeline,
        X,
        y,
    )

    print(f"Yeo-Johnson 5-fold CV mean R²: {yeo_cv_mean:.4f}")
    print(f"Yeo-Johnson CV scores: {yeo_cv_scores}")

    # -------------------------------------------------------------------------
    # 9. Compare original and Yeo-Johnson distributions
    # -------------------------------------------------------------------------
    plot_before_after(
        X_train,
        X_train_yeo,
        title_prefix="Yeo-Johnson — ",
    )

    # -------------------------------------------------------------------------
    # 10. Compare Box-Cox and Yeo-Johnson lambdas
    # -------------------------------------------------------------------------
    lambda_comparison = pd.DataFrame(
        {
            "feature": X_train.columns,
            "box_cox_lambda": box_cox_transformer.lambdas_,
            "yeo_johnson_lambda": yeo_johnson_transformer.lambdas_,
        }
    )

    print("\nBox-Cox vs Yeo-Johnson lambdas:")
    print(lambda_comparison)

    # -------------------------------------------------------------------------
    # 11. Final comparison
    # -------------------------------------------------------------------------
    results = pd.DataFrame(
        {
            "model": [
                "Linear Regression",
                "Box-Cox + Linear Regression",
                "Yeo-Johnson + Linear Regression",
            ],
            "test_r2": [
                baseline_r2,
                box_cox_r2,
                yeo_r2,
            ],
            "cv_mean_r2": [
                baseline_cv_mean,
                box_cox_cv_mean,
                yeo_cv_mean,
            ],
        }
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
