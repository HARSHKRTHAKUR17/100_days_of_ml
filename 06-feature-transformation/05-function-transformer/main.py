from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# 1. Load data
# =============================================================================

DATA_PATH = Path("train.csv")

df = pd.read_csv(DATA_PATH, usecols=["Age", "Fare", "Survived"])

print("First five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# =============================================================================
# 2. Split the data and handle missing Age values
# =============================================================================
# The original notebook fills Age before splitting. That can cause data
# leakage because the test set influences the imputation value.
# Here the mean is learned from the training set only.

X = df[["Age", "Fare"]].copy()
y = df["Survived"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

age_mean = X_train["Age"].mean()

X_train = X_train.copy()
X_test = X_test.copy()

X_train["Age"] = X_train["Age"].fillna(age_mean)
X_test["Age"] = X_test["Age"].fillna(age_mean)


# =============================================================================
# 3. Inspect Age distribution
# =============================================================================

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
sns.histplot(X_train["Age"], kde=True)
plt.title("Age PDF")

plt.subplot(1, 2, 2)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.tight_layout()
plt.show()


# =============================================================================
# 4. Inspect Fare distribution
# =============================================================================

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
sns.histplot(X_train["Fare"], kde=True)
plt.title("Fare PDF")

plt.subplot(1, 2, 2)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Fare QQ Plot")

plt.tight_layout()
plt.show()


# =============================================================================
# 5. Baseline models
# =============================================================================

logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

logistic_regression.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

y_pred_lr = logistic_regression.predict(X_test)
y_pred_dt = decision_tree.predict(X_test)

print("\nBaseline accuracy:")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))


# =============================================================================
# 6. Apply log1p transformation to both features
# =============================================================================
# log1p(x) = log(1 + x), so zero values are handled safely.

log_transform = FunctionTransformer(
    np.log1p,
    feature_names_out="one-to-one",
)

X_train_transformed = pd.DataFrame(
    log_transform.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_test_transformed = pd.DataFrame(
    log_transform.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

logistic_regression.fit(X_train_transformed, y_train)
decision_tree.fit(X_train_transformed, y_train)

y_pred_lr = logistic_regression.predict(X_test_transformed)
y_pred_dt = decision_tree.predict(X_test_transformed)

print("\nAccuracy after log1p transformation:")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))


# =============================================================================
# 7. Cross-validation after transforming both features
# =============================================================================

X_transformed = log_transform.fit_transform(X)

logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

lr_cv_score = cross_val_score(
    logistic_regression,
    X_transformed,
    y,
    scoring="accuracy",
    cv=10,
).mean()

dt_cv_score = cross_val_score(
    decision_tree,
    X_transformed,
    y,
    scoring="accuracy",
    cv=10,
).mean()

print("\n10-fold cross-validation after log1p transformation:")
print("Logistic Regression:", lr_cv_score)
print("Decision Tree:", dt_cv_score)


# =============================================================================
# 8. Fare before and after log transformation
# =============================================================================

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Fare Before Log")

plt.subplot(1, 2, 2)
stats.probplot(X_train_transformed["Fare"], dist="norm", plot=plt)
plt.title("Fare After Log")

plt.tight_layout()
plt.show()


# =============================================================================
# 9. Age before and after log transformation
# =============================================================================

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Age Before Log")

plt.subplot(1, 2, 2)
stats.probplot(X_train_transformed["Age"], dist="norm", plot=plt)
plt.title("Age After Log")

plt.tight_layout()
plt.show()


# =============================================================================
# 10. Apply log1p only to Fare using ColumnTransformer
# =============================================================================

fare_log_transformer = ColumnTransformer(
    transformers=[
        (
            "log_fare",
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ["Fare"],
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

X_train_transformed_fare = fare_log_transformer.fit_transform(X_train)
X_test_transformed_fare = fare_log_transformer.transform(X_test)

transformed_columns = fare_log_transformer.get_feature_names_out()

X_train_transformed_fare = pd.DataFrame(
    X_train_transformed_fare,
    columns=transformed_columns,
    index=X_train.index,
)

X_test_transformed_fare = pd.DataFrame(
    X_test_transformed_fare,
    columns=transformed_columns,
    index=X_test.index,
)

logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

logistic_regression.fit(X_train_transformed_fare, y_train)
decision_tree.fit(X_train_transformed_fare, y_train)

y_pred_lr = logistic_regression.predict(X_test_transformed_fare)
y_pred_dt = decision_tree.predict(X_test_transformed_fare)

print("\nAccuracy after transforming Fare only:")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))


# =============================================================================
# 11. Cross-validation for Fare-only transformation
# =============================================================================

X_transformed_fare = fare_log_transformer.fit_transform(X)

logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

lr_cv_score = cross_val_score(
    logistic_regression,
    X_transformed_fare,
    y,
    scoring="accuracy",
    cv=10,
).mean()

dt_cv_score = cross_val_score(
    decision_tree,
    X_transformed_fare,
    y,
    scoring="accuracy",
    cv=10,
).mean()

print("\n10-fold cross-validation after transforming Fare only:")
print("Logistic Regression:", lr_cv_score)
print("Decision Tree:", dt_cv_score)


# =============================================================================
# 12. Reusable custom transformation function
# =============================================================================

def apply_transform(transform, transform_name):
    """
    Apply a custom transformation to Fare, evaluate Logistic Regression,
    and visualize Fare before and after transformation.
    """
    X_local = df[["Age", "Fare"]].copy()
    y_local = df["Survived"].copy()

    transformer = ColumnTransformer(
        transformers=[
            (
                "transform_fare",
                FunctionTransformer(
                    transform,
                    feature_names_out="one-to-one",
                ),
                ["Fare"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_transformed = transformer.fit_transform(X_local)

    model = LogisticRegression(max_iter=1000)

    accuracy = cross_val_score(
        model,
        X_transformed,
        y_local,
        scoring="accuracy",
        cv=10,
    ).mean()

    print(f"\nAccuracy using {transform_name}: {accuracy:.4f}")

    transformed_columns = transformer.get_feature_names_out()
    fare_index = list(transformed_columns).index("Fare")

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    stats.probplot(X_local["Fare"], dist="norm", plot=plt)
    plt.title("Fare Before Transform")

    plt.subplot(1, 2, 2)
    stats.probplot(X_transformed[:, fare_index], dist="norm", plot=plt)
    plt.title(f"Fare After {transform_name}")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 13. Example: sine transformation
# =============================================================================
# This reproduces the notebook's final FunctionTransformer experiment.
# It is included for learning purposes, not because sine is a good choice
# for Titanic Fare modelling.

apply_transform(np.sin, "Sine Transform")
