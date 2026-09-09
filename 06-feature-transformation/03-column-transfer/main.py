# COVID Toy Dataset — Handling Missing and Categorical Data
#
# Demonstrates:
# 1. Inspecting missing values
# 2. Train-test split
# 3. SimpleImputer
# 4. OrdinalEncoder
# 5. OneHotEncoder
# 6. Combining transformed features manually
# 7. ColumnTransformer for a cleaner preprocessing pipeline


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# 1. Load and Inspect Dataset
# ============================================================

df = pd.read_csv("covid_toy.csv")

print("First five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. Train-Test Split
# ============================================================

X = df.drop(columns=["has_covid"])
y = df["has_covid"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(X_train)


# ============================================================
# 3. Simple Imputation — Fever
# ============================================================
# SimpleImputer() uses the mean strategy by default.
#
# IMPORTANT:
# Fit the imputer only on the training data.
# Use transform() on the test data to prevent data leakage.

fever_imputer = SimpleImputer(strategy="mean")

X_train_fever = fever_imputer.fit_transform(
    X_train[["fever"]]
)

X_test_fever = fever_imputer.transform(
    X_test[["fever"]]
)

print("\nTraining fever shape:")
print(X_train_fever.shape)


# ============================================================
# 4. Ordinal Encoding — Cough
# ============================================================
# Mild  -> 0
# Strong -> 1

cough_encoder = OrdinalEncoder(
    categories=[["Mild", "Strong"]]
)

X_train_cough = cough_encoder.fit_transform(
    X_train[["cough"]]
)

X_test_cough = cough_encoder.transform(
    X_test[["cough"]]
)

print("\nTraining cough shape:")
print(X_train_cough.shape)


# ============================================================
# 5. One-Hot Encoding — Gender and City
# ============================================================
# drop="first" performs K-1 encoding.
# handle_unknown="ignore" prevents errors if an unseen
# category appears in the test data.

try:
    gender_city_encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    gender_city_encoder = OneHotEncoder(
        drop="first",
        sparse=False,
        handle_unknown="ignore"
    )

X_train_gender_city = gender_city_encoder.fit_transform(
    X_train[["gender", "city"]]
)

X_test_gender_city = gender_city_encoder.transform(
    X_test[["gender", "city"]]
)

print("\nTraining gender/city shape:")
print(X_train_gender_city.shape)


# ============================================================
# 6. Extract Age
# ============================================================

X_train_age = X_train[["age"]].values
X_test_age = X_test[["age"]].values

print("\nTraining age shape:")
print(X_train_age.shape)


# ============================================================
# 7. Combine All Transformed Features
# ============================================================

X_train_transformed = np.concatenate(
    (
        X_train_age,
        X_train_fever,
        X_train_gender_city,
        X_train_cough
    ),
    axis=1
)

X_test_transformed = np.concatenate(
    (
        X_test_age,
        X_test_fever,
        X_test_gender_city,
        X_test_cough
    ),
    axis=1
)

print("\nFinal training shape:")
print(X_train_transformed.shape)

print("\nFinal test shape:")
print(X_test_transformed.shape)


# ============================================================
# 8. ColumnTransformer — Recommended Approach
# ============================================================
# ColumnTransformer allows all preprocessing steps to be
# defined in one object instead of manually concatenating
# transformed arrays.

try:
    one_hot_encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    one_hot_encoder = OneHotEncoder(
        drop="first",
        sparse=False,
        handle_unknown="ignore"
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "fever_imputer",
            SimpleImputer(strategy="mean"),
            ["fever"]
        ),
        (
            "cough_encoder",
            OrdinalEncoder(
                categories=[["Mild", "Strong"]]
            ),
            ["cough"]
        ),
        (
            "gender_city_encoder",
            one_hot_encoder,
            ["gender", "city"]
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 9. Transform Training and Test Data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nColumnTransformer training shape:")
print(X_train_processed.shape)

print("\nColumnTransformer test shape:")
print(X_test_processed.shape)