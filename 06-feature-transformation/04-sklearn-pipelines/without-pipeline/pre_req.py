# Titanic Dataset — Imputation, Encoding and ColumnTransformer
#
# Demonstrates:
# 1. Loading and inspecting data
# 2. Removing unnecessary features
# 3. Train-test split
# 4. Missing-value imputation
# 5. One-hot encoding
# 6. Manual feature transformation
# 7. ColumnTransformer
# 8. Decision Tree classification
# 9. Model evaluation
# 10. Saving preprocessing objects and model


import os
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. Load and Inspect Dataset
# ============================================================

df = pd.read_csv("train.csv")

print("First five rows:")
print(df.head())


# ============================================================
# 2. Remove Unnecessary Features
# ============================================================
# PassengerId, Name, Ticket and Cabin are excluded here because
# this example focuses on basic preprocessing techniques.

df = df.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)

print("\nDataset after removing unnecessary columns:")
print(df.head())


# ============================================================
# 3. Train-Test Split
# ============================================================

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(X_train.head(2))

print("\nTraining target:")
print(y_train.head())


# ============================================================
# 4. Inspect Missing Values
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 5. Manual Preprocessing
# ============================================================
# Age:
#   Missing values are replaced with the training-set mean.
#
# Embarked:
#   Missing values are replaced with the most frequent value.
#
# IMPORTANT:
# Fit preprocessing objects only on training data.
# Use transform() on test data to avoid data leakage.


# ----------------------------
# Impute Age
# ----------------------------

age_imputer = SimpleImputer(strategy="mean")

X_train_age = age_imputer.fit_transform(
    X_train[["Age"]]
)

X_test_age = age_imputer.transform(
    X_test[["Age"]]
)


# ----------------------------
# Impute Embarked
# ----------------------------

embarked_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train_embarked = embarked_imputer.fit_transform(
    X_train[["Embarked"]]
)

X_test_embarked = embarked_imputer.transform(
    X_test[["Embarked"]]
)


# ============================================================
# 6. One-Hot Encode Sex and Embarked
# ============================================================
# handle_unknown="ignore" prevents errors if a category appears
# in test data that was not present during training.
#
# sparse_output=False is the modern scikit-learn parameter.
# The fallback keeps the code compatible with older versions.

try:
    sex_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    embarked_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

except TypeError:
    # Compatibility with older scikit-learn versions.
    sex_encoder = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    )

    embarked_encoder = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    )


X_train_sex = sex_encoder.fit_transform(
    X_train[["Sex"]]
)

X_test_sex = sex_encoder.transform(
    X_test[["Sex"]]
)


X_train_embarked_encoded = embarked_encoder.fit_transform(
    X_train_embarked
)

X_test_embarked_encoded = embarked_encoder.transform(
    X_test_embarked
)


# ============================================================
# 7. Extract Remaining Numerical Features
# ============================================================

remaining_columns = [
    "Pclass",
    "SibSp",
    "Parch",
    "Fare"
]

X_train_remaining = X_train[remaining_columns].values
X_test_remaining = X_test[remaining_columns].values


# ============================================================
# 8. Combine All Manually Transformed Features
# ============================================================

X_train_transformed = np.concatenate(
    (
        X_train_remaining,
        X_train_age,
        X_train_sex,
        X_train_embarked_encoded
    ),
    axis=1
)

X_test_transformed = np.concatenate(
    (
        X_test_remaining,
        X_test_age,
        X_test_sex,
        X_test_embarked_encoded
    ),
    axis=1
)

print("\nManually transformed training shape:")
print(X_train_transformed.shape)

print("\nManually transformed test shape:")
print(X_test_transformed.shape)


# ============================================================
# 9. Train Decision Tree
# ============================================================
# Decision trees do not require feature scaling, so MinMaxScaler
# is intentionally not used here.

classifier = DecisionTreeClassifier(
    random_state=42
)

classifier.fit(
    X_train_transformed,
    y_train
)


# ============================================================
# 10. Make Predictions
# ============================================================

y_pred = classifier.predict(X_test_transformed)

print("\nPredictions:")
print(y_pred)


# ============================================================
# 11. Evaluate Model
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# ============================================================
# 12. ColumnTransformer — Cleaner Approach
# ============================================================
# Instead of manually transforming and concatenating every
# feature, ColumnTransformer lets us define the complete
# preprocessing process in one object.
#
# This is the approach you would generally prefer in a real
# ML pipeline.


try:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:
    # Compatibility with older scikit-learn versions.
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


preprocessor = ColumnTransformer(
    transformers=[
        (
            "age_imputer",
            SimpleImputer(strategy="mean"),
            ["Age"]
        ),
        (
            "embarked_imputer_encoder",
            # First impute missing Embarked values,
            # then one-hot encode them.
            # A nested pipeline is cleaner for this case.
            # This transformer is replaced below with a Pipeline.
            SimpleImputer(strategy="most_frequent"),
            ["Embarked"]
        ),
        (
            "sex_encoder",
            one_hot_encoder,
            ["Sex"]
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 13. Recommended ColumnTransformer with Pipelines
# ============================================================

from sklearn.pipeline import Pipeline


# Fresh encoders are used because preprocessing objects should
# have a single clear fitting lifecycle.

try:
    sex_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    embarked_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:
    sex_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

    embarked_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


age_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="mean")
        )
    ]
)

embarked_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            embarked_ohe
        )
    ]
)

sex_pipeline = Pipeline(
    steps=[
        (
            "encoder",
            sex_ohe
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "age",
            age_pipeline,
            ["Age"]
        ),
        (
            "embarked",
            embarked_pipeline,
            ["Embarked"]
        ),
        (
            "sex",
            sex_pipeline,
            ["Sex"]
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 14. Transform Data using ColumnTransformer
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nColumnTransformer training shape:")
print(X_train_processed.shape)

print("\nColumnTransformer test shape:")
print(X_test_processed.shape)


# ============================================================
# 15. Train Decision Tree using ColumnTransformer Output
# ============================================================

classifier_processed = DecisionTreeClassifier(
    random_state=42
)

classifier_processed.fit(
    X_train_processed,
    y_train
)

y_pred_processed = classifier_processed.predict(
    X_test_processed
)

processed_accuracy = accuracy_score(
    y_test,
    y_pred_processed
)

print("\nColumnTransformer model accuracy:")
print(processed_accuracy)


# ============================================================
# 16. Save Model and Preprocessor
# ============================================================
# Saving the preprocessor together with the model is important:
# future data must go through exactly the same transformations
# before prediction.

os.makedirs("models", exist_ok=True)

with open("models/preprocessor.pkl", "wb") as file:
    pickle.dump(preprocessor, file)

with open("models/classifier.pkl", "wb") as file:
    pickle.dump(classifier_processed, file)

print("\nModel and preprocessor saved successfully.")