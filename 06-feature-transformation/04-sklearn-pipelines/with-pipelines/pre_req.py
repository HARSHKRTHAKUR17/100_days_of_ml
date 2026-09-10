"""
Titanic: End-to-End Scikit-Learn Pipeline

Concepts covered:
- Train/test split
- Missing-value imputation
- One-hot encoding
- Feature scaling
- Feature selection with SelectKBest + chi2
- Decision tree classification
- Pipeline
- make_pipeline
- Cross-validation
- GridSearchCV
- Pipeline export

Dataset expected:
    train.csv

Expected Titanic columns:
    PassengerId, Survived, Pclass, Name, Sex, Age, SibSp,
    Parch, Ticket, Fare, Cabin, Embarked
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# 1. Load and prepare the data
# =========================================================

df = pd.read_csv("train.csv")

# Remove columns that are not being used as model features.
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

X = df.drop(columns="Survived")
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# =========================================================
# 2. Preprocessing
# =========================================================
# Numerical columns:
#   Age -> median imputation
#   Pclass, SibSp, Parch, Fare -> kept as numerical
#
# Categorical columns:
#   Sex -> most-frequent imputation + one-hot encoding
#   Embarked -> most-frequent imputation + one-hot encoding
#
# Using column names instead of positional indices makes the
# pipeline robust and avoids feature-ordering mistakes.

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# =========================================================
# 3. Scaling
# =========================================================
# MinMaxScaler is used here because chi2 requires
# non-negative feature values.
#
# Decision trees themselves do NOT require scaling.
# Scaling is included because this pipeline is demonstrating
# the interaction between preprocessing and chi-square feature
# selection.

scaler = MinMaxScaler()


# =========================================================
# 4. Feature selection
# =========================================================
# Select the 8 features with the highest chi-square scores.
#
# chi2 requires non-negative features, which is why MinMaxScaler
# comes before SelectKBest.

feature_selector = SelectKBest(
    score_func=chi2,
    k=8,
)


# =========================================================
# 5. Model
# =========================================================

classifier = DecisionTreeClassifier(random_state=42)


# =========================================================
# 6. Create the Pipeline
# =========================================================

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", scaler),
        ("feature_selector", feature_selector),
        ("classifier", classifier),
    ]
)


# =========================================================
# 7. Pipeline vs make_pipeline
# =========================================================
# Pipeline requires explicit step names.
#
# make_pipeline automatically generates step names.
#
# Both approaches are valid. We use `pipe` for the rest of
# this script because named steps are especially useful when
# accessing parameters during GridSearchCV.

# Equivalent syntax:
alternative_pipe = make_pipeline(
    preprocessor,
    scaler,
    feature_selector,
    classifier,
)


# =========================================================
# 8. Train the Pipeline
# =========================================================

pipe.fit(X_train, y_train)


# =========================================================
# 9. Explore the Pipeline
# =========================================================

print("\nPipeline steps:")
print(pipe.named_steps)


# =========================================================
# 10. Make predictions
# =========================================================

y_pred = pipe.predict(X_test)

print("\nTest predictions:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.4f}")


# =========================================================
# 11. Cross-validation
# =========================================================

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

print("\nCross-validation scores:")
print(cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")


# =========================================================
# 12. GridSearchCV using the Pipeline
# =========================================================
# Parameters inside a Pipeline are accessed using:
#
#     step_name__parameter_name
#
# For example:
#     classifier__max_depth

params = {
    "classifier__max_depth": [1, 2, 3, 4, 5, None],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print("\nBest CV score:")
print(f"{grid.best_score_:.4f}")

print("\nBest parameters:")
print(grid.best_params_)


# =========================================================
# 13. Evaluate the best pipeline
# =========================================================

best_model = grid.best_estimator_
best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_predictions)

print("\nBest model test accuracy:")
print(f"{best_accuracy:.4f}")


# =========================================================
# 14. Export the fitted Pipeline
# =========================================================
# Exporting the complete pipeline is preferable to separately
# saving each preprocessing object and the classifier.
#
# The exported pipeline contains:
#   preprocessing
#   scaling
#   feature selection
#   trained classifier
#
# Therefore, new raw Titanic rows can be passed directly to
# `pipeline.predict()` after loading it.

model_path = Path("pipe.pkl")

with model_path.open("wb") as file:
    pickle.dump(best_model, file)

print(f"\nPipeline saved to: {model_path}")


# NOTE:
# Only load pickle files that you trust. Pickle can execute
# arbitrary code during deserialization.
