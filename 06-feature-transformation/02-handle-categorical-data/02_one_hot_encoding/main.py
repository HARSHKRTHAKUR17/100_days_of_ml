import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. Load and Inspect Dataset
# ============================================================

df = pd.read_csv("cars.csv")

print("First five rows:")
print(df.head())

print("\nOwner value counts:")
print(df["owner"].value_counts())


# ============================================================
# 2. One-Hot Encoding using Pandas
# ============================================================

df_one_hot = pd.get_dummies(
    df,
    columns=["fuel", "owner"]
)

print("\nOne-Hot Encoded Dataset:")
print(df_one_hot.head())


# ============================================================
# 3. K-1 One-Hot Encoding
# ============================================================
# drop_first=True removes one category from each
# categorical variable.

df_k1 = pd.get_dummies(
    df,
    columns=["fuel", "owner"],
    drop_first=True
)

print("\nK-1 One-Hot Encoded Dataset:")
print(df_k1.head())


# ============================================================
# 4. One-Hot Encoding using Scikit-Learn
# ============================================================

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

print("\nTraining Features:")
print(X_train.head())


# Encode fuel and owner.
# drop="first" performs K-1 encoding.

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    dtype=np.int32,
    handle_unknown="ignore"
)

X_train_encoded = ohe.fit_transform(
    X_train[["fuel", "owner"]]
)

X_test_encoded = ohe.transform(
    X_test[["fuel", "owner"]]
)

print("\nEncoded Training Shape:")
print(X_train_encoded.shape)

print("\nEncoded Training Data:")
print(X_train_encoded)


# ============================================================
# 5. Combine Numerical and Encoded Features
# ============================================================

X_train_final = np.hstack(
    (
        X_train[["brand", "km_driven"]].values,
        X_train_encoded
    )
)

print("\nFinal Training Feature Matrix:")
print(X_train_final)


# ============================================================
# 6. One-Hot Encoding with Top Categories
# ============================================================
# Keep frequently occurring brands as separate categories.
# Replace less frequent brands with "uncommon".

counts = df["brand"].value_counts()

print("\nNumber of Unique Brands:")
print(df["brand"].nunique())

threshold = 100

repl = counts[counts <= threshold].index

df["brand_grouped"] = df["brand"].replace(
    repl,
    "uncommon"
)

brand_encoded = pd.get_dummies(
    df["brand_grouped"]
)

print("\nTop-Category Encoded Brands:")
print(brand_encoded.sample(5))