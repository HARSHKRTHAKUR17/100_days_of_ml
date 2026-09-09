import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("customer.csv")

print("Sample data:")
print(df.sample(5))

# Keep the categorical feature columns and target column.
# Original notebook removed the first two columns.
df = df.iloc[:, 2:]

print("\nDataset after removing the first two columns:")
print(df.head())


# ============================================================
# 2. Separate features and target
# ============================================================

X = df.drop("purchased", axis=1)
y = df["purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining features:")
print(X_train)


# ============================================================
# 3. Ordinal Encoding
# ============================================================
# The categories have a natural order:
#
# review:
# Poor < Average < Good
#
# education:
# School < UG < PG

oe = OrdinalEncoder(
    categories=[
        ["Poor", "Average", "Good"],
        ["School", "UG", "PG"]
    ]
)

oe.fit(X_train)

X_train_encoded = oe.transform(X_train)
X_test_encoded = oe.transform(X_test)

print("\nOrdinal-encoded training data:")
print(X_train_encoded)

print("\nOrdinal encoder categories:")
print(oe.categories_)


# ============================================================
# 4. Label Encoding
# ============================================================
# Encode the target:
# No  -> 0
# Yes -> 1

le = LabelEncoder()

le.fit(y_train)

print("\nLabel encoder classes:")
print(le.classes_)

y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

print("\nEncoded training target:")
print(y_train_encoded)