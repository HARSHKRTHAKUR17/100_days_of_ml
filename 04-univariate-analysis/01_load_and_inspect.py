import pandas as pd


# Load the dataset into a DataFrame.
df = pd.read_csv("train.csv")

# Display the first five rows.
print(df.head())

# Useful basic information about the dataset.
print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
