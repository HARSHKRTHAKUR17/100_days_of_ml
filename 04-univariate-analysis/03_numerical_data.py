import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset.
df = pd.read_csv("train.csv")


# -----------------------------
# 1. Histogram: Age
# -----------------------------

# A histogram shows how numerical values are distributed.
plt.hist(df["Age"].dropna(), bins=5)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()


# -----------------------------
# 2. Distribution plot: Age
# -----------------------------

# Modern replacement for the older seaborn distplot().
sns.histplot(data=df, x="Age", kde=True)

plt.title("Age Distribution with Density Curve")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# -----------------------------
# 3. Boxplot: Age
# -----------------------------

# A boxplot helps identify the median, spread and possible outliers.
sns.boxplot(data=df, x="Age")

plt.title("Age Boxplot")
plt.xlabel("Age")
plt.show()


# -----------------------------
# 4. Basic statistics
# -----------------------------

print("Minimum age:", df["Age"].min())
print("Maximum age:", df["Age"].max())
print("Mean age:", df["Age"].mean())

# Skewness describes asymmetry in the distribution.
print("Age skewness:", df["Age"].skew())
