import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset.
df = pd.read_csv("train.csv")


# -----------------------------
# 1. Countplot: Embarked
# -----------------------------

# Count how many passengers embarked from each port.
sns.countplot(data=df, x="Embarked")

plt.title("Passengers by Embarkation Port")
plt.xlabel("Embarked")
plt.ylabel("Number of Passengers")
plt.show()


# -----------------------------
# 2. Countplot: Sex
# -----------------------------

# Count passengers by gender.
sns.countplot(data=df, x="Sex")

plt.title("Passengers by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.show()


# -----------------------------
# 3. Pie chart: Sex
# -----------------------------

# value_counts() gives the frequency of each category.
sex_counts = df["Sex"].value_counts()

sex_counts.plot(
    kind="pie",
    autopct="%.2f%%",
    ylabel=""
)

plt.title("Passenger Distribution by Sex")
plt.show()
