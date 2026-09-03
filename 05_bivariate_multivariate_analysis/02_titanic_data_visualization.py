import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the Titanic dataset you already have.
titanic = pd.read_csv("train.csv")

# Inspect the dataset.
print(titanic.head())
print(titanic.head(3))


# ---------------------------------
# 1. Bar Plot
# ---------------------------------

# Compare average Age across passenger classes.
# hue separates the result by Sex.
sns.barplot(
    data=titanic,
    x="Pclass",
    y="Age",
    hue="Sex"
)

plt.title("Age by Passenger Class and Sex")
plt.show()


# ---------------------------------
# 2. Box Plot
# ---------------------------------

# Compare the Age distribution by Sex.
# hue separates survivors from non-survivors.
sns.boxplot(
    data=titanic,
    x="Sex",
    y="Age",
    hue="Survived"
)

plt.title("Age by Sex and Survival")
plt.show()


# ---------------------------------
# 3. Distribution Plot
# ---------------------------------

# Compare the Age distributions of passengers
# who survived vs passengers who did not.
#
# distplot() from the original notebook is deprecated,
# so histplot/kdeplot is used instead.
sns.kdeplot(
    data=titanic[titanic["Survived"] == 0],
    x="Age",
    label="Did Not Survive"
)

sns.kdeplot(
    data=titanic[titanic["Survived"] == 1],
    x="Age",
    label="Survived"
)

plt.title("Age Distribution by Survival")
plt.legend()
plt.show()


# ---------------------------------
# 4. Heatmap with Crosstab
# ---------------------------------

# Crosstab creates a frequency table:
# Passenger Class x Survival.
class_survival = pd.crosstab(
    titanic["Pclass"],
    titanic["Survived"]
)

sns.heatmap(
    class_survival,
    annot=True,
    fmt="d"
)

plt.title("Passenger Class vs Survival")
plt.show()


# ---------------------------------
# 5. GroupBy + Mean
# ---------------------------------

# Since Survived is 0/1, its mean gives the survival rate.
survival_rate = titanic.groupby("Embarked")["Survived"].mean() * 100

print("\nSurvival percentage by embarkation port:")
print(survival_rate)


# ---------------------------------
# 6. Clustermap
# ---------------------------------

# Show the relationship between number of parents/children
# and survival using a clustered heatmap.
parch_survival = pd.crosstab(
    titanic["Parch"],
    titanic["Survived"]
)

sns.clustermap(
    parch_survival,
    annot=True,
    fmt="d"
)

plt.show()
