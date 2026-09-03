import seaborn as sns
import matplotlib.pyplot as plt


# Load the Iris dataset.
iris = sns.load_dataset("iris")

# Inspect the data.
print(iris.head())


# Pairplot compares every numerical feature with
# every other numerical feature.
# hue separates the three flower species.
sns.pairplot(
    iris,
    hue="species"
)

plt.show()
