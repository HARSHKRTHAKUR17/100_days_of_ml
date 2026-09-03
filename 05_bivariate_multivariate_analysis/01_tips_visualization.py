import seaborn as sns
import matplotlib.pyplot as plt


# Load Seaborn's sample restaurant tips dataset.
tips = sns.load_dataset("tips")

# Inspect the data.
print(tips.head())


# Scatterplot: numerical vs numerical
# total_bill -> tip
# hue -> color/group by sex
# style -> marker style by smoker
# size -> marker size by party size
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="sex",
    style="smoker",
    size="size"
)

plt.title("Total Bill vs Tip")
plt.show()