import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the flights dataset.
flights = sns.load_dataset("flights")

# Inspect the data.
print(flights.head())


# ---------------------------------
# 1. Aggregate passengers by year
# ---------------------------------

# Sum the number of passengers for each year.
new = flights.groupby("year")["passengers"].sum().reset_index()

print("\nPassengers by year:")
print(new)


# ---------------------------------
# 2. Line Plot
# ---------------------------------

# Show how total passengers changed over time.
sns.lineplot(
    data=new,
    x="year",
    y="passengers"
)

plt.title("Total Passengers by Year")
plt.show()


# ---------------------------------
# 3. Pivot Table + Clustermap
# ---------------------------------

# Convert the dataset into a matrix:
# rows    -> month
# columns -> year
# values  -> passengers
flight_matrix = flights.pivot_table(
    values="passengers",
    index="month",
    columns="year"
)

sns.clustermap(
    flight_matrix
)

plt.show()
