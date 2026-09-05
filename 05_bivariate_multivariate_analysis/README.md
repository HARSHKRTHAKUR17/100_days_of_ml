# Bivariate & Multivariate Analysis

This folder contains exploratory data analysis focused on understanding
relationships between two or more variables.

## What I Learned

### Bivariate Analysis

- Numerical vs numerical relationships
- Numerical vs categorical relationships
- Categorical vs categorical relationships
- Scatterplots
- Barplots
- Boxplots
- Crosstabs
- GroupBy analysis

### Multivariate Analysis

- Pairplots
- Heatmaps
- Clustermaps
- Using multiple variables in a single visualization
- Using `hue` to compare groups
- Pivot tables
- Correlation/relationship exploration

## Visualizations

### Scatterplot

Used to investigate relationships between two numerical variables.

### Barplot

Used to compare numerical values across categories.

### Boxplot

Used to compare distributions across groups and identify potential outliers.

### Heatmap

Used to visualize a matrix of values, such as counts or correlations.

### Pairplot

Used to examine relationships among several numerical features at once.

### Clustermap

Used to visualize a matrix while also grouping similar rows/columns.

## Datasets

The analysis work used datasets including:

- Titanic
- Tips
- Iris
- Flights

## Examples

Titanic:

- Passenger class vs age
- Sex vs age
- Passenger class vs survival
- Embarkation port vs survival
- Parch vs survival

Tips:

- Total bill vs tip
- Using `hue`, `style`, and `size`

Iris:

- Relationships among sepal and petal measurements
- Species comparison using `hue`

Flights:

- Passenger trends over time
- GroupBy aggregation
- Pivot tables
- Clustermap

## Key Takeaway

The purpose of bivariate and multivariate analysis is to move from:

"What does one feature look like?"

to:

"How do features relate to one another?"

This is an important step before feature selection, hypothesis formation,
and ML modeling.