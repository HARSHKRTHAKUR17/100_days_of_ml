# Univariate Analysis

This folder contains the foundational exploratory data analysis work focused
on understanding one variable at a time.

## What I Learned

- Understanding numerical vs categorical variables
- Frequency distributions
- `value_counts()`
- Histograms
- Distribution plots
- KDE curves
- Boxplots
- Mean, minimum and maximum
- Skewness
- Identifying spread and potential outliers
- Understanding the shape of distributions

## Numerical Analysis

For numerical variables, I practiced:

- Histogram
- Histogram + KDE
- Boxplot
- Mean
- Minimum
- Maximum
- Skewness

Example dataset:
Titanic `train.csv`

Important variables included:

- `Age`
- `Fare`
- `SibSp`
- `Parch`

## Categorical Analysis

For categorical variables, I practiced:

- Countplots
- Frequency counts
- Pie charts
- Comparing category frequencies

Examples included:

- `Sex`
- `Embarked`
- `Pclass`

## Why This Matters for ML

Univariate analysis is one of the first steps when receiving a new dataset.

Before building a model, I should understand:

- What values does a feature contain?
- Is the feature heavily skewed?
- Are there unusual values?
- Are there missing values?
- Are some categories extremely rare?
- Does the feature have a sensible distribution?

## Main Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn

## Key Takeaway

Univariate analysis teaches me to understand individual features before
looking for relationships between features or building ML models.