import pandas as pd
from ydata_profiling import ProfileReport


# Load dataset
df = pd.read_csv("train.csv")


# Quick inspection
print(df.head())


# Generate automated EDA report
profile = ProfileReport(
    df,
    title="Titanic Dataset Profiling Report",
    explorative=True
)


# Save report
profile.to_file("output.html")