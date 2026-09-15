"""
Outlier Treatment — Trimming using Percentiles

Dataset expected:
    weight-height.csv

Trimming removes observations outside the chosen percentile limits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLUMN = "Height"
LOWER_PERCENTILE = 0.01
UPPER_PERCENTILE = 0.99


def main() -> None:
    df = pd.read_csv("weight-height.csv")

    print("Original shape:", df.shape)
    print("\nOriginal statistics:")
    print(df[COLUMN].describe())

    lower_limit = df[COLUMN].quantile(LOWER_PERCENTILE)
    upper_limit = df[COLUMN].quantile(UPPER_PERCENTILE)

    print(f"\nLower limit ({LOWER_PERCENTILE:.0%}): {lower_limit:.4f}")
    print(f"Upper limit ({UPPER_PERCENTILE:.0%}): {upper_limit:.4f}")

    trimmed_df = df[
        (df[COLUMN] >= lower_limit) & (df[COLUMN] <= upper_limit)
    ].copy()

    print("\nTrimmed shape:", trimmed_df.shape)
    print("Rows removed:", len(df) - len(trimmed_df))
    print("\nStatistics after trimming:")
    print(trimmed_df[COLUMN].describe())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sns.histplot(df[COLUMN], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Before Trimming — Distribution")

    sns.boxplot(x=df[COLUMN], ax=axes[0, 1])
    axes[0, 1].set_title("Before Trimming — Boxplot")

    sns.histplot(trimmed_df[COLUMN], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("After Trimming — Distribution")

    sns.boxplot(x=trimmed_df[COLUMN], ax=axes[1, 1])
    axes[1, 1].set_title("After Trimming — Boxplot")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()