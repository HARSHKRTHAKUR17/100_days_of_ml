"""
Outlier Treatment — Capping (Winsorization) using Percentiles

Dataset expected:
    weight-height.csv

Capping keeps every observation but replaces values outside the chosen
percentile limits with the corresponding boundary values.
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

    capped_df = df.copy()
    capped_df[COLUMN] = capped_df[COLUMN].clip(
        lower=lower_limit,
        upper=upper_limit,
    )

    print("\nShape after capping:", capped_df.shape)
    print("Rows removed: 0")
    print("\nStatistics after capping:")
    print(capped_df[COLUMN].describe())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sns.histplot(df[COLUMN], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Before Capping — Distribution")

    sns.boxplot(x=df[COLUMN], ax=axes[0, 1])
    axes[0, 1].set_title("Before Capping — Boxplot")

    sns.histplot(capped_df[COLUMN], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("After Capping — Distribution")

    sns.boxplot(x=capped_df[COLUMN], ax=axes[1, 1])
    axes[1, 1].set_title("After Capping — Boxplot")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()