"""
Outlier Treatment — Capping using IQR

Dataset expected:
    placement.csv

Capping keeps every row but replaces values outside the IQR limits
with the corresponding lower or upper limit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLUMN = "placement_exam_marks"


def main() -> None:
    df = pd.read_csv("placement.csv")

    print("Original shape:", df.shape)
    print("\nOriginal statistics:")
    print(df[COLUMN].describe())

    # Calculate the IQR limits.
    percentile25 = df[COLUMN].quantile(0.25)
    percentile75 = df[COLUMN].quantile(0.75)
    iqr = percentile75 - percentile25

    lower_limit = percentile25 - 1.5 * iqr
    upper_limit = percentile75 + 1.5 * iqr

    print("\nIQR values:")
    print(f"Q1: {percentile25:.4f}")
    print(f"Q3: {percentile75:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Lower limit: {lower_limit:.4f}")
    print(f"Upper limit: {upper_limit:.4f}")

    capped_df = df.copy()

    # Cap lower outliers at the lower limit and upper outliers at the upper limit.
    capped_df[COLUMN] = capped_df[COLUMN].clip(
        lower=lower_limit,
        upper=upper_limit,
    )

    print("\nShape after capping:", capped_df.shape)
    print("Rows removed: 0")

    affected = df[COLUMN] != capped_df[COLUMN]

    print("\nValues affected by capping:")
    print(
        pd.DataFrame(
            {
                "original": df.loc[affected, COLUMN],
                "capped": capped_df.loc[affected, COLUMN],
            }
        )
    )

    print("\nStatistics after capping:")
    print(capped_df[COLUMN].describe())

    # Compare distributions before and after capping.
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

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