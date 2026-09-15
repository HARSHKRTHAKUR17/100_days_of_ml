"""
Outlier Treatment — Trimming using IQR

Dataset expected:
    placement.csv

Trimming removes rows containing outliers from the selected feature.
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

    outliers = df[
        (df[COLUMN] < lower_limit) | (df[COLUMN] > upper_limit)
    ]

    print("\nOutliers:")
    print(outliers)

    # Trim both lower and upper outliers.
    trimmed_df = df[
        (df[COLUMN] >= lower_limit) & (df[COLUMN] <= upper_limit)
    ].copy()

    print("\nTrimmed shape:", trimmed_df.shape)
    print("Rows removed:", len(df) - len(trimmed_df))

    print("\nStatistics after trimming:")
    print(trimmed_df[COLUMN].describe())

    # Compare distributions before and after trimming.
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

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