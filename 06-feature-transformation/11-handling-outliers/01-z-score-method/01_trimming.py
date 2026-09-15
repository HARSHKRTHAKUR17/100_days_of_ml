"""
Outlier Treatment — Trimming

Dataset expected: placement.csv
"""

from __future__ import annotations

import pandas as pd

LOWER_LIMIT = 5.11
UPPER_LIMIT = 8.80
Z_THRESHOLD = 3


def main() -> None:
    df = pd.read_csv("placement.csv")

    print("Original shape:", df.shape)

    # Approach 1: Trim using fixed lower and upper limits.
    trimmed_df = df[
        (df["cgpa"] > LOWER_LIMIT)
        & (df["cgpa"] < UPPER_LIMIT)
    ].copy()

    print("\nFixed-limit trimming")
    print(f"Lower limit: {LOWER_LIMIT}")
    print(f"Upper limit: {UPPER_LIMIT}")
    print("Trimmed shape:", trimmed_df.shape)
    print("Rows removed:", len(df) - len(trimmed_df))

    # Approach 2: Trim using the Z-score method.
    df_with_zscore = df.copy()

    mean = df_with_zscore["cgpa"].mean()
    std = df_with_zscore["cgpa"].std()

    df_with_zscore["cgpa_zscore"] = (
        (df_with_zscore["cgpa"] - mean) / std
    )

    zscore_trimmed_df = df_with_zscore[
        df_with_zscore["cgpa_zscore"].abs() < Z_THRESHOLD
    ].copy()

    print("\nZ-score trimming")
    print(f"Z-score threshold: ±{Z_THRESHOLD}")
    print("Trimmed shape:", zscore_trimmed_df.shape)
    print("Rows removed:", len(df) - len(zscore_trimmed_df))

    print("\nOutliers identified by Z-score:")
    print(
        df_with_zscore[
            df_with_zscore["cgpa_zscore"].abs() > Z_THRESHOLD
        ]
    )


if __name__ == "__main__":
    main()
