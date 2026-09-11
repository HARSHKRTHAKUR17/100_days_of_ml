from __future__ import annotations

import numpy as np
import pandas as pd


def main() -> None:
    # ------------------------------------------------------------------
    # Load and inspect the dataset
    # ------------------------------------------------------------------
    df = pd.read_csv("data_science_job.csv")

    print("First five rows:")
    print(df.head())

    print("\nMissing-value percentage by column:")
    missing_percentage = df.isna().mean().mul(100)
    print(missing_percentage)

    print(f"\nDataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # Select columns with a small amount of missing data
    # ------------------------------------------------------------------
    # The notebook uses:
    #   0% < missing percentage < 5%
    #
    # This identifies columns where relatively few observations are
    # missing and complete-case analysis may be worth demonstrating.
    cols = [
        column
        for column in df.columns
        if 0 < df[column].isna().mean() < 0.05
    ]

    print("\nColumns with between 0% and 5% missing values:")
    print(cols)

    print("\nSample of selected columns:")
    print(df[cols].sample(5, random_state=42))

    # ------------------------------------------------------------------
    # Inspect one categorical variable
    # ------------------------------------------------------------------
    print("\nEducation-level value counts:")
    print(df["education_level"].value_counts())

    # ------------------------------------------------------------------
    # Complete-case analysis
    # ------------------------------------------------------------------
    # Keep only rows that have no missing values in the selected columns.
    new_df = df[cols].dropna()

    retention_ratio = len(new_df) / len(df)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nShape after dropping rows with missing values in selected columns:")
    print(new_df.shape)

    print(f"\nFraction of rows retained: {retention_ratio:.4f}")
    print(f"Percentage of rows retained: {retention_ratio:.2%}")
    print(f"Percentage of rows removed: {(1 - retention_ratio):.2%}")

    # ------------------------------------------------------------------
    # Compare distributions before and after complete-case analysis
    # ------------------------------------------------------------------
    # The notebook uses plots to compare the original and reduced data.
    # Here we provide a compact numerical comparison as well, which is
    # useful when running the script from VS Code.
    numeric_cols = new_df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        print("\nOriginal numeric means:")
        print(df[numeric_cols].mean())

        print("\nComplete-case numeric means:")
        print(new_df[numeric_cols].mean())


if __name__ == "__main__":
    main()