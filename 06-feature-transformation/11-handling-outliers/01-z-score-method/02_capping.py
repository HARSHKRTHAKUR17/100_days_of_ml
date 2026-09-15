"""
Outlier Treatment — Capping

Dataset expected: placement.csv
"""

from __future__ import annotations

import pandas as pd

Z_THRESHOLD = 3


def main() -> None:
    df = pd.read_csv("placement.csv")

    print("Original shape:", df.shape)

    # Calculate mean ± 3 standard deviations.
    mean = df["cgpa"].mean()
    std = df["cgpa"].std()

    lower_limit = mean - Z_THRESHOLD * std
    upper_limit = mean + Z_THRESHOLD * std

    print("\nCapping limits:")
    print(f"Lower limit: {lower_limit:.4f}")
    print(f"Upper limit: {upper_limit:.4f}")

    # Capping changes extreme values but keeps every row.
    capped_df = df.copy()

    capped_df["cgpa"] = capped_df["cgpa"].clip(
        lower=lower_limit,
        upper=upper_limit,
    )

    print("\nOriginal shape:", df.shape)
    print("Shape after capping:", capped_df.shape)

    print("\nCGPA before capping:")
    print(df["cgpa"].describe())

    print("\nCGPA after capping:")
    print(capped_df["cgpa"].describe())

    affected = df["cgpa"] != capped_df["cgpa"]

    print("\nValues affected by capping:")
    print(
        pd.DataFrame(
            {
                "original_cgpa": df.loc[affected, "cgpa"],
                "capped_cgpa": capped_df.loc[affected, "cgpa"],
            }
        )
    )


if __name__ == "__main__":
    main()