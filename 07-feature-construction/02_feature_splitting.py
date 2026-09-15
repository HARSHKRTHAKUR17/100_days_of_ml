"""
Titanic Feature Splitting — Name Features

Demonstrates:
- Splitting a composite text feature into useful components
- Extracting a passenger's title from Name
- Comparing survival rates across titles
- Creating a binary Is_Married feature from the extracted title

Dataset expected:
    train.csv
"""

from __future__ import annotations

import pandas as pd


def main() -> None:
    df = pd.read_csv("train.csv")

    print("First five rows:")
    print(df.head())

    # Extract the title from names such as:
    # "Braund, Mr. Owen Harris" -> "Mr"
    df["Title"] = (
        df["Name"]
        .str.split(", ", expand=True)[1]
        .str.split(".", expand=True)[0]
    )

    print("\nTitle and original Name:")
    print(df[["Title", "Name"]].head())

    print("\nSurvival rate by title:")
    title_survival = (
        df.groupby("Title")["Survived"]
        .mean()
        .sort_values(ascending=False)
    )
    print(title_survival)

    # Avoid chained assignment / SettingWithCopyWarning.
    df["Is_Married"] = (df["Title"] == "Mrs").astype(int)

    print("\nIs_Married feature:")
    print(df[["Name", "Title", "Is_Married"]].head())

    print("\nIs_Married value counts:")
    print(df["Is_Married"].value_counts())


if __name__ == "__main__":
    main()