"""
Titanic — Mixed Feature Extraction

Demonstrates extracting useful components from mixed-format features:
- Numeric and categorical parts of `number`
- Numeric and categorical parts of `Cabin`

Dataset expected: titanic.csv
Expected columns: Cabin, Ticket, number, Survived
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_number_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric and categorical components from `number`."""
    df = df.copy()

    df["number_numerical"] = pd.to_numeric(
        df["number"],
        errors="coerce",
        downcast="integer",
    )

    df["number_categorical"] = np.where(
        df["number_numerical"].isna(),
        df["number"],
        np.nan,
    )

    return df


def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cabin number and deck/category from `Cabin`."""
    df = df.copy()

    df["cabin_num"] = pd.to_numeric(
        df["Cabin"].str.extract(r"(\d+)", expand=False),
        errors="coerce",
        downcast="integer",
    )

    df["cabin_cat"] = df["Cabin"].str[0]

    return df


def plot_value_counts(series: pd.Series, title: str, xlabel: str) -> None:
    """Plot value counts for a categorical feature."""
    ax = series.value_counts(dropna=False).plot.bar()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load the Titanic dataset
    # ------------------------------------------------------------------
    df = pd.read_csv("titanic.csv")

    print("First five rows:")
    print(df.head())

    # ------------------------------------------------------------------
    # 2. Inspect the mixed `number` feature
    # ------------------------------------------------------------------
    print("\nUnique values in `number`:")
    print(df["number"].unique())

    plot_value_counts(
        df["number"],
        title="Passengers travelling with",
        xlabel="number",
    )

    # ------------------------------------------------------------------
    # 3. Extract numeric and categorical parts of `number`
    # ------------------------------------------------------------------
    df = extract_number_features(df)

    print("\nExtracted `number` features:")
    print(
        df[
            ["number", "number_numerical", "number_categorical"]
        ].head()
    )

    # ------------------------------------------------------------------
    # 4. Inspect Cabin and Ticket values
    # ------------------------------------------------------------------
    print("\nUnique Cabin values:")
    print(df["Cabin"].unique())

    print("\nUnique Ticket values:")
    print(df["Ticket"].unique())

    # ------------------------------------------------------------------
    # 5. Extract numeric and categorical parts of Cabin
    # ------------------------------------------------------------------
    df = extract_cabin_features(df)

    print("\nExtracted Cabin features:")
    print(df[["Cabin", "cabin_num", "cabin_cat"]].head())

    # ------------------------------------------------------------------
    # 6. Inspect extracted cabin categories
    # ------------------------------------------------------------------
    plot_value_counts(
        df["cabin_cat"],
        title="Cabin deck/category distribution",
        xlabel="Cabin category",
    )

    # ------------------------------------------------------------------
    # 7. Final inspection
    # ------------------------------------------------------------------
    engineered_columns = [
        "number_numerical",
        "number_categorical",
        "cabin_num",
        "cabin_cat",
    ]

    print("\nEngineered features:")
    print(df[engineered_columns].head())

    print("\nMissing values in engineered features:")
    print(df[engineered_columns].isna().sum())


if __name__ == "__main__":
    main()