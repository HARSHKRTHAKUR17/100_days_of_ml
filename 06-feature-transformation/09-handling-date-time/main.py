from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def main() -> None:
    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    orders = pd.read_csv("orders.csv")
    messages = pd.read_csv("messages.csv")

    print("Orders:")
    print(orders.head())

    print("\nMessages:")
    print(messages.head())

    print("\nOrders info:")
    orders.info()

    print("\nMessages info:")
    messages.info()

    # ------------------------------------------------------------------
    # Working with dates
    # ------------------------------------------------------------------
    orders["date"] = pd.to_datetime(orders["date"])
    print("\nOrders after datetime conversion:")
    orders.info()

    # 1. Extract year
    orders["date_year"] = orders["date"].dt.year

    # 2. Extract month number
    orders["date_month_no"] = orders["date"].dt.month

    # 3. Extract month name
    orders["date_month_name"] = orders["date"].dt.month_name()

    # 4. Extract day of month
    orders["date_day"] = orders["date"].dt.day

    # 5. Extract day of week as a number
    # Monday = 0, Sunday = 6
    orders["date_dow"] = orders["date"].dt.dayofweek

    # 6. Extract day of week name
    orders["date_dow_name"] = orders["date"].dt.day_name()

    # 7. Is it a weekend?
    orders["date_is_weekend"] = orders["date_dow"].isin([5, 6]).astype(int)

    # 8. Extract ISO week of year.
    # `.dt.week` is deprecated/removed in modern pandas.
    orders["date_week"] = orders["date"].dt.isocalendar().week.astype("Int64")

    # 9. Extract quarter
    orders["quarter"] = orders["date"].dt.quarter

    # 10. Extract semester
    orders["semester"] = np.where(orders["quarter"].isin([1, 2]), 1, 2)

    date_features = [
        "date",
        "date_year",
        "date_month_no",
        "date_month_name",
        "date_day",
        "date_dow",
        "date_dow_name",
        "date_is_weekend",
        "date_week",
        "quarter",
        "semester",
    ]

    print("\nExtracted date features:")
    print(orders[date_features].head())

    # ------------------------------------------------------------------
    # Extract elapsed time between dates
    # ------------------------------------------------------------------
    # The notebook uses today's datetime as the reference point.
    today = datetime.today()

    elapsed = today - orders["date"]

    print("\nElapsed time as a timedelta:")
    print(elapsed.head())

    print("\nElapsed time in days:")
    print(elapsed.dt.days.head())

    # The original notebook approximates months using a NumPy month
    # duration. A month is not a fixed-length unit, so use an explicit
    # approximate calculation for this teaching example.
    months_elapsed_approx = np.round(
        elapsed / np.timedelta64(1, "D") / 30.4375,
        0,
    )

    print("\nApproximate months elapsed:")
    print(months_elapsed_approx.head())

    # ------------------------------------------------------------------
    # Working with time
    # ------------------------------------------------------------------
    messages["date"] = pd.to_datetime(messages["date"])
    print("\nMessages after datetime conversion:")
    messages.info()

    # Extract hour, minute, and second.
    messages["hour"] = messages["date"].dt.hour
    messages["min"] = messages["date"].dt.minute
    messages["sec"] = messages["date"].dt.second

    # Extract the time component.
    messages["time"] = messages["date"].dt.time

    time_features = ["date", "msg", "hour", "min", "sec", "time"]

    print("\nExtracted time features:")
    print(messages[time_features].head())

    # ------------------------------------------------------------------
    # Time difference
    # ------------------------------------------------------------------
    time_elapsed = today - messages["date"]

    print("\nTime elapsed from the reference datetime:")
    print(time_elapsed.head())

    print("\nTime elapsed in seconds:")
    print(time_elapsed.dt.total_seconds().head())

    print("\nTime elapsed in days:")
    print(time_elapsed.dt.days.head())


if __name__ == "__main__":
    main()
