"""
eda.py
Basic exploratory analysis used before predictive modeling.
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "logistics_data.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nAverage delay by route:")
    print(df.groupby("Route_Type")["Delay_hr"].mean().sort_values(ascending=False))

    print("\nAverage delay by transport mode:")
    print(df.groupby("Transport_Mode")["Delay_hr"].mean().sort_values(ascending=False))

    print("\nNumeric correlations:")
    print(df.select_dtypes("number").corr()["Delay_hr"].sort_values(ascending=False))


if __name__ == "__main__":
    main()
