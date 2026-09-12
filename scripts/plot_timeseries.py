from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("data/raw/timeseries.csv")
OUTPUT = Path("artifacts/timeseries_preview.png")


def main():
    df = pd.read_csv(INPUT)

    sample = df.iloc[:500]

    plt.figure(figsize=(12, 5))

    plt.plot(
        sample["time_idx"],
        sample["value"],
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Synthetic Time-Series Preview")

    plt.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT,
        dpi=150,
    )

    print(f"Saved plot to {OUTPUT}")


if __name__ == "__main__":
    main()
