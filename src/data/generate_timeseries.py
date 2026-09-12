from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
NUM_POINTS = 20_000

OUTPUT = Path("data/raw/timeseries.csv")


def generate_series(num_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    t = np.arange(num_points, dtype=np.float32)

    trend = 0.0005 * t

    daily = 2.0 * np.sin(
        2 * np.pi * t / 24
    )

    weekly = 0.8 * np.sin(
        2 * np.pi * t / (24 * 7)
    )

    noise = rng.normal(
        loc=0.0,
        scale=0.35,
        size=num_points,
    )

    values = (
        10
        + trend
        + daily
        + weekly
        + noise
    )

    return pd.DataFrame(
        {
            "time_idx": t.astype(int),
            "value": values.astype(np.float32),
        }
    )


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_series(
        num_points=NUM_POINTS,
        seed=SEED,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"Generated rows: {len(df)}")
    print(f"Saved to: {OUTPUT}")
    print()
    print(df.head())
    print()
    print(df.describe())


if __name__ == "__main__":
    main()
