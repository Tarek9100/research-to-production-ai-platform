from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mean: float
    std: float


def load_and_split(
    path: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DataSplits:

    df = pd.read_csv(path)

    values = (
        df["value"]
        .to_numpy(dtype=np.float32, copy=True)
    )

    n = len(values)

    train_end = int(
        n * train_fraction
    )

    validation_end = int(
        n
        * (
            train_fraction
            + validation_fraction
        )
    )

    train = values[:train_end]

    validation = values[
        train_end:validation_end
    ]

    test = values[
        validation_end:
    ]

    mean = float(train.mean())
    std = float(train.std())

    train = (
        train - mean
    ) / std

    validation = (
        validation - mean
    ) / std

    test = (
        test - mean
    ) / std

    return DataSplits(
        train=train,
        validation=validation,
        test=test,
        mean=mean,
        std=std,
    )
