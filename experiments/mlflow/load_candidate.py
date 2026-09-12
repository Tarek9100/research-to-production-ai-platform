import mlflow
import mlflow.pytorch
import torch


MODEL_URI = (
    "models:/timeseries-lstm-forecaster@candidate"
)


def main():

    print("Loading:")
    print(MODEL_URI)
    print()

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    x = torch.tensor(
        [
            [
                [18.0],
                [18.2],
                [18.1],
                [17.9],
                [18.3],
                [18.5],
                [18.4],
                [18.6],
                [18.7],
                [18.5],
                [18.8],
                [19.0],
                [18.9],
                [19.1],
                [19.2],
                [19.0],
                [19.3],
                [19.4],
                [19.2],
                [19.5],
                [19.6],
                [19.4],
                [19.7],
                [19.8],
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction = model(x)

    print(
        "Prediction:",
        prediction,
    )


if __name__ == "__main__":
    main()
