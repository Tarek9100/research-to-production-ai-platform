import mlflow
import mlflow.pytorch
import torch


MODEL_URI = (
    "models:/m-8d3a73abc5324745aaa992fbdf641105"
)


def main():

    print("Tracking URI:")
    print(mlflow.get_tracking_uri())
    print()

    print("Loading model:")
    print(MODEL_URI)
    print()

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    print("Loaded object type:")
    print(type(model))
    print()

    print("Loaded object:")
    print(model)
    print()

    x = torch.zeros(
        (
            2,
            24,
            1,
        ),
        dtype=torch.float32,
    )

    print("Input shape:")
    print(x.shape)

    with torch.no_grad():
        prediction = model(x)

    print()
    print("Output shape:")
    print(prediction.shape)

    print()
    print("Prediction:")
    print(prediction)


if __name__ == "__main__":
    main()
