import mlflow
from mlflow import MlflowClient


MODEL_NAME = (
    "timeseries-lstm-forecaster"
)

MODEL_URI = (
    "models:/m-c5641547ccae4f7baf0fc4c8e8f35765"
)


def main():

    print("Tracking URI:")
    print(mlflow.get_tracking_uri())
    print()

    print("Registering:")
    print(MODEL_URI)

    print()
    print("Registered model name:")
    print(MODEL_NAME)

    result = mlflow.register_model(
        model_uri=MODEL_URI,
        name=MODEL_NAME,
    )

    version = result.version

    print()
    print("Created model version:")
    print(version)

    client = MlflowClient()

    # --------------------------------------------------
    # Model-level metadata
    # --------------------------------------------------

    client.set_registered_model_tag(
        MODEL_NAME,
        "task",
        "time-series-forecasting",
    )

    client.set_registered_model_tag(
        MODEL_NAME,
        "framework",
        "pytorch",
    )

    # --------------------------------------------------
    # Version-level validation metadata
    # --------------------------------------------------

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "validation_status",
        "passed",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "serving_contract",
        "raw-input-raw-output",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "test_rmse",
        "0.857547",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "test_mae",
        "0.706870",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "contract_equivalence",
        "passed",
    )

    # --------------------------------------------------
    # Promote this validated version to candidate
    # --------------------------------------------------

    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        version,
    )

    print()
    print(
        "Alias candidate -> version",
        version,
    )

    # --------------------------------------------------
    # Verify alias resolution
    # --------------------------------------------------

    candidate = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "candidate",
        )
    )

    print()
    print("Resolved candidate:")
    print("  Name:", candidate.name)
    print("  Version:", candidate.version)
    print("  Aliases:", candidate.aliases)
    print("  Tags:", candidate.tags)

    print()
    print("Candidate URI:")
    print(
        f"models:/{MODEL_NAME}@candidate"
    )


if __name__ == "__main__":
    main()
