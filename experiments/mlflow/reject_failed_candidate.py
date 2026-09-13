import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "timeseries-lstm-forecaster"

CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"

CANDIDATE_RMSE = 0.857547
PERSISTENCE_RMSE = 0.616636
SEASONAL_NAIVE_RMSE = 0.698644

PERSISTENCE_IMPROVEMENT_PCT = -39.07
SEASONAL_IMPROVEMENT_PCT = -22.74


def delete_alias_if_points_to(
    client,
    alias,
    rejected_version,
):
    try:
        resolved = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                alias,
            )
        )
    except MlflowException:
        print(
            f"Alias {alias!r}: "
            "not currently assigned"
        )
        return

    if str(resolved.version) != str(
        rejected_version
    ):
        print(
            f"Alias {alias!r}: "
            f"points to v{resolved.version}; "
            "leaving unchanged"
        )
        return

    client.delete_registered_model_alias(
        MODEL_NAME,
        alias,
    )

    print(
        f"Deleted alias {alias!r} "
        f"from v{rejected_version}"
    )


def main():
    print("Tracking URI:")
    print(mlflow.get_tracking_uri())
    print()

    client = MlflowClient()

    candidate = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS,
        )
    )

    version = candidate.version

    print(
        "Rejecting model version:",
        version,
    )

    print(
        "Current aliases:",
        candidate.aliases,
    )

    tags = {
        "validation_status":
            "failed_baseline_gate",

        "baseline_gate":
            "failed",

        "promotion_status":
            "rejected",

        "quality_gate":
            "failed",

        "rejection_reason":
            "underperforms_trivial_baselines",

        "candidate_rmse":
            str(CANDIDATE_RMSE),

        "persistence_rmse":
            str(PERSISTENCE_RMSE),

        "daily_seasonal_naive_rmse":
            str(SEASONAL_NAIVE_RMSE),

        "improvement_vs_persistence_pct":
            str(
                PERSISTENCE_IMPROVEMENT_PCT
            ),

        "improvement_vs_daily_naive_pct":
            str(
                SEASONAL_IMPROVEMENT_PCT
            ),

        "baseline_evaluation_samples":
            "2976",

        "baseline_evaluation_status":
            "completed",
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            MODEL_NAME,
            version,
            key,
            value,
        )

    # Remove deployment/promotion aliases only if
    # they still point to this rejected version.
    delete_alias_if_points_to(
        client,
        CHAMPION_ALIAS,
        version,
    )

    delete_alias_if_points_to(
        client,
        CANDIDATE_ALIAS,
        version,
    )

    updated = client.get_model_version(
        MODEL_NAME,
        version,
    )

    registered = (
        client.get_registered_model(
            MODEL_NAME
        )
    )

    print()
    print("===== UPDATED VERSION =====")
    print("Version:", updated.version)
    print("Aliases:", updated.aliases)

    print()
    print("Tags:")

    for key in sorted(updated.tags):
        print(
            f"  {key}="
            f"{updated.tags[key]}"
        )

    print()
    print("===== REGISTERED MODEL =====")
    print(
        "Aliases:",
        registered.aliases,
    )

    print()
    print(
        "REJECTION COMPLETE: "
        "model retained for audit history "
        "but is no longer candidate/champion."
    )


if __name__ == "__main__":
    main()
