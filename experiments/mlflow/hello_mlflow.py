import mlflow


mlflow.set_experiment(
    "research-to-production-ai-platform"
)

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "lstm"
    )

    mlflow.log_param(
        "batch_size",
        1024
    )

    mlflow.log_param(
        "device",
        "cuda"
    )

    mlflow.log_param(
        "precision",
        "fp32"
    )

    mlflow.log_metric(
        "throughput_samples_sec",
        49514.5
    )

    mlflow.log_metric(
        "peak_gpu_memory_mib",
        256.9
    )

    mlflow.log_metric(
        "final_train_loss",
        0.033208
    )

    print(
        "Logged first MLflow run."
    )
