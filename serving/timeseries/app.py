import os
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.lstm import LSTMForecaster
from src.models.serving import TimeSeriesServingModel


MODEL_NAME = os.getenv("MODEL_NAME", "timeseries-lstm")
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "/app/artifacts/checkpoints/lstm_weekly_context.pt",
)


class PredictRequest(BaseModel):
    instances: List[List[float]]


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

sequence_length = int(checkpoint["sequence_length"])

base_model = LSTMForecaster(
    input_size=1,
    hidden_size=int(checkpoint["hidden_size"]),
    num_layers=int(checkpoint["num_layers"]),
    dropout=0.1,
)

base_model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = TimeSeriesServingModel(
    model=base_model,
    mean=float(checkpoint["mean"]),
    std=float(checkpoint["std"]),
)

model.eval()

app = FastAPI(
    title="Timeseries LSTM KServe Demo",
)


@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "sequence_length": sequence_length,
        "governance_status": "quality_rejected_demo_only",
    }


@app.post("/v1/models/{model_name}:predict")
def predict(
    model_name: str,
    request: PredictRequest,
):
    if model_name != MODEL_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}",
        )

    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="instances must not be empty",
        )

    for i, instance in enumerate(request.instances):
        if len(instance) != sequence_length:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"instances[{i}] must contain "
                    f"{sequence_length} values; "
                    f"received {len(instance)}"
                ),
            )

    x = torch.tensor(
        request.instances,
        dtype=torch.float32,
    ).unsqueeze(-1)

    with torch.inference_mode():
        predictions = model(x)

    return {
        "predictions": predictions.tolist(),
        "model": MODEL_NAME,
        "sequence_length": sequence_length,
        "governance_status": "quality_rejected_demo_only",
    }
