FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=timeseries-lstm
ENV CHECKPOINT_PATH=/app/artifacts/checkpoints/lstm_weekly_context.pt

COPY serving/timeseries/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0 \
    && pip install --no-cache-dir \
      -r /tmp/requirements.txt

COPY src /app/src

COPY artifacts/checkpoints/lstm_weekly_context.pt \
     /app/artifacts/checkpoints/lstm_weekly_context.pt

COPY serving/timeseries/app.py /app/app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
