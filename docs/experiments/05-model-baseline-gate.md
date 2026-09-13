# Model Baseline Quality Gate

## Objective

Validate whether trained LSTM forecasting models outperform simple forecasting baselines before promotion.

## V1

Sequence length: 24

Candidate RMSE:

- 0.857547

Baselines:

- Persistence RMSE: 0.616636
- Daily seasonal-naive RMSE: 0.698644

Result:

- Candidate vs persistence: -39.07%
- Candidate vs daily seasonal naive: -22.74%

Decision:

V1 rejected.

The model version was retained in MLflow for audit history, but the `candidate` and `champion` aliases were removed.

## V2

Sequence length: 168

The only intended ML change was increasing the context window from 24 to 168 so the model could observe a full weekly cycle.

MLflow run:

- `03c1b77895b94d42acd362452508d0e6`

Test metrics:

- MSE: 0.716461
- RMSE: 0.846440
- MAE: 0.687974

Evaluation samples:

- 2832

Baselines:

| Model | RMSE | MAE |
|---|---:|---:|
| Persistence t-1 | 0.615620 | 0.491653 |
| Daily naive t-24 | 0.697272 | 0.565751 |
| Weekly naive t-168 | 0.512335 | 0.411950 |
| LSTM v2 | 0.846440 | 0.687974 |

Relative RMSE performance:

- vs persistence: -37.49%
- vs daily naive: -21.39%
- vs weekly naive: -65.21%

## Decision

V2 rejected.

It was not registered or promoted because it failed the baseline quality gate.

No further model tuning was performed because the objective of this project is MLOps and AI infrastructure engineering rather than model research.

## Engineering Lesson

Absolute model metrics are insufficient for promotion decisions.

A model can have apparently reasonable validation and test metrics while still being inferior to trivial domain baselines.

The promotion workflow therefore requires comparison against appropriate baselines before assigning deployment aliases.

This prevented a technically valid but low-value model from progressing toward production.

## Platform Outcome

The platform successfully demonstrated:

- reproducible training
- Git lineage
- configuration lineage
- DVC dataset versioning
- MinIO-backed dataset storage
- SHA-256 integrity verification
- GPU container execution
- MLflow experiment tracking
- model packaging
- model registry governance
- candidate/champion aliases
- failed quality-gate handling
- preservation of rejected model history
