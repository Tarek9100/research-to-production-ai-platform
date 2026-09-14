# KServe Observability Queries

These PromQL queries were validated against the K3s + KServe lab.

## Predictor Replica Count

PromQL:

    kube_deployment_status_replicas{
      namespace="ml-serving",
      deployment=~"timeseries-lstm-predictor-.*-deployment"
    }

Expected lifecycle:

    0 -> 1 -> 0

When Knative scales the revision to zero, the generated Deployment may remain
while its replica count becomes zero.

## Available Predictor Replicas

PromQL:

    kube_deployment_status_replicas_available{
      namespace="ml-serving",
      deployment=~"timeseries-lstm-predictor-.*-deployment"
    }

Use this as the primary serving-readiness metric.

## Predictor CPU Usage

PromQL:

    sum(
      rate(
        container_cpu_usage_seconds_total{
          namespace="ml-serving",
          container="kserve-container"
        }[5m]
      )
    )

A five-minute rate window is used because the predictor is short-lived and a
one-minute window may contain too few Prometheus scrape samples immediately
after a cold start.

## Predictor Memory

PromQL:

    sum(
      container_memory_working_set_bytes{
        namespace="ml-serving",
        container="kserve-container"
      }
    )

During the validated test run, predictor working-set memory was approximately
173 MiB.

## Predictor Pod State

PromQL:

    count(
      kube_pod_info{
        namespace="ml-serving",
        pod=~"timeseries-lstm-predictor-.*"
      }
    )

This provides another view of the Knative scale-to-zero lifecycle.

## Notes

The deployed model is marked quality_rejected_demo_only.

This deployment demonstrates serving infrastructure and observability, not
model promotion or production approval.

CPU and memory series disappear after the predictor scales to zero because
the corresponding container no longer exists.
