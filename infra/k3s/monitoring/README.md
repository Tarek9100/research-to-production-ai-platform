# KServe Observability Lab

This directory contains the lightweight monitoring configuration for the
K3s + KServe serving lab.

## Stack

- Prometheus
- kube-state-metrics
- Grafana
- KServe
- Knative Serving
- Istio
- K3s

The monitoring stack is intentionally lightweight and designed for a
resource-constrained laptop lab.

It is not a production HA monitoring architecture.

## Architecture

Inference flow:

    Inference Client
          |
          v
    Istio Ingress
          |
          v
    KServe / Knative
          |
          v
    PyTorch Predictor Pod

Monitoring flow:

    Kubernetes API ------\
    kubelet / cAdvisor ---+--> Prometheus --> Grafana
    kube-state-metrics --/

## Monitoring Installation

The monitoring stack is installed using:

    infra/k3s/scripts/install-monitoring.sh

Prometheus configuration:

    infra/k3s/monitoring/prometheus-values.yaml

Grafana configuration:

    infra/k3s/monitoring/grafana-values.yaml

The deployment intentionally disables:

- Alertmanager
- Pushgateway
- node-exporter
- Prometheus Operator
- persistent storage

## Grafana Dashboard

The KServe-specific Grafana dashboard is stored at:

    infra/k3s/monitoring/kserve-serving-overview.json

It visualizes:

- predictor replica count
- available predictor replicas
- predictor pod count
- model-container restarts
- Knative scale-to-zero lifecycle
- predictor CPU usage
- predictor memory working set

## Access Grafana

On k3s-cp1:

    sudo kubectl port-forward \
      --address 192.168.56.30 \
      -n monitoring \
      svc/grafana \
      3000:80

Then open:

    http://192.168.56.30:3000

## Inference Demo

The deployed model is a technical serving demo only.

Its governance status is:

    quality_rejected_demo_only

The model was rejected by the ML quality gate and is not a
production-approved or promoted model.

## Initial State

Knative can scale the predictor to zero:

    sudo kubectl get pods -n ml-serving

Expected while idle:

    No resources found in ml-serving namespace.

The KServe InferenceService itself remains Ready.

Check it with:

    sudo kubectl get inferenceservice -n ml-serving

## Generate Load

Send inference traffic through Istio:

    seq 1 200 | xargs -n1 -P10 -I{} \
      curl -sS \
        -o /dev/null \
        -w "%{http_code}\n" \
        -X POST \
        -H "Host: timeseries-lstm.ml-serving.example.com" \
        -H "Content-Type: application/json" \
        --data @/tmp/request.json \
        http://10.43.253.222/v1/models/timeseries-lstm:predict \
      | sort | uniq -c

Successful requests should return HTTP 200.

Example:

    200 200

## Serving Lifecycle

Expected behavior:

    0 predictor replicas
            |
            | inference request
            v
    Knative cold start
            |
            v
    1+ predictor replicas
            |
            | CPU and memory become visible
            v
    successful inference
            |
            | idle timeout
            v
    0 predictor replicas

maxReplicas is a ceiling and does not guarantee that the predictor
will scale to that number.

Knative scales according to observed traffic and concurrency.

## Verified PromQL

See:

    infra/k3s/monitoring/queries.md

Core queries are listed below.

### Predictor Replicas

    kube_deployment_status_replicas{
      namespace="ml-serving",
      deployment=~"timeseries-lstm-predictor-.*-deployment"
    }

### Available Predictor Replicas

    kube_deployment_status_replicas_available{
      namespace="ml-serving",
      deployment=~"timeseries-lstm-predictor-.*-deployment"
    }

### Predictor CPU Usage

    sum(
      rate(
        container_cpu_usage_seconds_total{
          namespace="ml-serving",
          container="kserve-container"
        }[5m]
      )
    )

### Predictor Memory Usage

    sum(
      container_memory_working_set_bytes{
        namespace="ml-serving",
        container="kserve-container"
      }
    )

During validation, the predictor used approximately 173 MiB of
working-set memory.

## Final Validation

Check the cluster:

    sudo kubectl get nodes

Check monitoring components:

    sudo kubectl get pods -n monitoring

Check KServe:

    sudo kubectl get inferenceservice -n ml-serving

Check predictor pods:

    sudo kubectl get pods -n ml-serving

Expected final state:

    k3s-cp1   Ready
    k3s-w1    Ready

    grafana                         Running
    prometheus-server               Running
    prometheus-kube-state-metrics   Running

    timeseries-lstm   Ready

    No resources found in ml-serving namespace.

## Lab Limitations

This environment intentionally does not implement:

- Prometheus HA
- long-term metrics storage
- Alertmanager
- Loki
- distributed tracing
- production Grafana authentication
- external DNS
- TLS
- production ingress
- GPU inference

The purpose of this lab is to demonstrate the end-to-end
infrastructure lifecycle from model serving through serverless
autoscaling and observability.
