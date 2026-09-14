# KServe Observability Lab Design

## Goal

Add lightweight observability to the existing K3s + KServe serving lab.

The monitoring stack must demonstrate:

1. Predictor is at zero replicas while idle.
2. An inference request causes scale-from-zero.
3. Predictor CPU and memory activity can be observed.
4. Predictor returns to zero after inactivity.

This is a portfolio/interview lab, not a production monitoring platform.

## Existing Platform

The lab already contains:

- K3s v1.34.11+k3s1
- k3s-cp1: 192.168.56.30
- k3s-w1: 192.168.56.31
- Istio 1.27.1
- Knative Serving 1.21.1
- KServe v0.20.0
- timeseries-lstm InferenceService
- Knative scale-to-zero verified
- CPU PyTorch inference container
- Quality-rejected model deployed only as a technical serving demo

The laptop is resource constrained, so monitoring overhead must remain small.

## Architecture

Flow:

Request -> Istio -> KServe/Knative -> Predictor Pod

Kubernetes API -> kube-state-metrics -> Prometheus -> Grafana

Prometheus collects Kubernetes object and workload metrics.

Grafana queries Prometheus and visualizes the serving lifecycle.

## Components

### Prometheus

Deploy one Prometheus server.

Do not deploy:

- Alertmanager
- Pushgateway
- Prometheus Operator
- HA replicas
- production long-term storage

Prometheus storage may remain ephemeral.

### kube-state-metrics

Deploy kube-state-metrics to expose Kubernetes object state.

It will provide metrics used to observe deployments, pods, replicas, and readiness.

### Grafana

Deploy one Grafana instance.

Grafana will use Prometheus as its datasource.

Persistence is not required for this lab.

## Namespace

All observability components will run in:

monitoring

## Resource Constraints

Prometheus target sizing:

- CPU request: 100m
- Memory request: 256Mi
- Memory limit: 512Mi

Grafana target sizing:

- CPU request: 50m
- Memory request: 128Mi
- Memory limit: 256Mi

These limits are for the laptop lab only.

## Dashboard Requirements

The minimum dashboard must show:

### Predictor Replica Lifecycle

Demonstrate:

0 replicas -> 1 replica -> 0 replicas

for the Knative predictor workload.

### Predictor CPU

Show CPU activity during model startup and inference where metrics are available.

### Predictor Memory

Show memory consumption while the predictor pod is running.

### Serving State

Show whether the predictor workload exists and whether it is ready.

Request-rate metrics may be added later if already exposed cleanly by Istio,
Knative, or KServe.

The inference application will not be modified solely for the first
observability milestone.

## Repository Structure

Store the monitoring configuration under:

infra/k3s/monitoring/

Expected files:

- prometheus-values.yaml
- grafana-values.yaml
- README.md

Helm chart versions will be pinned during implementation.

## Access

Grafana will initially be accessed using Kubernetes port forwarding.

No production ingress, external DNS, TLS, or LoadBalancer configuration is
required.

Prometheus may also be port-forwarded for troubleshooting.

## Validation Flow

The final demonstration will be:

1. Confirm ml-serving has zero predictor pods.
2. Open Grafana.
3. Send an inference request through Istio/KServe.
4. Observe predictor scale from zero to one.
5. Observe predictor CPU and memory activity.
6. Confirm successful inference.
7. Wait for the Knative idle timeout.
8. Observe predictor return to zero replicas.

## Non-Goals

The following are excluded:

- Prometheus HA
- Thanos
- Loki
- distributed tracing
- OpenTelemetry architecture
- Alertmanager
- paging and on-call alerts
- long-term metrics storage
- production Grafana authentication
- external DNS
- TLS
- production retention policies

## Success Criteria

The phase is complete when:

- Prometheus is healthy.
- kube-state-metrics is healthy.
- Grafana is healthy.
- Grafana can query Prometheus.
- Predictor replica transition 0 -> 1 -> 0 is observable.
- Predictor CPU or memory activity is observable.
- Successful inference can be correlated with the scaling event.
- Monitoring configuration is committed to Git.
- README documents the demo procedure.
