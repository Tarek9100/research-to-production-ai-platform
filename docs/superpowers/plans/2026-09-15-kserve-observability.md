# KServe Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add lightweight Prometheus and Grafana observability to the existing K3s/KServe lab and demonstrate the predictor lifecycle from 0 -> 1 -> 0 replicas with CPU and memory visibility.

**Architecture:** A single Prometheus server scrapes Kubernetes state and kubelet/cAdvisor metrics. kube-state-metrics exposes Kubernetes object state. Grafana queries Prometheus and is accessed using port forwarding.

**Tech Stack:** K3s v1.34.11+k3s1, KServe v0.20.0, Knative Serving 1.21.1, Prometheus, kube-state-metrics, Grafana, Helm.

**Spec:** docs/superpowers/specs/2026-09-15-kserve-observability-design.md

## Global Constraints

- Run observability components in namespace monitoring.
- One Prometheus server.
- One Grafana instance.
- Disable Alertmanager and Pushgateway.
- Do not deploy Prometheus Operator.
- No persistent storage required.
- Prometheus request: 100m CPU / 256Mi memory.
- Prometheus memory limit: 512Mi.
- Grafana request: 50m CPU / 128Mi memory.
- Grafana memory limit: 256Mi.
- Access Grafana through port-forwarding.
- Do not modify the inference application for the first milestone.
- Monitor the existing timeseries-lstm InferenceService.
- The model remains quality-rejected and is used only as a technical serving demo.

---

## Task 1: Deploy Lightweight Monitoring Stack

Files:

- infra/k3s/monitoring/prometheus-values.yaml
- infra/k3s/monitoring/grafana-values.yaml
- infra/k3s/scripts/install-monitoring.sh

Steps:

1. Configure Prometheus with Alertmanager, Pushgateway, and node-exporter disabled.
2. Enable kube-state-metrics.
3. Configure one ephemeral Prometheus server with the required resource limits.
4. Configure Grafana with persistence disabled and Prometheus as its datasource.
5. Pin Helm chart versions in install-monitoring.sh.
6. Install both charts into namespace monitoring.
7. Verify Prometheus, kube-state-metrics, and Grafana pods are Running.
8. Commit the monitoring configuration.

Verification commands:

    sudo kubectl get pods -n monitoring -o wide
    sudo kubectl get svc -n monitoring

Commit:

    git add infra/k3s/monitoring infra/k3s/scripts/install-monitoring.sh
    git commit -m "Add lightweight KServe monitoring stack"

---

## Task 2: Validate KServe Metrics

Create:

- infra/k3s/monitoring/queries.md

Validate these metric categories:

Replica state:

    kube_deployment_status_replicas

Pod readiness:

    kube_pod_status_ready

Predictor CPU:

    rate(container_cpu_usage_seconds_total{namespace="ml-serving",container="kserve-container"}[1m])

Predictor memory:

    container_memory_working_set_bytes{namespace="ml-serving",container="kserve-container"}

Lifecycle to prove:

    idle -> 0 predictor pods
    request -> predictor pod appears
    serving -> CPU and memory metrics appear
    idle -> predictor pod disappears

Do not hard-code generated pod names.

Commit:

    git add infra/k3s/monitoring/queries.md
    git commit -m "Document KServe observability queries"

---

## Task 3: Grafana Demo and Documentation

Create:

- infra/k3s/monitoring/README.md

Minimum Grafana dashboard:

1. Predictor replica lifecycle.
2. Predictor CPU usage.
3. Predictor memory usage.
4. Predictor readiness/state.

Final demonstration:

    0 predictor pods
        ->
    send inference request
        ->
    prediction succeeds
        ->
    1 predictor pod
        ->
    CPU and memory metrics visible
        ->
    idle timeout
        ->
    0 predictor pods

README must document:

- monitoring installation
- Grafana access
- inference request procedure
- verified PromQL queries
- scale-to-zero behavior
- laptop-lab limitations
- non-production nature of the monitoring architecture

Final verification:

    sudo kubectl get nodes
    sudo kubectl get pods -n monitoring
    sudo kubectl get inferenceservice -n ml-serving
    sudo kubectl get pods -n ml-serving

Completion criteria:

- Prometheus healthy.
- kube-state-metrics healthy.
- Grafana healthy.
- Grafana can query Prometheus.
- Replica transition 0 -> 1 -> 0 observable.
- Predictor CPU or memory observable.
- Inference correlated with scaling.
- Monitoring configuration committed.
- README contains repeatable demo procedure.
