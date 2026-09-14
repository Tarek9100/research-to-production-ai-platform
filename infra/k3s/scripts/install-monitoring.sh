#!/usr/bin/env bash

set -euo pipefail

PROMETHEUS_CHART_VERSION="29.27.0"
GRAFANA_CHART_VERSION="10.4.1"

KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

export KUBECONFIG

echo "===== HELM ====="
helm version --short

echo
echo "===== REPOSITORIES ====="

helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts \
  --force-update

helm repo add grafana \
  https://grafana.github.io/helm-charts \
  --force-update

helm repo update

echo
echo "===== NAMESPACE ====="

kubectl create namespace monitoring \
  --dry-run=client \
  -o yaml | kubectl apply -f -

echo
echo "===== PROMETHEUS ====="

helm upgrade --install prometheus \
  prometheus-community/prometheus \
  --version "${PROMETHEUS_CHART_VERSION}" \
  --namespace monitoring \
  --values /tmp/prometheus-values.yaml \
  --wait \
  --timeout 10m

echo
echo "===== GRAFANA ====="

helm upgrade --install grafana \
  grafana/grafana \
  --version "${GRAFANA_CHART_VERSION}" \
  --namespace monitoring \
  --values /tmp/grafana-values.yaml \
  --wait \
  --timeout 10m

echo
echo "===== DEPLOYMENTS ====="

kubectl get deployments -n monitoring

echo
echo "===== PODS ====="

kubectl get pods -n monitoring -o wide

echo
echo "===== SERVICES ====="

kubectl get svc -n monitoring
