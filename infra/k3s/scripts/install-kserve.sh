#!/usr/bin/env bash

set -euo pipefail

KSERVE_VERSION="v0.20.0"

curl -fsSL \
  "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-knative-mode-full-install-with-manifests.sh" \
  -o /tmp/install-kserve.sh

chmod +x /tmp/install-kserve.sh

env \
  KUBECONFIG=/etc/rancher/k3s/k3s.yaml \
  bash /tmp/install-kserve.sh
