#!/usr/bin/env bash

set -euo pipefail

K3S_VERSION="v1.34.11+k3s1"
K3S_URL="https://192.168.56.30:6443"
NODE_IP="192.168.56.31"
FLANNEL_IFACE="enp0s8"

TOKEN="${1:-}"

if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: K3s server token must be passed as argument 1"
  exit 1
fi

echo "Installing K3s agent ${K3S_VERSION}"

curl -sfL https://get.k3s.io | \
  INSTALL_K3S_VERSION="${K3S_VERSION}" \
  K3S_URL="${K3S_URL}" \
  K3S_TOKEN="${TOKEN}" \
  sh -s - agent \
    --node-ip="${NODE_IP}" \
    --flannel-iface="${FLANNEL_IFACE}"

echo
echo "===== K3S AGENT ====="
systemctl is-active k3s-agent
