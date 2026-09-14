#!/usr/bin/env bash

set -euo pipefail

K3S_VERSION="v1.34.11+k3s1"
NODE_IP="192.168.56.30"
FLANNEL_IFACE="enp0s8"

echo "Installing K3s server ${K3S_VERSION}"

curl -sfL https://get.k3s.io | \
  INSTALL_K3S_VERSION="${K3S_VERSION}" \
  sh -s - server \
    --node-ip="${NODE_IP}" \
    --advertise-address="${NODE_IP}" \
    --tls-san="${NODE_IP}" \
    --flannel-iface="${FLANNEL_IFACE}" \
    --disable=traefik \
    --disable=servicelb \
    --write-kubeconfig-mode=644

echo
echo "===== K3S VERSION ====="
k3s --version

echo
echo "===== SERVICE ====="
systemctl is-active k3s

echo
echo "===== NODE ====="
k3s kubectl get nodes -o wide
