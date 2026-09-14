# K3s + KServe Serving Implementation Plan

Goal: Build a reproducible two-node K3s cluster and deploy a KServe predictive inference service with Knative scale-to-zero and scale-from-zero.

Architecture:
- k3s-cp1: 192.168.56.30, 2 vCPU, 3 GB RAM
- k3s-w1: 192.168.56.31, 2 vCPU, 3 GB RAM
- K3s server on cp1
- K3s worker on w1
- cert-manager
- Istio
- Knative Serving
- KServe
- CPU-only inference

Spec:
docs/superpowers/specs/2026-09-14-k3s-kserve-serving-design.md

## Constraints

- Slurm VMs remain halted while K3s is running.
- Vagrant must work through WSL -> Windows -> VirtualBox.
- Disable the /vagrant synced folder.
- Use pinned component versions.
- KServe must use Knative/serverless mode.
- The existing LSTM remains quality-rejected and is used only as a serving-infrastructure demo.
- Scale-to-zero and scale-from-zero must be demonstrated with real HTTP traffic.

## Task 1 - Vagrant topology

Create:
- infra/k3s/Vagrantfile
- infra/k3s/README.md

Requirements:
- k3s-cp1 at 192.168.56.30
- k3s-w1 at 192.168.56.31
- 2 CPUs and 3 GB RAM each
- ubuntu/jammy64
- disable /vagrant
- disable problematic VirtualBox UART
- configure WSL Windows-host SSH routing
- configure /etc/hosts on both VMs

Verification commands:

    vagrant status
    vagrant ssh k3s-cp1 -c 'hostname; ip -br addr; ping -c 2 k3s-w1'
    vagrant ssh k3s-w1 -c 'hostname; ip -br addr; ping -c 2 k3s-cp1'

Expected result:

    k3s-cp1 running
    k3s-w1  running

## Task 2 - K3s installation

Create:
- infra/k3s/scripts/install-k3s-server.sh
- infra/k3s/scripts/install-k3s-agent.sh

Requirements:
- pin K3s version
- install server on k3s-cp1
- bind server to 192.168.56.30
- retrieve node token
- join k3s-w1 to https://192.168.56.30:6443

Verification:

    sudo k3s kubectl get nodes -o wide
    sudo k3s kubectl get pods -A

Expected:

    k3s-cp1 Ready
    k3s-w1  Ready

## Task 3 - kubectl from WSL

Requirements:
- copy kubeconfig from k3s-cp1
- replace localhost API endpoint with reachable control-plane address
- verify kubectl from WSL

Verification:

    kubectl get nodes
    kubectl cluster-info

## Task 4 - KServe dependency stack

Create:
- infra/k3s/scripts/install-kserve.sh

Install in this order:
1. cert-manager
2. Istio
3. Knative Serving
4. Knative networking integration
5. KServe

All versions must be pinned.

Verification:

    kubectl get pods -A
    kubectl get crd | grep -E 'inferenceservice|serving.knative'
    kubectl get pods -n kserve
    kubectl get pods -n knative-serving

## Task 5 - Serving container

Build a CPU model-serving image using the existing packaged LSTM artifact.

Requirements:
- health endpoint
- prediction endpoint
- input validation
- structured JSON response
- reuse the packaged model preprocessing
- do not claim that the model passed the quality gate

Local verification:

    curl http://127.0.0.1:<port>/health
    curl -X POST http://127.0.0.1:<port>/predict ...

## Task 6 - Make image available to K3s

Preferred order:
1. import OCI image into K3s/containerd
2. local registry if needed
3. external registry only if intentionally chosen

Verification:

    sudo k3s ctr images list
    kubectl get pods

## Task 7 - KServe InferenceService

Create:
- infra/k3s/manifests/inference/timeseries-lstm.yaml

Requirements:
- separate serving namespace
- custom predictor container
- CPU/memory requests
- Knative autoscaling configuration
- scale-to-zero enabled

Verification:

    kubectl get inferenceservice -A
    kubectl get ksvc -A
    kubectl get revision -A
    kubectl get pods -A

## Task 8 - Scale-to-zero demonstration

Demonstrate this lifecycle:

    0 predictor pods
           |
           v
      HTTP request
           |
           v
    1 predictor pod
           |
           v
       prediction
           |
           v
          idle
           |
           v
    0 predictor pods

Evidence required:
- zero replicas before request
- predictor created after request
- successful prediction
- predictor removed after idle timeout

## Task 9 - Documentation

Expand:
- infra/k3s/README.md
- root README.md

Document:
- architecture
- K3s bootstrap
- KServe
- Knative
- scale-to-zero
- inference
- CPU-only limitation
- model governance disclaimer
- troubleshooting

Recommended screenshots:
- k3s-01-nodes.png
- k3s-02-inferenceservice.png
- k3s-03-scale-zero.png
- k3s-04-scale-up.png
- k3s-05-inference.png

## Task 10 - Final verification

Verify:

    kubectl get nodes
    kubectl get inferenceservice -A
    kubectl get ksvc -A
    kubectl get pods -A

Final repository message must remain:

    model quality gate: FAILED
    serving infrastructure demonstration: SUCCESS
