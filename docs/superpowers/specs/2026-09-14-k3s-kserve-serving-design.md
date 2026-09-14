# K3s + KServe Serving Lab Design

## Goal

Build a reproducible laptop-scale Kubernetes serving environment that demonstrates:

- Kubernetes control-plane and worker separation
- KServe `InferenceService`
- Knative request-driven autoscaling
- scale-to-zero
- scale-from-zero
- HTTP inference
- model-serving troubleshooting

This environment is a portfolio and interview-preparation lab, not a production HA Kubernetes platform.

## Architecture

    k3s-cp1
    192.168.56.30
    2 vCPU / 3 GB RAM
    K3s server
          |
          |
          v
    k3s-w1
    192.168.56.31
    2 vCPU / 3 GB RAM
    K3s worker

The Slurm VMs remain halted while this environment is running to stay within laptop memory constraints.

## Repository Layout

    infra/k3s/
    ├── Vagrantfile
    ├── README.md
    ├── scripts/
    │   ├── install-k3s-server.sh
    │   ├── install-k3s-agent.sh
    │   └── install-kserve.sh
    └── manifests/
        └── inference/

## Platform Stack

The target serving stack is:

    K3s
     |
     +-- cert-manager
     |
     +-- Istio
     |
     +-- Knative Serving
     |
     +-- KServe
           |
           +-- InferenceService

KServe will use Knative/serverless deployment because this project specifically needs to demonstrate HTTP-driven scale-to-zero and scale-from-zero.

## Kubernetes Nodes

### Control Plane

`k3s-cp1` runs the K3s server and Kubernetes control-plane components.

It is also the administrative node used for cluster bootstrap and inspection.

### Worker

`k3s-w1` runs application workloads, including model-serving workloads.

Keeping the worker separate from the control plane allows actual Kubernetes scheduling behavior to be demonstrated.

## Networking

VirtualBox host-only networking:

    k3s-cp1    192.168.56.30
    k3s-w1     192.168.56.31

The design must remain compatible with the WSL -> Windows -> VirtualBox arrangement already encountered in the Slurm lab.

The Vagrant configuration therefore needs to:

- disable the `/vagrant` synced folder
- expose SSH using VirtualBox NAT forwarding
- use the dynamically discovered Windows host IP from WSL
- disable the problematic VirtualBox UART configuration

## K3s Installation

K3s will use a pinned version rather than an uncontrolled latest-version installation.

The server will:

- initialize the Kubernetes cluster
- expose the Kubernetes API
- create the node join token

The worker will join using:

    https://192.168.56.30:6443

and the K3s server-generated token.

## Model Governance Context

The project already contains an LSTM forecasting model packaged and tracked through MLflow.

That model failed the project's baseline quality gate.

Therefore:

- it must not be described as an approved production model
- it may still be used to demonstrate serving infrastructure
- documentation must explicitly identify it as technically deployable but quality-rejected

This preserves the distinction between:

    technical deployability

and:

    model-quality acceptance

## Serving Architecture

The target serving path is:

    packaged model
          |
          v
    serving container
          |
          v
    KServe InferenceService
          |
          v
    Knative Revision
          |
          v
    predictor pod

When idle:

    replicas = 0

After an HTTP request:

    request
       |
       v
    Knative
       |
       v
    0 -> 1 predictor pod
       |
       v
    prediction response

After the idle period:

    1 -> 0 predictor pods

## Serving Contract

The service should expose a simple HTTP inference contract.

It should:

- accept the time-series input expected by the model
- validate incoming input
- execute inference
- return a numeric prediction in structured JSON

Normalization and preprocessing must remain consistent with the packaged serving model rather than being reimplemented differently in Kubernetes.

## Model Packaging Decision

KServe will not receive a raw arbitrary `.pkl` file and be expected to understand custom Python behavior.

The deployment will use an explicit serving container.

The serving container is responsible for:

- loading the packaged model
- defining the HTTP inference contract
- validating payloads
- executing prediction
- returning structured output

KServe is responsible for:

- application lifecycle
- networking
- revision management
- routing
- autoscaling
- scale-to-zero
- scale-from-zero

## Resource Constraints

This Kubernetes environment is a CPU serving lab.

The VirtualBox worker does not expose the physical GTX 1650 GPU.

Real GPU experiments remain in the WSL/Docker section of the project.

The KServe portion must therefore not claim GPU-backed Kubernetes inference.

## Verification

The serving phase is considered successful only when all of the following are demonstrated.

### Kubernetes Nodes

    kubectl get nodes

must show both nodes as:

    Ready

### KServe Service

    kubectl get inferenceservice

must show the model service as ready.

### Scale to Zero

After sufficient idle time, no predictor pod should remain running.

### Scale From Zero

Sending an HTTP request must cause a serving pod to be created.

### Inference

The HTTP request must receive a valid prediction.

### Scale Down

After the configured idle period, the predictor must return to zero replicas.

## Portfolio Evidence

Keep screenshots concise.

Recommended evidence:

1. Two-node Kubernetes cluster healthy
2. KServe `InferenceService` ready
3. Predictor scaled to zero
4. Predictor pod created following an inference request
5. Successful HTTP prediction

The README should contain the explanation. Screenshots are supporting evidence rather than the documentation itself.

## Non-Goals

This phase does not attempt to implement:

- highly available Kubernetes control planes
- production DNS
- production ingress architecture
- distributed persistent storage
- GPU passthrough into VirtualBox
- multi-GPU scheduling
- production Kubernetes security hardening
- production service-mesh architecture
- production model approval

## Portfolio Positioning

The complete project demonstrates:

    data versioning
          |
          v
    reproducible training
          |
          v
    GPU experiments
          |
          v
    MLflow tracking and governance
          |
          v
    Slurm scheduling
          |
          v
    Kubernetes / KServe serving

The serving phase demonstrates infrastructure capability independently from the model-quality decision made earlier in the project.
