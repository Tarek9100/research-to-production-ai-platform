# K3s + KServe Serving Lab

Two-node Kubernetes lab for the research-to-production AI platform.

## Topology

| Node | Role | IP | CPU | RAM |
|---|---|---|---:|---:|
| `k3s-cp1` | K3s server/control plane | `192.168.56.30` | 2 | 3 GB |
| `k3s-w1` | K3s worker | `192.168.56.31` | 2 | 3 GB |

The environment runs in VirtualBox on Windows and is managed by Vagrant
from WSL2.

The final serving stack will be:

    K3s
      |
      +-- cert-manager
      +-- Istio
      +-- Knative Serving
      +-- KServe
            |
            +-- InferenceService

This is a laptop-scale infrastructure lab, not a production HA cluster.
