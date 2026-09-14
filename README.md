
---

## Demo Evidence

### Slurm Scheduling

The Slurm lab demonstrates finite-job scheduling across multiple compute nodes, including normal execution and resource contention.

#### Running Job

![Slurm running job](docs/screenshots/slurm/slurm-running-job.png)

#### Resource Contention

A second job remains pending while the cluster resources are occupied, then becomes schedulable once resources are released.

![Slurm resource contention](docs/screenshots/slurm/slurm-resource-contention.png)

---

### KServe Scale-to-Zero

The serving platform uses KServe in Knative mode.

When the model is idle, the predictor scales completely to zero.

#### Idle — CLI

No predictor pods are running.

![KServe idle CLI](docs/screenshots/kserve/kserve-idle-cli.png)

#### Idle — Grafana

The observability dashboard shows zero serving replicas and no active predictor resource consumption.

![KServe idle Grafana](docs/screenshots/kserve/kserve-idle-grafana.png)

---

### KServe Scale-from-Zero

An inference request triggers a Knative cold start and provisions the predictor workload.

#### Active — CLI

The predictor pod is created and becomes Ready.

![KServe active CLI](docs/screenshots/kserve/kserve-active-cli.png)

#### Active — Grafana

Prometheus and Grafana expose the replica transition together with predictor CPU and memory utilization.

![KServe active Grafana](docs/screenshots/kserve/kserve-active-grafana.png)

The demonstrated serving lifecycle is:

    idle
      |
      v
    0 predictor replicas
      |
      | inference request
      v
    Knative cold start
      |
      v
    predictor provisioned
      |
      v
    inference served
      |
      v
    CPU / memory observable
      |
      | idle timeout
      v
    0 predictor replicas

> The deployed LSTM is intentionally marked `quality_rejected_demo_only`.
> It is used to demonstrate the infrastructure lifecycle and was not promoted
> through the model quality gate.

