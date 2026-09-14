# Slurm Scheduling Lab

This directory contains a laptop-scale Slurm cluster used to practice HPC scheduling and operations.

The purpose is scheduler and infrastructure learning, not performance benchmarking.

## Architecture

    slurm-controller
    192.168.56.20
    slurmctld
          |
          +-------------------------+
          |                         |
          v                         v
    slurm-compute01           slurm-compute02
    192.168.56.21             192.168.56.22
    slurmd                    slurmd
    2 CPUs                    2 CPUs
    ~1.9 GB RAM               ~1.9 GB RAM

Cluster:

- 1 Slurm controller
- 2 compute nodes
- 4 schedulable CPU cores total
- Munge authentication
- `cpu` partition
- backfill scheduler
- consumable CPU/memory resources

## Environment

The VMs run in VirtualBox on Windows while Vagrant is invoked from WSL2.

Because VirtualBox runs in the Windows networking namespace, additional handling was required for:

- SSH forwarding from WSL to the Windows host
- disabling the default `/vagrant` synced folder for the WSL-native repository
- Ubuntu box UART configuration
- WSL-to-Windows Vagrant communication

These are laptop-lab compatibility concerns rather than Slurm architecture requirements.

## Authentication

Slurm uses Munge for authentication.

All nodes use the same Munge key and synchronized clocks.

Munge was verified across nodes with successful credential decode.

## Slurm Components

### Controller

`slurm-controller` runs `slurmctld`.

The controller:

- accepts job submissions
- maintains node state
- determines scheduling decisions
- allocates resources

### Compute Nodes

Both compute nodes run `slurmd`.

`slurmd`:

- registers node resources with the controller
- receives job launch requests
- starts job steps
- reports node and job status

## Scheduler Configuration

The cluster uses:

    SchedulerType=sched/backfill
    SelectType=select/cons_tres
    SelectTypeParameters=CR_Core_Memory

Resources are therefore scheduled as consumable CPU and memory resources.

## cgroup Compatibility Finding

The original configuration used:

    ProctrackType=proctrack/cgroup

The Ubuntu 22.04 VMs use unified cgroup v2:

    cgroup2fs

The packaged Slurm 21.08.5 build failed during `slurmd` initialization because it expected the legacy freezer cgroup:

    cgroup namespace 'freezer' not mounted
    unable to create freezer cgroup namespace
    slurmd initialization failed

For this learning lab the process tracker was changed to:

    ProctrackType=proctrack/linuxproc

After the change both nodes registered successfully:

    PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
    cpu*         up  infinite     2 idle  slurm-compute[01-02]

This is a lab compatibility workaround, not the preferred configuration for a modern production Slurm deployment.

## Multi-Node Execution

Interactive execution was verified with:

    srun \
      --nodes=2 \
      --ntasks=2 \
      hostname

Result:

    slurm-compute02
    slurm-compute01

This proves that the controller can allocate resources and launch tasks on both compute nodes.

## Batch Scheduling

A normal batch workload requests:

- 1 node
- 1 task
- 1 CPU
- 256 MB RAM

Example:

    sbatch basic_job.sbatch

During execution:

    ST=R
    NodeList=slurm-compute01

The completed job reported:

    JobState=COMPLETED
    ExitCode=0:0
    TRES=cpu=1,mem=256M,node=1

## Resource Contention

The cluster has four total CPUs.

The `occupy_cluster.sbatch` workload requests all four:

- 2 nodes
- 2 tasks
- 2 CPUs per task

While that job was running:

    JOBID  NAME            ST  NODES  NODELIST(REASON)
    6      occupy-cluster  R   2      slurm-compute[01-02]

A second workload was then submitted.

Because no CPU resources remained, Slurm correctly kept it pending:

    JOBID  NAME          ST  NODES  NODELIST(REASON)
    7      waiting-demo  PD  1      (Resources)

Inspection showed:

    JobState=PENDING
    Reason=Resources
    TRES=cpu=1,mem=256M,node=1

After job 6 released its resources, job 7 automatically transitioned:

    PENDING
       |
       v
    RUNNING
       |
       v
    COMPLETED

This demonstrates scheduler resource arbitration rather than merely executing commands remotely.

## Job Cancellation

A long-running workload was submitted as job 8.

While running:

    JobState=RUNNING

It was cancelled using:

    scancel 8

Final state:

    JobState=CANCELLED
    ExitCode=143:0
    RunTime=00:00:29

Exit code 143 corresponds to termination by SIGTERM.

## Failure Investigation

Jobs 4 and 5 initially failed with:

    JobState=FAILED
    Reason=NonZeroExitCode
    ExitCode=1:0

Slurm itself successfully allocated all requested resources.

The failure occurred because the batch submission working directory was:

    /home/vagrant/jobs

That path existed on the controller but not on the compute nodes.

This demonstrated an important HPC architecture concept:

> Slurm schedules compute resources; it is not a distributed filesystem.

The lab therefore changed batch working directories and output paths to locations present on each worker.

Production HPC systems commonly solve this through shared storage such as:

- NFS
- Lustre
- BeeGFS
- GPFS / Spectrum Scale

## Accounting

`sacct` was tested but reported:

    Slurm accounting storage is disabled

This is expected because the lab does not currently deploy:

    slurmdbd
    +
    SQL accounting database

The distinction between scheduler state and persistent accounting was therefore also demonstrated.

## Demonstrated Concepts

The lab covers:

- `slurmctld`
- `slurmd`
- Munge
- partitions
- nodes
- jobs
- job steps
- `sinfo`
- `scontrol`
- `srun`
- `sbatch`
- `squeue`
- `scancel`
- CPU allocation
- memory requests
- TRES
- multi-node execution
- `RUNNING`
- `PENDING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`
- pending reasons
- resource contention
- automatic rescheduling
- process tracking
- cgroup compatibility
- shared-storage implications
- accounting architecture

## Hardware Scope

The Slurm VMs do not contain physical GPUs.

Real GPU experiments elsewhere in this project run directly against the laptop's NVIDIA GTX 1650 through WSL2 and Docker.

Any future Slurm GPU/GRES example without GPU passthrough must therefore be treated explicitly as simulated scheduler configuration rather than physical GPU execution.
