# Kira

**K**ubernetes **I**ncident **R**ecovery **A**gent: a memory-augmented multi-agent
pipeline that diagnoses and repairs live microservice incidents using a privately
served open-weight language model.

An alert enters over HTTP. A read-only root-cause analysis role investigates cluster
state, metrics, logs and traces and emits a typed `RootCause`. A planning role turns
that record into a typed remediation `Plan`. An execution role, the only one holding
write-capable tools, applies the plan under a check-and-set discipline. Optional
mechanisms add an independent verification loop that can hand work back, and a gated
second diagnostic opinion informed by an experience store of past incidents.

No telemetry leaves the cluster: the model runs on your own infrastructure.

> **Status: under active construction.** This repository is a clean-room rebuild of a
> research prototype into a publishable artifact. See `refactor/` for the design
> specifications and the build order.

## Licence

MIT. See [LICENSE](LICENSE).
