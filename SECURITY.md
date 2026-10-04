# Reporting security issues

Report suspected vulnerabilities privately to the maintainer at
**even0758@gmail.com**. Include the affected revision, configuration, expected
behavior, and a minimal reproduction. Remove credentials, cluster telemetry, and
other sensitive data from the report.

Use GitHub issues for ordinary bugs and feature requests. Security fixes target
the current `main` branch; older revisions do not have a separate maintenance
schedule.

Before deploying MAVER, review the [deployment boundaries](README.md#limitations-and-known-issues),
[Kubernetes deployment guide](deploy/k8s/README.md), and
[approval controls](docs/operations.md). The API currently requires network-level
access restrictions, and pod execution requires human approval by default.
