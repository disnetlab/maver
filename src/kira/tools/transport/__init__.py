"""Transport implementations behind the cluster and telemetry tools.

The direct transport is the default and needs only a kubeconfig and the telemetry
endpoints. An environment adapter may supply its own transport instead; the core
never branches on which environment it is running under.
"""
