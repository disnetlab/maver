# Kubernetes manifests

Run these commands from the repository root with a kind cluster already running and
with matching node and image architectures. `make image` builds `linux/amd64`; customize
the Docker build platform first when the kind nodes use another architecture.

```bash
make image
kind load docker-image maver:local
kubectl apply -k deploy/k8s/overlays/local
```

The build tags target `ghcr.io/disnetlab/maver`; `make image` does not publish them.
Publish the image to a registry your cluster can pull from before using the production
overlay.
For the production overlay, replace its all-zero image digest and add workload-namespace
role bindings first, then run `kubectl apply -k deploy/k8s/overlays/production`.

`base/` is not meant to be applied on its own. It has no image digest, its ConfigMap
points at service names your cluster may not have, and it binds nothing in a workload
namespace, so an overlay is where a deployment becomes a deployment.

## Applying overlays safely

**Apply an overlay with `-k`, never a file inside it with `-f`.** The files under
`overlays/` are patches rather than manifests. Running
`kubectl apply -f deploy/k8s/overlays/*/configmap-patch.yaml` replaces `maver-config`
with only the keys that patch names and silently drops every other one. Settings default
conservatively, so the pod comes back up with the alert consumer disabled and the memory
store off, with no startup error identifying the dropped keys.

**A `configMapGenerator` cannot read files outside its own kustomization root** without
`--load-restrictor LoadRestrictionsNone`, which turns off a safety check. That is why the
model profiles and arm files are baked into the image instead of being projected from
`configs/` at the repository root: the manifest that looks like it should work does not,
and the workaround is worse than the alternative. An overlay that genuinely needs to
override a profile copies the YAML into the overlay directory and generates from there,
which also gives the generated ConfigMap a content hash and therefore a rolling restart
when the profile changes.

## What is in the base

| File | What it is |
|---|---|
| `namespace.yaml` | the `maver` namespace, labelled for the restricted Pod Security Standard |
| `serviceaccount.yaml` | the identity every rule below is granted to |
| `rbac-cluster-read.yaml` | `maver-cluster-read` and the one binding that is cluster-wide. Reads only |
| `rbac-namespace-read.yaml` | `maver-namespace-read`. A ClusterRole, deliberately unbound |
| `rbac-namespace-write.yaml` | `maver-namespace-write`. A ClusterRole, deliberately unbound |
| `configmap.yaml` | the fifteen-odd settings a deployment actually sets |
| `pvc.yaml` | 10Gi ReadWriteOnce at `/workspace` |
| `deployment.yaml` | one replica, `Recreate`, three probes, a read-only root filesystem |
| `service.yaml` | ClusterIP, port 80 to the container's 8080 |
| `networkpolicy.yaml` | default-deny both ways, with a short allow list |
| `poddisruptionbudget.yaml` | `minAvailable: 0`, which is deliberate |

`base/secret.example.yaml` is a template and is **not** in `kustomization.yaml`, so
`apply -k` cannot overwrite a real Secret with its placeholders.

## Configure recovery verification

Add the following keys under `data` in your overlay's `configmap-patch.yaml`:

```yaml
data:
  MAVER_VERIFY__ENABLED: "true"
  MAVER_VERIFY__HANDBACK_RETRIES: "1"
```

Apply the complete overlay with `-k` and restart the deployment to reload its
environment; changing this static ConfigMap alone does not restart existing pods.
The cap permits one Verify return to RCA or Planning within the shared remediation
budget. Zero permits none; omitting the cap inherits the shared limit. Execution
replans and inconclusive rechecks keep their own accounting.

To select named pipelines, mount a YAML file and set `MAVER_CUSTOMISATION` to its
container path. The [built-in example](../../examples/verify-handbacks.yaml) defines
two pipelines with different caps. Custom Python implementations must be included
in the image and registered through `src/maver/custom.py`. See
[the customisation guide](../../docs/customisation.md) for configuration and selection.

## Granting access to a workload namespace

The base grants nothing inside any workload namespace. Nothing is diagnosable until you
bind the namespaced roles where you want the agent to work:

```bash
sed 's/REPLACE_TARGET_NAMESPACE/shop/' \
  deploy/k8s/overlays/production/rolebindings.example.yaml \
  > deploy/k8s/overlays/production/rolebinding-shop.yaml
# then add the file to deploy/k8s/overlays/production/kustomization.yaml
```

Bind the read role in every namespace the agent should be able to see, and the write role
only where it should be able to act. Do not bind either in a platform, observability, or
fault-injection namespace.

## What the agent may and may not do

Reads are broad, writes are narrow, and the split is enforced twice: by RBAC here, and by
the approval gate in the process, which holds every cluster write until a person decides
it unless `MAVER_AUTHORITY__MODE` says otherwise. `kubectl exec` is a separate boundary:
it requires human approval by default even in `auto` mode or under a matching `auto`
rule, because an in-container command can mutate application state without a Kubernetes
API write. A diagnostic role may therefore exec after approval; role name alone is not
a read-only guarantee.

Cluster-wide, the agent can read nodes, persistent volumes, storage classes, volume
attachments, priority classes, ingress classes and node metrics, and can `get` a single
namespace by name. That is what tells a pending pod on an unschedulable node, a pod
naming a priority class that does not exist, and an ordinary application fault apart. It
cannot list namespaces and it cannot write anything cluster-scoped.

Per namespace, it can read the workloads and their revision history, pods and their logs,
services, endpoints and endpoint slices, config maps, claims, service accounts, events,
quotas, disruption budgets, autoscalers, ingresses, network policies, pod metrics and KEDA
objects. With the write role it can additionally create, patch and delete workloads, pods,
services, config maps, claims, service accounts, jobs, ingresses, autoscalers and KEDA
objects, evict a pod, and run `kubectl exec`.

The roles are derived from what the kubectl tool can actually issue rather than assembled
by hand: `tests/unit/test_deploy.py` reads the verb and resource vocabulary out of
`maver.tools.kubernetes` and fails if a kind the tool knows is neither granted below nor
listed as a decision. Six things are decisions:

- **Secrets**, read or write. `overlays/production/rbac-secrets-write.yaml` is the opt-in
  fragment, and it says what opting in costs. Without it, one line of the namespace
  snapshot — the check for a workload referring to a secret that does not exist — reports
  `(unavailable)`, and nothing else changes.
- **RBAC objects** — roles, bindings, and their cluster-scoped forms. Anything that can
  write a `RoleBinding` can grant itself every other line on this page.
- **Node mutation.** `cordon`, `uncordon`, `drain` and `taint` are cluster-scoped writes.
  The kubectl tool will classify them and the gate will offer them; the API server
  refuses them, which is the answer we want.
- **Cluster-wide pod reads**, which is why `kubectl describe node` fails: it lists pods in
  every namespace. `kubectl get node <name> -o yaml` answers the node's own state and is
  granted. Granting the describe means letting the agent read every workload in the
  cluster, including the namespaces the read role is deliberately never bound in.
- **Disruption budget writes.** They are readable, because an eviction a budget refuses
  looks like a stuck drain from every other angle; widening one so the eviction goes
  through is disabling the control rather than repairing the fault.
- **`kubectl exec` in a read-only namespace.** Running a command inside a container
  cannot be proven read-only, so it lives in the write role and requires human approval
  by default. Production deployments should prefer application-specific read tools or
  read-only application credentials over arbitrary exec.

## The approval surface has no authentication

The API that lists pending writes and approves them is served on the same port as
everything else, and anything that can reach port 8080 can approve a cluster write.
The shipped deployment contains this unauthenticated surface with the NetworkPolicy:

- ingress on 8080 only from namespaces labelled `maver.dev/operator=true`;
- no `Ingress` object, in this repository or in your cluster;
- `kubectl port-forward` for an operator, which the API server proxies through the
  kubelet and which therefore keeps working under the policy.

```bash
kubectl label namespace <your-operator-namespace> maver.dev/operator=true
kubectl -n maver port-forward svc/maver 8080:80
```

The `local` overlay removes the policy, because kind ships no controller to enforce it
and an inert policy that looks enforced is worse than an absent one.

## Validating a change

```bash
uv run pytest tests/unit/test_deploy.py   # renders both overlays, no cluster
kubectl kustomize deploy/k8s/overlays/production | kubeconform -strict -summary
```

`tests/unit/test_deploy.py` renders both overlays with
`kubectl kustomize`, validates every rendered `MAVER_`-prefixed key against the settings
model, builds `Settings` from the rendered environment so a value the validators refuse
fails here rather than in a crash loop, checks the probe paths against the routes
`maver.api.health` actually declares, and compares the RBAC verbs with the tool's own.

**`kubectl apply --dry-run=client` is not an offline check**, which is worth knowing
before you point it at a cluster you did not mean to touch. It still performs API
discovery against the current context, `--validate=false` included: with `KUBECONFIG`
pointed at nothing it fails with `unable to recognize ...: the server could not find the
requested resource` rather than rendering. `kubectl kustomize` is the pure-rendering
command.

The rendered production ConfigMap must contain every key the base ConfigMap has. An
overlay that drops one is the first trap above, happening quietly; the test asserts it.

## Upgrading existing deployments

The supported package and CLI are `maver`. Settings use the `MAVER_` prefix;
unprefixed prototype aliases remain available for recorded experiments. Update
operator environment files and deployment settings before upgrading.

The PostgreSQL schema defaults to `maver_memory`. To reuse an existing schema, set
`MAVER_MEMORY__POSTGRES_SCHEMA` to its actual name in your environment file or
ConfigMap. Point it at the same database and retain a backup before upgrading;
changing the default does not move existing records.

Compose uses physical workspace volumes `docker_maver-workspace` and
`docker_maver-eval-workspace` by default. To retain an existing volume, set
`MAVER_WORKSPACE_VOLUME` or `MAVER_EVAL_WORKSPACE_VOLUME` to its actual name in the
Compose interpolation environment (for example, export it in the host shell before
running Compose). A service's `maver.env` file configures the container; it does not
supply these Compose interpolation overrides. Changing a volume name starts a separate
workspace rather than migrating its contents.

For Kubernetes, retain the existing namespace and PVC in an upgrade overlay, or
migrate the workspace before starting the new deployment. Applying the new base
alone creates separate resources; it neither migrates existing state nor stops an
existing deployment.

Scheduled job stores with callable references to removed package modules cannot
continue unchanged. Before upgrading, drain pending approvals and scheduled jobs
under the existing deployment, or migrate their stored references to the supported
MAVER modules and validate that the continuations can load. Preserve the database,
job store and workspace during this process. Stop the existing deployment before
starting the upgraded one against the same managed applications.
