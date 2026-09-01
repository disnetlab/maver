# Kubernetes manifests

```bash
kubectl apply -k deploy/k8s/overlays/local        # kind, a locally built image
kubectl apply -k deploy/k8s/overlays/production   # a real cluster, digest-pinned
```

`base/` is not meant to be applied on its own. It has no image digest, its ConfigMap
points at service names your cluster may not have, and it binds nothing in a workload
namespace, so an overlay is where a deployment becomes a deployment.

## Two traps, both already paid for once

**Apply an overlay with `-k`, never a file inside it with `-f`.** The files under
`overlays/` are patches rather than manifests. `kubectl apply -f overlays/*/configmap-patch.yaml`
replaces `kira-config` with only the keys that patch names and silently drops every other
one. Settings default conservatively, so the pod comes back up with the alert consumer
disabled, the memory store off, and no error anywhere saying so. The prototype lost a run
to exactly this.

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
| `namespace.yaml` | the `kira` namespace, labelled for the restricted Pod Security Standard |
| `serviceaccount.yaml` | the identity every rule below is granted to |
| `rbac-cluster-read.yaml` | `kira-cluster-read` and the one binding that is cluster-wide. Reads only |
| `rbac-namespace-read.yaml` | `kira-namespace-read`. A ClusterRole, deliberately unbound |
| `rbac-namespace-write.yaml` | `kira-namespace-write`. A ClusterRole, deliberately unbound |
| `configmap.yaml` | the fifteen-odd settings a deployment actually sets |
| `pvc.yaml` | 10Gi ReadWriteOnce at `/workspace` |
| `deployment.yaml` | one replica, `Recreate`, three probes, a read-only root filesystem |
| `service.yaml` | ClusterIP, port 80 to the container's 8080 |
| `networkpolicy.yaml` | default-deny both ways, with a short allow list |
| `poddisruptionbudget.yaml` | `minAvailable: 0`, which is deliberate |

`base/secret.example.yaml` is a template and is **not** in `kustomization.yaml`, so
`apply -k` cannot overwrite a real Secret with its placeholders.

## Granting access to a workload namespace

The base grants nothing inside any workload namespace. Nothing is diagnosable until you
bind the namespaced roles where you want the agent to work:

```bash
sed 's/REPLACE_TARGET_NAMESPACE/shop/' \
  overlays/production/rolebindings.example.yaml > overlays/production/rolebinding-shop.yaml
# then add the file to overlays/production/kustomization.yaml
```

Bind the read role in every namespace the agent should be able to see, and the write role
only where it should be able to act. Do not bind either in a platform, observability, or
fault-injection namespace.

## What the agent may and may not do

Reads are broad, writes are narrow, and the split is enforced twice: by RBAC here, and by
the approval gate in the process, which holds every cluster write until a person decides
it unless `KIRA_AUTHORITY__MODE` says otherwise.

Cluster-wide, the agent can read nodes, persistent volumes, storage classes, volume
attachments and node metrics, and can `get` a single namespace by name. That is what
tells a pending pod on an unschedulable node apart from an application fault. It cannot
list namespaces and it cannot write anything cluster-scoped.

Per namespace, it can read the workloads, pods and their logs, services, endpoints and
endpoint slices, config maps, claims, events, quotas, autoscalers, ingresses, network
policies, pod metrics and KEDA objects. With the write role it can additionally create,
patch and delete workloads, pods, services, config maps, claims, jobs, ingresses,
autoscalers and KEDA objects, evict a pod, and run `kubectl exec`.

Three things are deliberately not granted:

- **Secrets**, read or write. `overlays/production/rbac-secrets-write.yaml` is the opt-in
  fragment, and it says what opting in costs. Without it, one line of the namespace
  snapshot — the check for a workload referring to a secret that does not exist — reports
  `(unavailable)`, and nothing else changes.
- **Node mutation.** `cordon`, `uncordon`, `drain` and `taint` are cluster-scoped writes.
  The kubectl tool will classify them and the gate will offer them; the API server
  refuses them, which is the answer we want.
- **`kubectl exec` in a read-only namespace.** Running a command inside a container
  cannot be proven read-only, so it lives in the write role.

## The approval surface has no authentication

This is tracked and not yet fixed. The API that lists pending writes and approves them is
served on the same port as everything else, and anything that can reach port 8080 can
approve a cluster write. Until authentication lands the containment is the NetworkPolicy:

- ingress on 8080 only from namespaces labelled `kira.dev/operator=true`;
- no `Ingress` object, in this repository or in your cluster;
- `kubectl port-forward` for an operator, which the API server proxies through the
  kubelet and which therefore keeps working under the policy.

```bash
kubectl label namespace <your-operator-namespace> kira.dev/operator=true
kubectl -n kira port-forward svc/kira 8080:80
```

The `local` overlay removes the policy, because kind ships no controller to enforce it
and an inert policy that looks enforced is worse than an absent one.

## Validating a change

```bash
kubectl kustomize deploy/k8s/overlays/production | kubeconform -strict -summary
kubectl apply --dry-run=client -k deploy/k8s/overlays/production
```

The rendered production ConfigMap must contain every key the base ConfigMap has. An
overlay that drops one is the first trap above, happening quietly.
