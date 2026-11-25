"""Check the rendered GitOps manifests against the demo's rules.

    kubectl kustomize deploy/app | python scripts/check_deploy.py

Reads YAML on stdin, needs no cluster, prints every broken rule and exits 1 on any.
"""

import re
import sys
from collections.abc import Callable, Iterator

import yaml

NAMESPACE = "fastapi-demo"
# The same six kinds as deploy/argocd/project.yaml.
KINDS = {
    ("v1", "ConfigMap"),
    ("v1", "Service"),
    ("apps/v1", "Deployment"),
    ("apps/v1", "StatefulSet"),
    ("batch/v1", "Job"),
    ("networking.k8s.io/v1", "NetworkPolicy"),
}
WORKLOAD_KINDS = {"Deployment", "StatefulSet", "Job"}
# Every workload and its sync wave; a Service shares its workload's wave.
WAVES = {"db": "0", "redis": "0", "migrate": "1", "api": "2"}
WAVE = "argocd.argoproj.io/sync-wave"
SYNC_OPTIONS = "argocd.argoproj.io/sync-options"
HOOK = "argocd.argoproj.io/hook"
HOOK_DELETE = "argocd.argoproj.io/hook-delete-policy"
# The published image of each workload built here.
REGISTRY = "ghcr.io/manshoor/fastapi-ai-agent-platform"
OWN_IMAGES = {"api": REGISTRY, "migrate": REGISTRY, "db": f"{REGISTRY}-postgres"}
OWN_TAG = re.compile(r":sha-[0-9a-f]{40}@sha256:[0-9a-f]{64}")
PINNED = re.compile(r"[\w./-]+(:[\w.-]+)?@sha256:[0-9a-f]{64}")
# Secret -> the workloads that may read it.
SECRET_READERS = {"db-credentials": {"db", "migrate", "api"}, "api-secrets": {"api"}}
CREDENTIAL_NAME = re.compile(r"(PASSWORD|PASSWD|SECRET|KEY|TOKEN|CREDENTIALS?)$", re.IGNORECASE)
CREDENTIAL_URL = re.compile(r"://[^/\s@]*:[^/\s@]+@")
CONTAINER_SECURITY = {
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "capabilities": {"drop": ["ALL"]},
}
POD_SECURITY_KEYS = {"runAsNonRoot", "runAsUser", "runAsGroup", "fsGroup", "seccompProfile"}
HOST_KEYS = ("hostNetwork", "hostPID", "hostIPC", "hostUsers", "hostAliases")
VOLUME_TYPES = {"emptyDir", "configMap", "persistentVolumeClaim"}
CLAIM_KEYS = {"apiVersion", "kind", "metadata", "spec"}
CLAIM_SPEC_KEYS = {"accessModes", "volumeMode", "resources"}
DATA_PATH = "/var/lib/postgresql"
MAX_CONNECTIONS = ["-c", "max_connections=100"]
API_REPLICAS = 2

Docs = list[dict]
Problems = Iterator[str]


# Reading the documents


def _name(doc: dict) -> str:
    return doc.get("metadata", {}).get("name", "?")


def _annotations(doc: dict) -> dict:
    return doc.get("metadata", {}).get("annotations") or {}


def _workloads(docs: Docs) -> Iterator[tuple[str, dict, dict]]:
    """Name, document and pod spec of every workload."""
    for doc in docs:
        if doc.get("kind") in WORKLOAD_KINDS:
            yield _name(doc), doc, doc["spec"]["template"]["spec"]


def _containers(pod: dict) -> list[dict]:
    return [*(pod.get("initContainers") or []), *pod["containers"]]


def _all_containers(docs: Docs) -> Iterator[tuple[str, dict]]:
    for name, _, pod in _workloads(docs):
        for container in _containers(pod):
            yield name, container


def _one(docs: Docs, kind: str, name: str) -> dict | None:
    found = [doc for doc in docs if doc.get("kind") == kind and _name(doc) == name]
    return found[0] if len(found) == 1 else None


# Rules


def rule_kinds(docs: Docs) -> Problems:
    for doc in docs:
        if (doc.get("apiVersion"), doc.get("kind")) not in KINDS:
            yield f"{doc.get('kind')} {_name(doc)}: kind is not allowed"
        if doc.get("metadata", {}).get("namespace") != NAMESPACE:
            yield f"{doc.get('kind')} {_name(doc)}: not in namespace {NAMESPACE}"


def rule_workloads(docs: Docs) -> Problems:
    names = sorted(name for name, _, _ in _workloads(docs))
    if names != sorted(WAVES):
        yield f"workloads are {names}, expected {sorted(WAVES)}"


def rule_waves(docs: Docs) -> Problems:
    for doc in docs:
        if doc.get("kind") not in WORKLOAD_KINDS | {"Service"}:
            continue
        if _annotations(doc).get(WAVE) != WAVES.get(_name(doc)):
            yield f"{doc['kind']} {_name(doc)}: wrong sync wave"


def rule_images(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        image = container.get("image", "")
        if not PINNED.fullmatch(image):
            yield f"{name}: image is not pinned by digest"
        own = OWN_IMAGES.get(name)
        if own and not (image.startswith(own) and OWN_TAG.fullmatch(image[len(own) :])):
            yield f"{name}: image is not {own} at a commit and digest"


def rule_pod_security(docs: Docs) -> Problems:
    for name, _, pod in _workloads(docs):
        security = pod.get("securityContext") or {}
        if security.get("runAsNonRoot") is not True:
            yield f"{name}: runAsNonRoot is not set"
        user = security.get("runAsUser")
        if type(user) is not int or user <= 0:
            yield f"{name}: runAsUser is not a non-root number"
        if security.get("seccompProfile") != {"type": "RuntimeDefault"}:
            yield f"{name}: seccomp profile is not RuntimeDefault"
        if set(security) - POD_SECURITY_KEYS:
            yield f"{name}: unexpected pod securityContext field"
        if any(key in pod for key in HOST_KEYS):
            yield f"{name}: uses a host setting"


def rule_pod_access(docs: Docs) -> Problems:
    for name, _, pod in _workloads(docs):
        if pod.get("automountServiceAccountToken") is not False:
            yield f"{name}: mounts a service account token"
        if "serviceAccountName" in pod or "serviceAccount" in pod:
            yield f"{name}: names a service account"
        if pod.get("enableServiceLinks") is not False:
            yield f"{name}: enableServiceLinks is not false"
        for volume in pod.get("volumes") or []:
            if not set(volume) - {"name"} <= VOLUME_TYPES:
                yield f"{name}: volume type is not allowed"


def rule_container_security(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        # Exact: an extra field could undo the pod's settings.
        if container.get("securityContext") != CONTAINER_SECURITY:
            yield f"{name}: container securityContext is not the expected one"
        if any("hostPort" in port for port in container.get("ports") or []):
            yield f"{name}: uses a host port"


def rule_memory(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        if not (container.get("resources") or {}).get("limits", {}).get("memory"):
            yield f"{name}: no memory limit"


def rule_secret_access(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        if any("secretRef" in source for source in container.get("envFrom") or []):
            yield f"{name}: reads a whole Secret"
        for variable in container.get("env") or []:
            reference = (variable.get("valueFrom") or {}).get("secretKeyRef")
            if reference and name not in SECRET_READERS.get(reference.get("name"), set()):
                yield f"{name}: may not read Secret {reference.get('name')}"


def rule_no_credentials(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        for variable in container.get("env") or []:
            if CREDENTIAL_NAME.search(variable["name"]) and "secretKeyRef" not in (
                variable.get("valueFrom") or {}
            ):
                yield f"{name}: {variable['name']} is not read from a Secret"
    for doc in docs:
        if doc.get("kind") == "ConfigMap":
            for key in {**(doc.get("data") or {}), **(doc.get("binaryData") or {})}:
                if CREDENTIAL_NAME.search(key):
                    yield f"ConfigMap {_name(doc)}: {key} looks like a credential"
        if CREDENTIAL_URL.search(yaml.safe_dump(doc, width=sys.maxsize)):
            yield f"{doc.get('kind')} {_name(doc)}: holds a URL with a password"


def rule_services(docs: Docs) -> Problems:
    for doc in docs:
        if doc.get("kind") != "Service":
            continue
        spec = doc["spec"]
        if spec.get("type", "ClusterIP") != "ClusterIP" or "externalIPs" in spec:
            yield f"Service {_name(doc)}: reachable from outside the cluster"


def rule_database(docs: Docs) -> Problems:
    db = _one(docs, "StatefulSet", "db")
    if db is None:
        yield "db: no StatefulSet"
        return
    options = set(_annotations(db).get(SYNC_OPTIONS, "").split(","))
    if not {"Prune=false", "Delete=false"} <= options:
        yield "db: Argo CD may remove it"
    container = db["spec"]["template"]["spec"]["containers"][0]
    if container.get("args", [])[-2:] != MAX_CONNECTIONS:
        yield "db: max_connections is not 100"
    paths = {mount["name"]: mount["mountPath"] for mount in container.get("volumeMounts") or []}
    claims = db["spec"].get("volumeClaimTemplates") or []
    if not any(paths.get(_name(claim)) == DATA_PATH for claim in claims):
        yield f"db: no claim at {DATA_PATH}"
    probe = ((container.get("readinessProbe") or {}).get("exec") or {}).get("command") or []
    if probe[:3] != ["pg_isready", "-h", "127.0.0.1"]:
        yield "db: readiness does not use TCP"


def rule_claims(docs: Docs) -> Problems:
    for name, doc, _ in _workloads(docs):
        for claim in doc["spec"].get("volumeClaimTemplates") or []:
            if not CLAIM_KEYS <= set(claim) or not CLAIM_SPEC_KEYS <= set(claim["spec"]):
                yield f"{name}: volumeClaimTemplates defaults are not spelled out"


def rule_migrate(docs: Docs) -> Problems:
    job = _one(docs, "Job", "migrate")
    if job is None:
        yield "migrate: no Job"
        return
    annotations = _annotations(job)
    if annotations.get(HOOK) != "Sync" or annotations.get(HOOK_DELETE) != "BeforeHookCreation":
        yield "migrate: not a Sync hook that replaces its last run"
    if job["spec"].get("backoffLimit") != 0:
        yield "migrate: the Job retries by itself"


def rule_api(docs: Docs) -> Problems:
    api = _one(docs, "Deployment", "api")
    if api is None:
        yield "api: no Deployment"
        return
    if api["spec"].get("replicas") != API_REPLICAS:
        yield f"api: replicas is not {API_REPLICAS}"
    strategy = api["spec"].get("strategy") or {}
    if (strategy.get("rollingUpdate") or {}).get("maxUnavailable") != 0:
        yield "api: maxUnavailable is not 0"


RULES: list[Callable[[Docs], Problems]] = [
    rule_kinds,
    rule_workloads,
    rule_waves,
    rule_images,
    rule_pod_security,
    rule_pod_access,
    rule_container_security,
    rule_memory,
    rule_secret_access,
    rule_no_credentials,
    rule_services,
    rule_database,
    rule_claims,
    rule_migrate,
    rule_api,
]


def check(docs: Docs) -> list[str]:
    return [problem for rule in RULES for problem in rule(docs)]


def main() -> int:
    docs = [doc for doc in yaml.safe_load_all(sys.stdin) if doc is not None]
    problems = check(docs)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"PASS: {len(docs)} manifests follow {len(RULES)} rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
