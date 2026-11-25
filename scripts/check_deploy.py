"""Check the rendered GitOps manifests against the demo's rules.

    kubectl kustomize deploy/app | python scripts/check_deploy.py

Reads YAML on stdin and the files in deploy/app, needs no cluster, prints every broken
rule and exits 1 on any.
"""

import importlib.util
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import yaml


def _load_promote():
    # A sibling script, not a package: load it by path.
    path = Path(__file__).resolve().parent / "promote_images.py"
    spec = importlib.util.spec_from_file_location("promote_images", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


promote = _load_promote()

APP = Path(__file__).resolve().parents[1] / "deploy" / "app"
KUSTOMIZATION = "kustomization.yaml"
PINS = "images/kustomization.yaml"
# Nothing else: patches, transformers or image overrides could change any pin.
KUSTOMIZATION_KEYS = {
    "apiVersion",
    "kind",
    "namespace",
    "resources",
    "components",
    "configMapGenerator",
}
GENERATOR_KEYS = {"name", "literals"}
LOCAL_RESOURCE = re.compile(r"[a-z][a-z-]*\.yaml")
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
WAVED_KINDS = WORKLOAD_KINDS | {"Service"}
# Every workload and its sync wave; a Service shares its workload's wave.
WAVES = {"db": "0", "redis": "0", "migrate": "1", "api": "2"}
WAVE = "argocd.argoproj.io/sync-wave"
SYNC_OPTIONS = "argocd.argoproj.io/sync-options"
HOOK = "argocd.argoproj.io/hook"
HOOK_DELETE = "argocd.argoproj.io/hook-delete-policy"
# Besides the wave. Anything more, such as a hook or Replace=true, could delete data.
EXTRA_ANNOTATIONS = {
    ("StatefulSet", "db"): {SYNC_OPTIONS: "Prune=false,Delete=false"},
    ("Job", "migrate"): {HOOK: "Sync", HOOK_DELETE: "BeforeHookCreation"},
}
# Workload -> its pin in deploy/app/images/.
PIN_NAMES = {"api": "app", "migrate": "migrate", "db": "postgres"}
REDIS = "redis:7.4.11@sha256:44960b3100c48159a66d28c18b0588c193492e25f7551d60e51dd464fc3eca5b"
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
POD_IDS = ("runAsUser", "runAsGroup", "fsGroup")
POD_SECURITY_KEYS = {"runAsNonRoot", "seccompProfile", *POD_IDS}
HOST_KEYS = ("hostNetwork", "hostPID", "hostIPC", "hostUsers", "hostAliases")
VOLUME_TYPES = {"emptyDir", "configMap", "persistentVolumeClaim"}
CLAIM_KEYS = {"apiVersion", "kind", "metadata", "spec"}
CLAIM_SPEC_KEYS = {"accessModes", "volumeMode", "resources"}
DATA_PATH = "/var/lib/postgresql"
MAX_CONNECTIONS = ["-c", "max_connections=100"]
DB_REPLICAS = 1
MIGRATE_COMMAND = ["alembic", "upgrade", "head"]
API_REPLICAS = 2
API_PROBES = {"readinessProbe": "/health/ready", "livenessProbe": "/health/live"}
# The connection budget: (2 replicas + 1 surge) x 10 connections.
API_MAX_SURGE = 1
POOL_SETTING = "DB_POOL_SIZE"
POOL_SIZE = re.compile(r"[1-9]|10")

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


def rule_annotations(docs: Docs) -> Problems:
    for doc in docs:
        expected = dict(EXTRA_ANNOTATIONS.get((doc.get("kind"), _name(doc)), {}))
        if doc.get("kind") in WAVED_KINDS:
            expected[WAVE] = WAVES.get(_name(doc))
        if _annotations(doc) != expected:
            yield f"{doc.get('kind')} {_name(doc)}: annotations are not the expected ones"


def rule_containers(docs: Docs) -> Problems:
    for name, _, pod in _workloads(docs):
        if len(_containers(pod)) != 1:
            yield f"{name}: not exactly one container"


def rule_images(docs: Docs) -> Problems:
    try:
        pins = promote.parse_pins((APP / PINS).read_text())
    except promote.PromoteError as error:
        yield str(error)
        return
    expected = {"redis": REDIS}
    for name, pin_name in PIN_NAMES.items():
        pin = pins[pin_name]
        expected[name] = f"{promote.IMAGES[pin_name]}:sha-{pin.commit}@{pin.digest}"
    for name, container in _all_containers(docs):
        if container.get("image") != expected.get(name):
            yield f"{name}: image is not its pin"


def rule_pod_security(docs: Docs) -> Problems:
    for name, _, pod in _workloads(docs):
        security = pod.get("securityContext") or {}
        if security.get("runAsNonRoot") is not True:
            yield f"{name}: runAsNonRoot is not set"
        for key in POD_IDS:
            value = security.get(key)
            if type(value) is not int or value <= 0:
                yield f"{name}: {key} is not a non-root number"
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
    if db["spec"].get("replicas") != DB_REPLICAS:
        yield f"db: replicas is not {DB_REPLICAS}"
    if "persistentVolumeClaimRetentionPolicy" in db["spec"]:
        yield "db: its claims may be deleted"
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
    container = job["spec"]["template"]["spec"]["containers"][0]
    if container.get("command") != MIGRATE_COMMAND or "args" in container:
        yield f"migrate: the command is not {' '.join(MIGRATE_COMMAND)}"
    if job["spec"].get("backoffLimit") != 0:
        yield "migrate: the Job retries by itself"


def rule_api(docs: Docs) -> Problems:
    api = _one(docs, "Deployment", "api")
    if api is None:
        yield "api: no Deployment"
        return
    if api["spec"].get("replicas") != API_REPLICAS:
        yield f"api: replicas is not {API_REPLICAS}"
    if _rolling(api).get("maxUnavailable") != 0:
        yield "api: maxUnavailable is not 0"
    container = api["spec"]["template"]["spec"]["containers"][0]
    for probe, path in API_PROBES.items():
        if (container.get(probe) or {}).get("httpGet") != {"path": path, "port": "http"}:
            yield f"api: {probe} does not read {path}"


def _rolling(deployment: dict) -> dict:
    return (deployment["spec"].get("strategy") or {}).get("rollingUpdate") or {}


def rule_connections(docs: Docs) -> Problems:
    api = _one(docs, "Deployment", "api")
    if api is None:
        yield "api: no Deployment"
        return
    surge = _rolling(api).get("maxSurge")
    if type(surge) is not int or surge > API_MAX_SURGE:
        yield f"api: maxSurge is not at most {API_MAX_SURGE}"
    # The image's own command runs one Uvicorn worker.
    for container in _containers(api["spec"]["template"]["spec"]):
        if "command" in container or "args" in container:
            yield "api: replaces the image's command"

    # Settings ignore case.
    settings = []
    for name, container in _all_containers(docs):
        settings += [(v["name"], v.get("value")) for v in container.get("env") or []]
        if any("prefix" in source for source in container.get("envFrom") or []):
            yield f"{name}: envFrom renames settings"
    for doc in docs:
        if doc.get("kind") == "ConfigMap":
            settings += (doc.get("data") or {}).items()
    for key, value in settings:
        if key.upper() == POOL_SETTING and not POOL_SIZE.fullmatch(str(value)):
            yield f"{POOL_SETTING} is not 1 to 10"


def _kustomization(folder: Path) -> Problems:
    """What the kustomization itself may say."""
    base = yaml.safe_load((folder / KUSTOMIZATION).read_text())
    if not isinstance(base, dict) or not set(base) <= KUSTOMIZATION_KEYS:
        yield f"{KUSTOMIZATION}: only {', '.join(sorted(KUSTOMIZATION_KEYS))} are allowed"
        return
    if base.get("namespace") != NAMESPACE or base.get("components") != ["images"]:
        yield f"{KUSTOMIZATION}: unexpected namespace or components"
    for generated in base.get("configMapGenerator") or []:
        if not set(generated) <= GENERATOR_KEYS:
            yield f"{KUSTOMIZATION}: a ConfigMap may only have {sorted(GENERATOR_KEYS)}"
    resources = base.get("resources") or []
    if not all(LOCAL_RESOURCE.fullmatch(resource) for resource in resources):
        yield f"{KUSTOMIZATION}: a resource is not a file in this folder"
        return

    # No other file: Argo CD reads overrides from .argocd-source*.yaml.
    expected = {KUSTOMIZATION, PINS, *resources}
    files = {str(p.relative_to(folder)) for p in folder.rglob("*") if not p.is_dir()}
    if files != expected:
        yield f"deploy/app holds unexpected or missing files: {sorted(files ^ expected)}"
    if any(path.is_symlink() for path in folder.rglob("*")):
        yield "deploy/app holds a link"
    promote.parse_pins((folder / PINS).read_text())


def rule_source(folder: Path) -> Problems:
    """What could change the render without showing in a manifest rule."""
    try:
        yield from _kustomization(folder)
    except (*MALFORMED, OSError, promote.PromoteError) as error:
        yield f"deploy/app: {type(error).__name__}"


# What reading an unexpected document shape raises.
MALFORMED = (yaml.YAMLError, AttributeError, IndexError, KeyError, TypeError, ValueError)

RULES: list[Callable[[Docs], Problems]] = [
    rule_kinds,
    rule_workloads,
    rule_annotations,
    rule_containers,
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
    rule_connections,
]


def check(docs: Docs) -> list[str]:
    problems = []
    for rule in RULES:
        try:
            problems += rule(docs)
        except MALFORMED:
            problems.append(f"{rule.__name__}: malformed manifest")
    return problems


def main() -> int:
    try:
        docs = [doc for doc in yaml.safe_load_all(sys.stdin) if doc is not None]
    except yaml.YAMLError:
        print("the input is not YAML", file=sys.stderr)
        return 1
    problems = [*rule_source(APP), *check(docs)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"PASS: {len(docs)} manifests follow {len(RULES)} rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
