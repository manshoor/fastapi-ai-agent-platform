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
# macOS leaves these in any folder it shows; kustomize reads none.
FINDER_FILE = ".DS_Store"
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
WAVES = {
    "db": "0",
    "redis": "0",
    "kafka": "0",
    "provider": "0",
    "migrate": "1",
    "topic-init": "1",
    "api": "2",
    "dispatcher": "2",
    "worker": "2",
    "rag-worker": "2",
}
NAME_LABEL = "app.kubernetes.io/name"
WAVE = "argocd.argoproj.io/sync-wave"
SYNC_OPTIONS = "argocd.argoproj.io/sync-options"
HOOK = "argocd.argoproj.io/hook"
HOOK_DELETE = "argocd.argoproj.io/hook-delete-policy"
# Besides the wave. Anything more, such as a hook or Replace=true, could delete data.
SYNC_HOOK = {HOOK: "Sync", HOOK_DELETE: "BeforeHookCreation"}
EXTRA_ANNOTATIONS = {
    ("StatefulSet", "db"): {SYNC_OPTIONS: "Prune=false,Delete=false"},
    ("Job", "migrate"): SYNC_HOOK,
    ("Job", "topic-init"): SYNC_HOOK,
}
# Workload -> its pin in deploy/app/images/.
PIN_NAMES = {
    "api": "app",
    "dispatcher": "app",
    "worker": "app",
    "rag-worker": "app",
    "migrate": "migrate",
    "db": "postgres",
    "provider": "simulator",
}
REDIS = "redis:7.4.11@sha256:44960b3100c48159a66d28c18b0588c193492e25f7551d60e51dd464fc3eca5b"
KAFKA = "apache/kafka:4.1.2@sha256:5cc2a2fd93fa2687b44015eee04fb2c3edd9e526bd64bf8bec5ff1e268772e0e"
OUTSIDE_IMAGES = {"redis": REDIS, "kafka": KAFKA, "topic-init": KAFKA}
# Secret -> the workloads that may read it.
SECRET_READERS = {
    "db-credentials": {"db", "migrate", "api", "dispatcher", "worker", "rag-worker"},
    "api-secrets": {"api"},
}
DATABASE_CLIENTS = SECRET_READERS["db-credentials"] - {"db"}
CREDENTIAL_NAME = re.compile(r"(PASSWORD|PASSWD|SECRET|KEY|TOKEN|CREDENTIALS?)$", re.IGNORECASE)
# The one literal credential: the simulator takes any key, and no other provider is used.
FAKE_KEY_NAME = "OPENAI_API_KEY"
FAKE_KEY = "local-simulator-only-not-a-real-key"
FAKE_KEY_USERS = {"api", "worker", "rag-worker"}
PROVIDER_URL = "http://provider:8080/v1"
CREDENTIAL_URL = re.compile(r"://[^/\s@]*:[^/\s@]+@")
CONTAINER_SECURITY = {
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "capabilities": {"drop": ["ALL"]},
}
# Its entrypoint writes settings under /opt/kafka.
WRITABLE_ROOT = {"kafka"}
CONTAINER_KEYS = {
    "name",
    "image",
    "command",
    "args",
    "env",
    "envFrom",
    "ports",
    "readinessProbe",
    "livenessProbe",
    "lifecycle",
    "resources",
    "securityContext",
    "volumeMounts",
}
POD_IDS = ("runAsUser", "runAsGroup", "fsGroup")
POD_SECURITY_KEYS = {"runAsNonRoot", "seccompProfile", *POD_IDS}
HOST_KEYS = ("hostNetwork", "hostPID", "hostIPC", "hostUsers", "hostAliases")
VOLUME_TYPES = {"emptyDir", "configMap", "persistentVolumeClaim"}
CLAIM_KEYS = {"apiVersion", "kind", "metadata", "spec"}
CLAIM_SPEC_KEYS = {"accessModes", "volumeMode", "resources"}
DATA_PATH = "/var/lib/postgresql"
DB_ARGS = ["postgres", "-c", "max_connections=100"]
DB_MOUNTS = [
    {"name": "data", "mountPath": DATA_PATH},
    {"name": "run", "mountPath": "/var/run/postgresql"},
    {"name": "tmp", "mountPath": "/tmp"},
]
DB_REPLICAS = 1
MIGRATE_COMMAND = ["alembic", "upgrade", "head"]
API_REPLICAS = 2
API_PROBES = {"readinessProbe": "/health/ready", "livenessProbe": "/health/live"}
API_MAX_SURGE = 1
# The connection budget: 2 API x 10 + dispatcher 2 + two workers x 4, and more in a rollout.
STEADY_CONNECTIONS = 30
ROLLOUT_CONNECTIONS = 59
POOL_SETTING = "DB_POOL_SIZE"
POOL_SIZE = re.compile(r"[1-9]|10")
# Alembic uses one connection without a pool.
JOB_CONNECTIONS = 1
WORKERS = ("dispatcher", "worker", "rag-worker")
WORKER_REPLICAS = 1
WORKER_STOP_SECONDS = 50
# Two parser children of at most 1 GiB each, plus the worker.
EXACT_MEMORY = {"rag-worker": "2560Mi"}
KAFKA_DATA_PATH = "/var/lib/kafka/data"
KAFKA_REPLICAS = 1
KAFKA_VOTERS = "1@localhost:9093"
TOPICS = (
    "ai.item-rewrite.requested.v1",
    "ai.item-rewrite.dead-letter.v1",
    "rag.document-events.v1",
    "rag.document-events.dead-letter.v1",
)
KAFKA_TOOL = "/opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092"
TOPIC_SCRIPT = (
    f"for topic in {' '.join(TOPICS)}; do\n"
    f'  {KAFKA_TOOL} --create --if-not-exists --topic "$topic" --partitions 6'
    " --replication-factor 1 --config retention.ms=86400000"
    " --config retention.bytes=268435456 --config max.message.bytes=65536\n"
    f"  {KAFKA_TOOL} --describe --topic \"$topic\" | grep -F 'PartitionCount: 6' > /dev/null"
    " || { echo 'Topic requires exactly six partitions' >&2; exit 1; }\n"
    "done\n"
)
REDIS_ARGS = (
    "redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru --appendonly no --save"
).split() + [""]


def _secret(name: str, key: str) -> dict:
    return {"secretKeyRef": {"name": name, "key": key}}


def _config(key: str) -> dict:
    return {"configMapKeyRef": {"name": "database-settings", "key": key}}


def _database(pool: int) -> dict:
    return {"DB_PASSWORD": _secret("db-credentials", "password"), POOL_SETTING: str(pool)}


# Everything a container is told, per workload. Only these texts may be literal, so a
# credential cannot hide under an innocent name. Workloads not listed get nothing.
ENV: dict[str, dict[str, object]] = {
    "db": {
        "POSTGRES_DB": _config("DB_NAME"),
        "POSTGRES_USER": _config("DB_USER"),
        "POSTGRES_PASSWORD": _secret("db-credentials", "password"),
    },
    "migrate": {"DB_PASSWORD": _secret("db-credentials", "password")},
    "topic-init": {"LOG_DIR": "/tmp"},
    "api": {
        **_database(10),
        "JWT_SECRET": _secret("api-secrets", "jwt-secret"),
        "ASSISTANT_TOTP_KEY": _secret("api-secrets", "totp-key"),
        "ASSISTANT_MEMORY_KEY": _secret("api-secrets", "memory-key"),
        FAKE_KEY_NAME: FAKE_KEY,
    },
    "dispatcher": _database(2),
    "worker": {**_database(4), FAKE_KEY_NAME: FAKE_KEY},
    "rag-worker": {**_database(4), FAKE_KEY_NAME: FAKE_KEY},
}
ENV_FROM = {
    "kafka": ["kafka-settings"],
    "migrate": ["database-settings"],
    "api": ["database-settings", "api-settings", "ai-settings", "provider-settings"],
    "dispatcher": ["database-settings", "ai-settings"],
    "worker": ["database-settings", "ai-settings", "provider-settings"],
    "rag-worker": ["database-settings", "ai-settings", "provider-settings"],
}
TEXTS: dict[str, dict[str, object]] = {
    "db": {"args": DB_ARGS},
    "redis": {"args": REDIS_ARGS},
    "migrate": {"command": MIGRATE_COMMAND},
    "topic-init": {"command": ["/bin/bash", "-ec"], "args": [TOPIC_SCRIPT]},
    "api": {"lifecycle": {"preStop": {"sleep": {"seconds": 5}}}},
    "dispatcher": {"command": ["python", "-m", "app.ai.dispatcher"]},
    "worker": {"command": ["python", "-m", "app.ai.worker"]},
    "rag-worker": {"command": ["python", "-m", "app.rag.worker"]},
}
TEXT_KEYS = ("command", "args", "lifecycle")
CONFIG_KEYS = {
    "database-settings": {"DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"},
    "api-settings": {
        "REDIS_URL",
        "AUTH_TOKEN_TTL_SECONDS",
        "ASSISTANT_ENABLED",
        "ASSISTANT_DEMO_ENABLED",
        "ASSISTANT_MEMORY_ENABLED",
    },
    "ai-settings": {"AI_ENABLED", "AI_POLL_SECONDS", "RAG_ENABLED", "KAFKA_BOOTSTRAP_SERVERS"},
    "provider-settings": {"OPENAI_BASE_URL"},
    "kafka-settings": {
        "CLUSTER_ID",
        "KAFKA_NODE_ID",
        "KAFKA_PROCESS_ROLES",
        "KAFKA_LISTENERS",
        "KAFKA_ADVERTISED_LISTENERS",
        "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
        "KAFKA_CONTROLLER_LISTENER_NAMES",
        "KAFKA_CONTROLLER_QUORUM_VOTERS",
        "KAFKA_INTER_BROKER_LISTENER_NAME",
        "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR",
        "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR",
        "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR",
        "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS",
        "KAFKA_AUTO_CREATE_TOPICS_ENABLE",
        "KAFKA_LOG_RETENTION_HOURS",
        "KAFKA_LOG_RETENTION_BYTES",
        "KAFKA_LOG_SEGMENT_BYTES",
        "KAFKA_LOG_DIRS",
        "KAFKA_HEAP_OPTS",
    },
}
# The suffix kustomize adds to a generated ConfigMap's name.
CONFIG_HASH = re.compile(r"-[2456789bcdfghkmt]{10}$")
# Receiver -> its port and the workloads that may call it. Nothing else is open but DNS.
PATHS: dict[str, tuple[int, tuple[str, ...]]] = {
    "db": (5432, ("api", "migrate", "dispatcher", "worker", "rag-worker")),
    "redis": (6379, ("api",)),
    "kafka": (9092, ("topic-init", "dispatcher", "worker", "rag-worker")),
    "provider": (8080, ("api", "worker", "rag-worker")),
}
DNS_PODS = {
    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
    "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
}
DNS_PORTS = [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]
PROBE_KEYS = ("startupProbe", "readinessProbe", "livenessProbe")
PROBE_HANDLERS = {"exec", "httpGet", "tcpSocket", "grpc"}
# A worker has no probe: a Kafka outage must not restart-loop it.
PROBES: dict[str, dict[str, dict]] = {
    "db": {
        "readinessProbe": {
            "exec": {"command": ["pg_isready", "-h", "127.0.0.1", "-U", "app", "-d", "app"]}
        }
    },
    "redis": {"readinessProbe": {"exec": {"command": ["redis-cli", "PING"]}}},
    "kafka": {"readinessProbe": {"tcpSocket": {"port": "kafka"}}},
    "provider": {"readinessProbe": {"httpGet": {"path": "/health/live", "port": "http"}}},
    "api": {
        probe: {"httpGet": {"path": path, "port": "http"}} for probe, path in API_PROBES.items()
    },
}

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


def _base_name(name: str) -> str:
    return CONFIG_HASH.sub("", name)


def _config_maps(docs: Docs, name: str) -> Docs:
    return [d for d in docs if d.get("kind") == "ConfigMap" and _base_name(_name(d)) == name]


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
    for name, doc, _ in _workloads(docs):
        # Network policies select pods by this label: another name is another pod's access.
        label = {NAME_LABEL: name}
        template = doc["spec"]["template"]["metadata"]
        if template != {"labels": label}:
            yield f"{name}: pod metadata is not just its name label"
        if doc["kind"] != "Job" and doc["spec"].get("selector") != {"matchLabels": label}:
            yield f"{name}: does not select its own pods"


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
        for container in _containers(pod):
            extra = set(container) - CONTAINER_KEYS
            if extra:
                yield f"{name}: unexpected container field {sorted(extra)}"


def rule_images(docs: Docs) -> Problems:
    try:
        pins = promote.parse_pins((APP / PINS).read_text())
    except promote.PromoteError as error:
        yield str(error)
        return
    expected = dict(OUTSIDE_IMAGES)
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
        expected = {**CONTAINER_SECURITY, "readOnlyRootFilesystem": name not in WRITABLE_ROOT}
        if container.get("securityContext") != expected:
            yield f"{name}: container securityContext is not the expected one"
        if any("hostPort" in port for port in container.get("ports") or []):
            yield f"{name}: uses a host port"


def rule_memory(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        limit = (container.get("resources") or {}).get("limits", {}).get("memory")
        if not limit:
            yield f"{name}: no memory limit"
        if limit != EXACT_MEMORY.get(name, limit):
            yield f"{name}: memory limit is not {EXACT_MEMORY[name]}"


def rule_secret_access(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        if any("secretRef" in source for source in container.get("envFrom") or []):
            yield f"{name}: reads a whole Secret"
        for variable in container.get("env") or []:
            reference = (variable.get("valueFrom") or {}).get("secretKeyRef")
            if reference and name not in SECRET_READERS.get(reference.get("name"), set()):
                yield f"{name}: may not read Secret {reference.get('name')}"


def _environment(container: dict) -> dict[str, object] | None:
    """Variable -> its literal, or its Secret or ConfigMap key. None if one is repeated."""
    found: dict[str, object] = {}
    for variable in container.get("env") or []:
        name, source = variable["name"], variable.get("valueFrom")
        if name in found or set(variable) - {"name", "value", "valueFrom"}:
            return None
        if source and "configMapKeyRef" in source:
            reference = source["configMapKeyRef"]
            reference = {**reference, "name": _base_name(reference["name"])}
            source = {**source, "configMapKeyRef": reference}
        found[name] = source or variable.get("value")
    return found


def _sources(container: dict) -> list[str] | None:
    """The ConfigMaps read whole. None if anything else is."""
    for source in container.get("envFrom") or []:
        if set(source) != {"configMapRef"} or set(source["configMapRef"]) != {"name"}:
            return None
    return [_base_name(s["configMapRef"]["name"]) for s in container.get("envFrom") or []]


def rule_no_credentials(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        for variable in container.get("env") or []:
            fake = (
                name in FAKE_KEY_USERS
                and variable["name"] == FAKE_KEY_NAME
                and variable.get("value") == FAKE_KEY
            )
            if (
                CREDENTIAL_NAME.search(variable["name"])
                and not fake
                and "secretKeyRef" not in (variable.get("valueFrom") or {})
            ):
                yield f"{name}: {variable['name']} is not read from a Secret"
    for doc in docs:
        if doc.get("kind") == "ConfigMap":
            for key in {**(doc.get("data") or {}), **(doc.get("binaryData") or {})}:
                if CREDENTIAL_NAME.search(key):
                    yield f"ConfigMap {_name(doc)}: {key} looks like a credential"
        if CREDENTIAL_URL.search(yaml.safe_dump(doc, width=sys.maxsize)):
            yield f"{doc.get('kind')} {_name(doc)}: holds a URL with a password"


def rule_known_texts(docs: Docs) -> Problems:
    """Only the known variables, settings and arguments: names alone can hide a credential."""
    for name, container in _all_containers(docs):
        if _environment(container) != ENV.get(name, {}):
            yield f"{name}: env is not the expected one"
        if _sources(container) != ENV_FROM.get(name, []):
            yield f"{name}: envFrom is not the expected one"
        texts = {key: container[key] for key in TEXT_KEYS if key in container}
        if texts != TEXTS.get(name, {}):
            yield f"{name}: command, args or lifecycle is not the expected one"
    names = [_base_name(_name(doc)) for doc in docs if doc.get("kind") == "ConfigMap"]
    if sorted(names) != sorted(CONFIG_KEYS):
        yield f"ConfigMaps are {sorted(names)}, expected {sorted(CONFIG_KEYS)}"
        return
    for name, keys in CONFIG_KEYS.items():
        (doc,) = _config_maps(docs, name)
        if set(doc["data"]) != keys or "binaryData" in doc:
            yield f"ConfigMap {name}: keys are not the expected ones"


def rule_provider(docs: Docs) -> Problems:
    """The simulator is the only provider, so no paid call is possible."""
    for doc in _config_maps(docs, "provider-settings"):
        if doc["data"] == {"OPENAI_BASE_URL": PROVIDER_URL}:
            return
    yield f"provider-settings: the provider is not {PROVIDER_URL}"


def rule_services(docs: Docs) -> Problems:
    for doc in docs:
        if doc.get("kind") != "Service":
            continue
        spec = doc["spec"]
        if spec.get("type", "ClusterIP") != "ClusterIP" or "externalIPs" in spec:
            yield f"Service {_name(doc)}: reachable from outside the cluster"
        if spec.get("selector") != {NAME_LABEL: _name(doc)}:
            yield f"Service {_name(doc)}: does not select its own workload"


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
    # Anything else could move the data off its claim: PGDATA, -D, or a mount inside it.
    if "command" in container or container.get("args") != DB_ARGS:
        yield f"db: the arguments are not {' '.join(DB_ARGS)}"
    if _environment(container) != ENV["db"] or "envFrom" in container:
        yield "db: env is not the expected one"
    if container.get("volumeMounts") != DB_MOUNTS:
        yield "db: mounts are not the expected ones"
    claims = db["spec"].get("volumeClaimTemplates") or []
    if [_name(claim) for claim in claims] != [DB_MOUNTS[0]["name"]]:
        yield f"db: no single claim at {DATA_PATH}"
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


def _pool(container: dict) -> int | None:
    """The one literal pool size, or None."""
    # Settings ignore case.
    sizes = [v for v in container.get("env") or [] if v["name"].upper() == POOL_SETTING]
    if len(sizes) != 1 or not POOL_SIZE.fullmatch(str(sizes[0].get("value"))):
        return None
    return int(sizes[0]["value"])


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

    # One size per workload, in its own env.
    for name, container in _all_containers(docs):
        if any("prefix" in source for source in container.get("envFrom") or []):
            yield f"{name}: envFrom renames settings"
    for doc in docs:
        if doc.get("kind") == "ConfigMap":
            if any(key.upper() == POOL_SETTING for key in doc.get("data") or {}):
                yield f"ConfigMap {_name(doc)}: sets {POOL_SETTING}"

    # A rolling update may run every old pod beside every new one; Recreate never does.
    steady = rollout = 0
    for name, doc, pod in _workloads(docs):
        if name not in DATABASE_CLIENTS:
            continue
        if doc["kind"] == "Job":
            rollout += JOB_CONNECTIONS
            continue
        pool, replicas = _pool(pod["containers"][0]), doc["spec"].get("replicas")
        if pool is None or type(replicas) is not int:
            yield f"{name}: {POOL_SETTING} is not one value from 1 to 10, or no replicas"
            continue
        recreate = (doc["spec"].get("strategy") or {}).get("type") == "Recreate"
        steady += replicas * pool
        rollout += replicas * pool * (1 if recreate else 2)
    if steady > STEADY_CONNECTIONS or rollout > ROLLOUT_CONNECTIONS:
        yield f"{steady} database connections, {rollout} in a rollout: over the budget"


def rule_probes(docs: Docs) -> Problems:
    for name, container in _all_containers(docs):
        found = {
            probe: {k: v for k, v in container[probe].items() if k in PROBE_HANDLERS}
            for probe in PROBE_KEYS
            if probe in container
        }
        if found != PROBES.get(name, {}):
            yield f"{name}: probes are not the expected ones"


def rule_workers(docs: Docs) -> Problems:
    for name in WORKERS:
        worker = _one(docs, "Deployment", name)
        if worker is None:
            yield f"{name}: no Deployment"
            continue
        spec = worker["spec"]
        if spec.get("replicas") != WORKER_REPLICAS or spec.get("strategy") != {"type": "Recreate"}:
            yield f"{name}: not exactly one at a time"
        if spec["template"]["spec"].get("terminationGracePeriodSeconds") != WORKER_STOP_SECONDS:
            yield f"{name}: stop time is not {WORKER_STOP_SECONDS} s"


def rule_kafka(docs: Docs) -> Problems:
    service, kafka = _one(docs, "Service", "kafka"), _one(docs, "StatefulSet", "kafka")
    if service is None or kafka is None:
        yield "kafka: no Service or StatefulSet"
        return
    spec = service["spec"]
    if spec.get("clusterIP") != "None" or spec.get("publishNotReadyAddresses") is not True:
        yield "kafka: the Service is not headless, or hides a starting broker"
    if kafka["spec"].get("replicas") != KAFKA_REPLICAS:
        yield f"kafka: replicas is not {KAFKA_REPLICAS}"
    mounts = kafka["spec"]["template"]["spec"]["containers"][0].get("volumeMounts")
    claims = [_name(claim) for claim in kafka["spec"].get("volumeClaimTemplates") or []]
    if mounts != [{"name": "data", "mountPath": KAFKA_DATA_PATH}] or claims != ["data"]:
        yield f"kafka: no single claim at {KAFKA_DATA_PATH}"
    voters = [
        d["data"].get("KAFKA_CONTROLLER_QUORUM_VOTERS")
        for d in _config_maps(docs, "kafka-settings")
    ]
    if voters != [KAFKA_VOTERS]:
        yield f"kafka: the voter is not {KAFKA_VOTERS}"


def rule_topics(docs: Docs) -> Problems:
    job = _one(docs, "Job", "topic-init")
    if job is None:
        yield "topic-init: no Job"
        return
    container = job["spec"]["template"]["spec"]["containers"][0]
    if any(container.get(key) != text for key, text in TEXTS["topic-init"].items()):
        yield "topic-init: the script is not the expected one"
    if job["spec"].get("backoffLimit") != 0:
        yield "topic-init: the Job retries by itself"


def _selector(name: str) -> dict:
    return {"podSelector": {"matchLabels": {NAME_LABEL: name}}}


def _tcp(port: int) -> list[dict]:
    return [{"protocol": "TCP", "port": port}]


def expected_policies() -> dict[str, dict]:
    """The spec of every NetworkPolicy, from PATHS."""
    policies = {
        "default-deny": {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]},
        "dns": {
            "podSelector": {},
            "policyTypes": ["Egress"],
            "egress": [{"to": [DNS_PODS], "ports": DNS_PORTS}],
        },
    }
    for receiver, (port, senders) in PATHS.items():
        callers = [_selector(sender) for sender in senders]
        policies[f"to-{receiver}"] = {
            **_selector(receiver),
            "policyTypes": ["Ingress"],
            "ingress": [{"from": callers, "ports": _tcp(port)}],
        }
        for sender in senders:
            policy = policies.setdefault(
                f"from-{sender}", {**_selector(sender), "policyTypes": ["Egress"], "egress": []}
            )
            policy["egress"].append({"to": [_selector(receiver)], "ports": _tcp(port)})
    return policies


def rule_network(docs: Docs) -> Problems:
    """Exactly default-deny, DNS and PATHS: policies add up, so one more opens a path."""
    found = [doc for doc in docs if doc.get("kind") == "NetworkPolicy"]
    expected = expected_policies()
    names = sorted(_name(doc) for doc in found)
    if names != sorted(expected):
        yield f"network policies are {names}, expected {sorted(expected)}"
        return
    for doc in found:
        if doc["spec"] != expected[_name(doc)]:
            yield f"NetworkPolicy {_name(doc)}: not the expected paths"


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
    found = [p for p in folder.rglob("*") if not p.is_dir() and p.name != FINDER_FILE]
    files = {str(path.relative_to(folder)) for path in found}
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
    rule_known_texts,
    rule_provider,
    rule_services,
    rule_database,
    rule_claims,
    rule_migrate,
    rule_api,
    rule_connections,
    rule_probes,
    rule_workers,
    rule_kafka,
    rule_topics,
    rule_network,
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
