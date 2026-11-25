"""Every check_deploy.py rule accepts deploy/app and rejects a bad edit. No cluster."""

import copy
import importlib.util
import io
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "app"
_SPEC = importlib.util.spec_from_file_location("check_deploy", ROOT / "scripts" / "check_deploy.py")
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

DIGEST = "sha256:" + "0" * 64


def render() -> list[dict]:
    """The parts of `kubectl kustomize` this folder uses; the image has no kubectl."""
    # Anything else in the kustomization would change the real render but not this one.
    assert list(checker.rule_source(APP)) == []
    base = yaml.safe_load((APP / "kustomization.yaml").read_text())
    pins = yaml.safe_load((APP / "images" / "kustomization.yaml").read_text())["images"]
    images = {pin["name"]: f"{pin['newName']}:{pin['newTag']}@{pin['digest']}" for pin in pins}
    docs = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": generated["name"]},
            "data": dict(literal.split("=", 1) for literal in generated["literals"]),
        }
        for generated in base["configMapGenerator"]
    ]
    for resource in base["resources"]:
        docs += yaml.safe_load_all((APP / resource).read_text())
    for doc in docs:
        doc["metadata"]["namespace"] = base["namespace"]
        if doc["kind"] in checker.WORKLOAD_KINDS:
            for container in doc["spec"]["template"]["spec"]["containers"]:
                container["image"] = images.get(container["image"], container["image"])
    return docs


def find(docs: list[dict], kind: str, name: str) -> dict:
    (doc,) = [doc for doc in docs if (doc["kind"], doc["metadata"]["name"]) == (kind, name)]
    return doc


def edit(docs: list[dict], path: str, value: object) -> None:
    """Set `Kind/name/key/key/...`; an ellipsis value deletes the key."""
    kind, name, *keys = path.split("/")
    node = find(docs, kind, name)
    keys = [int(key) if key.isdigit() else key.replace("~", "/") for key in keys]
    for key in keys[:-1]:
        node = node[key]
    if isinstance(node, list) and keys[-1] == len(node):
        node.append(value)
    elif value is ...:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value


POD = "Deployment/api/spec/template/spec"
BOX = f"{POD}/containers/0"
DB = "StatefulSet/db/spec/template/spec"
DB_BOX = f"{DB}/containers/0"
CLAIM = "StatefulSet/db/spec/volumeClaimTemplates/0"
HOOK = "Job/migrate/metadata/annotations/argocd.argoproj.io~"
LITERAL = [{"name": "JWT_SECRET", "value": "x"}]
OTHER_SECRET = [{"name": "A", "valueFrom": {"secretKeyRef": {"name": "other", "key": "a"}}}]
API_SECRET = [{"name": "A", "valueFrom": {"secretKeyRef": {"name": "api-secrets", "key": "a"}}}]
APP_IMAGE = checker.promote.IMAGE
PIN = find(render(), "Deployment", "api")["spec"]["template"]["spec"]["containers"][0]["image"]
HEADER = "Deployment/api/metadata/annotations"
DB_HEADER = "StatefulSet/db/metadata/annotations"
SURGE = "Deployment/api/spec/strategy/rollingUpdate/maxSurge"
REDIS = "Deployment/redis/spec/template/spec/containers/0/image"
REDIS_BOX = "Deployment/redis/spec/template/spec/containers/0"
DB_CLAIMS = "StatefulSet/db/spec/volumeClaimTemplates"
FIELD = [{"name": "DB_POOL_SIZE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}}]
POOL_10 = {"name": "DB_POOL_SIZE", "value": "10"}
FIRST = "spec/template/spec/containers/0"
KAFKA_BOX = f"StatefulSet/kafka/{FIRST}"
WORKER_BOX = f"Deployment/worker/{FIRST}"
RAG_BOX = f"Deployment/rag-worker/{FIRST}"
TOPIC_BOX = f"Job/topic-init/{FIRST}"
READ_ONLY = "securityContext/readOnlyRootFilesystem"
LABELS = "Deployment/provider/spec/template/metadata/labels"
PROVIDER = "ConfigMap/provider-settings/data"
VOTERS = "ConfigMap/kafka-settings/data/KAFKA_CONTROLLER_QUORUM_VOTERS"
FAKE = {"name": "OPENAI_API_KEY", "value": checker.FAKE_KEY}
INIT = [{"name": "i", "image": f"{APP_IMAGE}:sha-{'a' * 40}@{DIGEST}"}]
POOL = [{"name": "db_pool_size", "value": "11"}]
POOL_REFERENCE = [{"name": "DB_POOL_SIZE", "valueFrom": {"fieldRef": {"fieldPath": "a"}}}]
PREFIX = [{"prefix": "DB_", "configMapRef": {"name": "api-settings"}}]
BAD_EDITS = [
    ("unpinned-commit", "rule_images", f"{BOX}/image", f"{APP_IMAGE}:sha-{'a' * 40}@{DIGEST}"),
    ("untagged-pin", "rule_images", f"{BOX}/image", PIN.replace(":sha-", ":x-")),
    ("other-redis", "rule_images", REDIS, f"redis:7.4.11@{DIGEST}"),
    ("app-for-db", "rule_images", f"{DB_BOX}/image", PIN),
    ("own-init", "rule_images", f"{POD}/initContainers", INIT),
    ("pinned-init", "rule_containers", f"{POD}/initContainers", [{"name": "i", "image": PIN}]),
    ("sidecar", "rule_containers", f"{POD}/containers/1", {"name": "i", "image": PIN}),
    ("db-hook", "rule_annotations", f"{DB_HEADER}/argocd.argoproj.io~hook", "Sync"),
    ("db-helm-hook", "rule_annotations", f"{DB_HEADER}/helm.sh~hook", "pre-install"),
    (
        "db-replace",
        "rule_annotations",
        f"{DB_HEADER}/argocd.argoproj.io~sync-options",
        "Prune=false,Delete=false,Replace=true,Force=true",
    ),
    ("api-hook", "rule_annotations", f"{HEADER}/argocd.argoproj.io~hook", "PostSync"),
    ("api-helm-hook", "rule_annotations", f"{HEADER}/helm.sh~hook", "post-install"),
    ("api-ignored", "rule_annotations", f"{HEADER}/argocd.argoproj.io~compare-options", "x"),
    ("config-hook", "rule_annotations", "ConfigMap/api-settings/metadata/annotations", {"a": "b"}),
    (
        "claim-deleted",
        "rule_database",
        "StatefulSet/db/spec/persistentVolumeClaimRetentionPolicy",
        {"whenScaled": "Delete"},
    ),
    ("two-dbs", "rule_database", "StatefulSet/db/spec/replicas", 2),
    ("no-db-replicas", "rule_database", "StatefulSet/db/spec/replicas", ...),
    ("root-group", "rule_pod_security", f"{POD}/securityContext/runAsGroup", 0),
    ("no-group", "rule_pod_security", f"{POD}/securityContext/runAsGroup", ...),
    ("root-volume", "rule_pod_security", f"{POD}/securityContext/fsGroup", 0),
    ("no-volume-group", "rule_pod_security", f"{POD}/securityContext/fsGroup", ...),
    ("no-readiness", "rule_api", f"{BOX}/readinessProbe", ...),
    ("other-readiness", "rule_api", f"{BOX}/readinessProbe/httpGet/path", "/health/live"),
    ("no-liveness", "rule_api", f"{BOX}/livenessProbe", ...),
    ("exec-liveness", "rule_api", f"{BOX}/livenessProbe", {"exec": {"command": ["true"]}}),
    ("surge", "rule_connections", SURGE, 5),
    ("percent-surge", "rule_connections", SURGE, "100%"),
    ("no-surge", "rule_connections", SURGE, ...),
    ("pool-env", "rule_connections", f"{BOX}/env", POOL),
    ("pool-reference", "rule_connections", f"{BOX}/env", POOL_REFERENCE),
    ("pool-config", "rule_connections", "ConfigMap/api-settings/data/DB_POOL_SIZE", "50"),
    ("pool-text", "rule_connections", "ConfigMap/api-settings/data/DB_POOL_SIZE", "x"),
    ("pool-prefix", "rule_connections", f"{BOX}/envFrom", PREFIX),
    ("api-workers", "rule_connections", f"{BOX}/args", ["uvicorn", "--workers", "4"]),
    ("api-command", "rule_connections", f"{BOX}/command", ["env", "DB_POOL_SIZE=50", "uvicorn"]),
    (
        "migrate-command",
        "rule_migrate",
        "Job/migrate/spec/template/spec/containers/0/command",
        ["alembic", "upgrade", "head~1"],
    ),
    ("migrate-args", "rule_migrate", "Job/migrate/spec/template/spec/containers/0/args", ["--sql"]),
    ("secret-kind", "rule_kinds", "ConfigMap/api-settings/kind", "Secret"),
    ("ingress-kind", "rule_kinds", "Service/api/kind", "Ingress"),
    ("other-group", "rule_kinds", "Deployment/api/apiVersion", "evil/v1"),
    ("other-namespace", "rule_kinds", "Service/api/metadata/namespace", "default"),
    ("extra-workload", "rule_workloads", "Deployment/api/metadata/name", "other"),
    ("no-wave", "rule_annotations", "Deployment/api/metadata/annotations", ...),
    (
        "early-api",
        "rule_annotations",
        "Deployment/api/metadata/annotations/argocd.argoproj.io~sync-wave",
        "0",
    ),
    ("tag-only", "rule_images", f"{BOX}/image", f"{APP_IMAGE}:sha-{'a' * 40}"),
    (
        "redis-tag",
        "rule_images",
        "Deployment/redis/spec/template/spec/containers/0/image",
        "redis:7.4",
    ),
    ("short-digest", "rule_images", f"{BOX}/image", f"{APP_IMAGE}@sha256:{'a' * 63}"),
    ("other-image", "rule_images", f"{BOX}/image", f"docker.io/x/y:sha-{'a' * 40}@{DIGEST}"),
    ("digest-no-commit", "rule_images", f"{BOX}/image", f"{APP_IMAGE}@{DIGEST}"),
    ("no-image", "rule_images", f"{BOX}/image", ...),
    ("init-image", "rule_images", f"{POD}/initContainers", [{"name": "i", "image": "busybox"}]),
    ("maybe-root", "rule_pod_security", f"{POD}/securityContext/runAsNonRoot", ...),
    ("root-user", "rule_pod_security", f"{POD}/securityContext/runAsUser", 0),
    ("no-user", "rule_pod_security", f"{POD}/securityContext/runAsUser", ...),
    ("boolean-user", "rule_pod_security", f"{POD}/securityContext/runAsUser", True),
    ("unconfined", "rule_pod_security", f"{POD}/securityContext/seccompProfile/type", "Unconfined"),
    ("sysctl", "rule_pod_security", f"{POD}/securityContext/sysctls", []),
    ("no-context", "rule_pod_security", f"{POD}/securityContext", ...),
    ("host-network", "rule_pod_security", f"{POD}/hostNetwork", True),
    ("token", "rule_pod_access", f"{POD}/automountServiceAccountToken", ...),
    ("account", "rule_pod_access", f"{POD}/serviceAccountName", "x"),
    ("service-links", "rule_pod_access", f"{POD}/enableServiceLinks", ...),
    (
        "host-path",
        "rule_pod_access",
        f"{POD}/volumes/0",
        {"name": "tmp", "hostPath": {"path": "/"}},
    ),
    ("secret-volume", "rule_pod_access", f"{POD}/volumes/0", {"name": "tmp", "secret": {}}),
    ("privileged", "rule_container_security", f"{BOX}/securityContext/privileged", True),
    (
        "escalation",
        "rule_container_security",
        f"{BOX}/securityContext/allowPrivilegeEscalation",
        ...,
    ),
    ("writable", "rule_container_security", f"{BOX}/securityContext/readOnlyRootFilesystem", False),
    (
        "capability",
        "rule_container_security",
        f"{BOX}/securityContext/capabilities/add",
        ["NET_RAW"],
    ),
    ("container-user", "rule_container_security", f"{BOX}/securityContext/runAsUser", 0),
    ("no-container-context", "rule_container_security", f"{BOX}/securityContext", ...),
    ("host-port", "rule_container_security", f"{BOX}/ports/0/hostPort", 80),
    ("no-memory", "rule_memory", f"{BOX}/resources/limits/memory", ...),
    ("no-resources", "rule_memory", f"{BOX}/resources", ...),
    ("other-secret", "rule_secret_access", f"{BOX}/env", OTHER_SECRET),
    ("db-reads-api", "rule_secret_access", f"{DB_BOX}/env", API_SECRET),
    (
        "whole-secret",
        "rule_secret_access",
        f"{BOX}/envFrom",
        [{"secretRef": {"name": "api-secrets"}}],
    ),
    ("literal", "rule_no_credentials", f"{BOX}/env", LITERAL),
    ("config-key", "rule_no_credentials", "ConfigMap/api-settings/data/JWT_SECRET", "x"),
    (
        "config-url",
        "rule_no_credentials",
        "ConfigMap/api-settings/data/REDIS_URL",
        "redis://:pw@redis",
    ),
    ("argument-url", "rule_no_credentials", f"{BOX}/args", ["postgresql://a:b@db/app"]),
    ("short-name", "rule_known_texts", f"{BOX}/env", [{"name": "ADMIN_PASS", "value": "x"}]),
    ("argument", "rule_known_texts", f"{REDIS_BOX}/args", ["redis-server", "--requirepass", "x"]),
    ("config-text", "rule_known_texts", "ConfigMap/api-settings/data/NOTES", "JWT_SECRET=abc"),
    ("binary-config", "rule_known_texts", "ConfigMap/api-settings/binaryData", {"a": "eA=="}),
    ("extra-config", "rule_known_texts", "ConfigMap/api-settings/metadata/name", "other"),
    ("no-config", "rule_known_texts", "ConfigMap/api-settings/kind", "Service"),
    ("field-env", "rule_known_texts", f"{BOX}/env", FIELD),
    ("same-name", "rule_known_texts", f"{BOX}/env", [POOL_10, POOL_10]),
    ("api-hook-command", "rule_known_texts", f"{BOX}/lifecycle/postStart", {"exec": {}}),
    ("db-pgdata", "rule_database", f"{DB_BOX}/env/0", {"name": "PGDATA", "value": "/tmp/data"}),
    ("db-config", "rule_database", f"{DB_BOX}/envFrom", [{"configMapRef": {"name": "x"}}]),
    (
        "db-mount",
        "rule_database",
        f"{DB_BOX}/volumeMounts/0",
        {"name": "tmp", "mountPath": "/var/lib/postgresql/19"},
    ),
    (
        "db-extra-mount",
        "rule_database",
        f"{DB_BOX}/volumeMounts/3",
        {"name": "tmp", "mountPath": "/var/lib/postgresql/19"},
    ),
    ("db-subpath", "rule_database", f"{DB_BOX}/volumeMounts/0/subPath", "x"),
    ("db-command", "rule_database", f"{DB_BOX}/command", ["sh"]),
    ("db-args", "rule_database", f"{DB_BOX}/args", ["-D", "/tmp", "-c", "max_connections=100"]),
    ("db-two-claims", "rule_database", f"{DB_CLAIMS}/1", {"metadata": {"name": "run"}}),
    ("real-key", "rule_no_credentials", f"{BOX}/env", [{**FAKE, "value": "sk-real"}]),
    ("key-in-dispatcher", "rule_no_credentials", "Deployment/dispatcher/" + FIRST + "/env/0", FAKE),
    ("key-renamed", "rule_no_credentials", f"{BOX}/env", [{**FAKE, "name": "OTHER_API_KEY"}]),
    ("other-provider", "rule_provider", f"{PROVIDER}/OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ("extra-provider", "rule_provider", f"{PROVIDER}/OPENAI_ORG", "x"),
    ("no-provider", "rule_provider", "ConfigMap/provider-settings/metadata/name", "other"),
    ("pod-label", "rule_workloads", f"{LABELS}/app.kubernetes.io~name", "api"),
    ("extra-label", "rule_workloads", f"{LABELS}/other", "x"),
    ("other-pods", "rule_workloads", "Deployment/provider/spec/selector/matchLabels", {"a": "b"}),
    ("service-pods", "rule_services", "Service/db/spec/selector/app.kubernetes.io~name", "api"),
    ("startup-field", "rule_containers", f"{BOX}/startupProbe", {"exec": {"command": ["true"]}}),
    ("working-dir", "rule_containers", f"{BOX}/workingDir", "/tmp"),
    ("other-kafka", "rule_images", f"{KAFKA_BOX}/image", f"apache/kafka:4.1.2@{DIGEST}"),
    (
        "app-for-worker",
        "rule_images",
        f"{WORKER_BOX}/image",
        f"{APP_IMAGE}:sha-{'a' * 40}@{DIGEST}",
    ),
    ("writable-worker", "rule_container_security", f"{WORKER_BOX}/{READ_ONLY}", False),
    ("writable-topics", "rule_container_security", f"{TOPIC_BOX}/{READ_ONLY}", False),
    (
        "kafka-privileged",
        "rule_container_security",
        f"{KAFKA_BOX}/securityContext/privileged",
        True,
    ),
    ("small-parser", "rule_memory", f"{RAG_BOX}/resources/limits/memory", "1Gi"),
    ("worker-reads-api", "rule_secret_access", f"{WORKER_BOX}/env/0", API_SECRET[0]),
    ("topic-hook", "rule_annotations", "Job/topic-init/metadata/annotations", {}),
    ("worker-wave", "rule_annotations", "Deployment/worker/metadata/annotations", {}),
    ("worker-pool", "rule_connections", f"{WORKER_BOX}/env/1/value", "10"),
    ("worker-no-pool", "rule_connections", f"{WORKER_BOX}/env/1", ...),
    ("worker-replicas", "rule_connections", "Deployment/rag-worker/spec/replicas", 3),
    ("no-worker-replicas", "rule_connections", "Deployment/worker/spec/replicas", ...),
    ("three-apis", "rule_connections", "Deployment/api/spec/replicas", 3),
    ("worker-liveness", "rule_probes", f"{WORKER_BOX}/livenessProbe", {"tcpSocket": {"port": 1}}),
    ("worker-startup", "rule_probes", f"{WORKER_BOX}/startupProbe", {"tcpSocket": {"port": 1}}),
    ("kafka-exec", "rule_probes", f"{KAFKA_BOX}/readinessProbe", {"exec": {"command": ["true"]}}),
    ("no-provider-probe", "rule_probes", "Deployment/provider/" + FIRST + "/readinessProbe", ...),
    ("two-workers", "rule_workers", "Deployment/worker/spec/replicas", 2),
    ("rolling-worker", "rule_workers", "Deployment/dispatcher/spec/strategy", ...),
    (
        "short-stop",
        "rule_workers",
        "Deployment/rag-worker/spec/template/spec/terminationGracePeriodSeconds",
        30,
    ),
    ("no-dispatcher", "rule_workers", "Deployment/dispatcher/metadata/name", "other"),
    ("kafka-address", "rule_kafka", "Service/kafka/spec/clusterIP", ...),
    ("kafka-hidden", "rule_kafka", "Service/kafka/spec/publishNotReadyAddresses", ...),
    ("kafka-voter", "rule_kafka", f"{VOTERS}", "1@kafka:9093"),
    ("two-kafkas", "rule_kafka", "StatefulSet/kafka/spec/replicas", 3),
    ("kafka-no-claim", "rule_kafka", "StatefulSet/kafka/spec/volumeClaimTemplates", ...),
    ("kafka-mount", "rule_kafka", f"{KAFKA_BOX}/volumeMounts/0/mountPath", "/data"),
    ("no-kafka", "rule_kafka", "Service/kafka/metadata/name", "other"),
    ("topic-script", "rule_topics", f"{TOPIC_BOX}/args/0", "true"),
    ("topic-retries", "rule_topics", "Job/topic-init/spec/backoffLimit", 6),
    ("no-topics", "rule_topics", "Job/topic-init/metadata/name", "other"),
    ("sqs", "rule_known_texts", "ConfigMap/ai-settings/data/RAG_EVENT_SINKS", "kafka,sqs"),
    ("config-prefix", "rule_known_texts", f"{BOX}/envFrom", PREFIX),
    ("node-port", "rule_services", "Service/api/spec/type", "NodePort"),
    ("external-ip", "rule_services", "Service/api/spec/externalIPs", ["1.2.3.4"]),
    ("pruned-db", "rule_annotations", "StatefulSet/db/metadata/annotations", {}),
    ("connections", "rule_database", f"{DB_BOX}/args", ["postgres"]),
    (
        "data-path",
        "rule_database",
        f"{DB_BOX}/volumeMounts/0/mountPath",
        "/var/lib/postgresql/data",
    ),
    ("no-claim", "rule_database", "StatefulSet/db/spec/volumeClaimTemplates", ...),
    ("socket-probe", "rule_database", f"{DB_BOX}/readinessProbe/exec/command", ["pg_isready"]),
    ("no-db", "rule_database", "StatefulSet/db/kind", "Deployment"),
    ("claim-kind", "rule_claims", f"{CLAIM}/kind", ...),
    ("claim-mode", "rule_claims", f"{CLAIM}/spec/volumeMode", ...),
    ("no-hook", "rule_annotations", f"{HOOK}hook", ...),
    ("hook-phase", "rule_annotations", f"{HOOK}hook", "PostSync"),
    ("hook-kept", "rule_annotations", f"{HOOK}hook-delete-policy", "HookSucceeded"),
    ("job-retries", "rule_migrate", "Job/migrate/spec/backoffLimit", 6),
    ("no-migrate", "rule_migrate", "Job/migrate/metadata/name", "other"),
    ("one-replica", "rule_api", "Deployment/api/spec/replicas", 1),
    ("downtime", "rule_api", "Deployment/api/spec/strategy/rollingUpdate/maxUnavailable", 1),
    ("no-api", "rule_api", "Deployment/api/metadata/name", "other"),
]


@pytest.fixture(scope="module")
def good() -> list[dict]:
    return render()


def test_deploy_app_follows_every_rule(good):
    assert checker.check(good) == []


@pytest.mark.parametrize(
    ("rule", "path", "value"), [case[1:] for case in BAD_EDITS], ids=[case[0] for case in BAD_EDITS]
)
def test_a_bad_edit_is_rejected(good, rule, path, value):
    docs = copy.deepcopy(good)
    edit(docs, path, value)
    assert list(getattr(checker, rule)(docs))


def test_every_rule_has_a_bad_edit():
    assert {rule.__name__ for rule in checker.RULES} == {case[1] for case in BAD_EDITS}


def test_generated_names_may_carry_the_kustomize_hash(good):
    docs = copy.deepcopy(good)
    text = yaml.safe_dump_all(docs)
    for name in checker.CONFIG_KEYS:
        text = text.replace(f"name: {name}\n", f"name: {name}-5g8fc84dg8\n")
    assert checker.check(list(yaml.safe_load_all(text))) == []


def test_the_fake_key_is_the_only_literal_credential(good):
    literals = {
        (name, variable["name"], variable["value"])
        for name, container in checker._all_containers(good)
        for variable in container.get("env") or []
        if checker.CREDENTIAL_NAME.search(variable["name"]) and "value" in variable
    }
    key = "local-simulator-only-not-a-real-key"
    assert literals == {(n, "OPENAI_API_KEY", key) for n in ("api", "worker", "rag-worker")}


def test_an_empty_render_is_rejected():
    assert checker.check([])


@pytest.mark.parametrize("path", ["", "Service/api/spec/type"])
def test_the_exit_code_follows_the_rules(good, monkeypatch, path):
    docs = copy.deepcopy(good)
    if path:
        edit(docs, path, "LoadBalancer")
    monkeypatch.setattr(sys, "stdin", io.StringIO(yaml.safe_dump_all(docs)))
    assert checker.main() == bool(path)


def test_placeholders_are_distinct_and_pinned():
    pins = yaml.safe_load((APP / "images" / "kustomization.yaml").read_text())["images"]
    names = [pin["name"] for pin in pins]
    assert sorted(names) == sorted(set(names))
    used = {
        container["image"]
        for path in APP.glob("*.yaml")
        for doc in yaml.safe_load_all(path.read_text())
        if doc["kind"] in checker.WORKLOAD_KINDS
        for container in doc["spec"]["template"]["spec"]["containers"]
    }
    assert {"app", "migrate", "postgres", "simulator"} <= used & set(names)


# The source folder: what could change the render without changing a file above


@pytest.fixture
def folder(tmp_path) -> Path:
    return Path(shutil.copytree(APP, tmp_path / "app"))


SOURCE_EDITS = [
    ("images", "\nimages:\n  - name: app\n    digest: sha256:x\n"),
    ("patches", "\npatches:\n  - patch: x\n"),
    ("replacements", "\nreplacements: []\n"),
    ("transformers", "\ntransformers: []\n"),
    ("generator-options", "\ngeneratorOptions:\n  annotations: {a: b}\n"),
]


@pytest.mark.parametrize("text", [e[1] for e in SOURCE_EDITS], ids=[e[0] for e in SOURCE_EDITS])
def test_an_extra_kustomization_key_is_rejected(folder, text):
    path = folder / "kustomization.yaml"
    path.write_text(path.read_text() + text)
    assert list(checker.rule_source(folder))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  - api.yaml", "  - https://example.com/api.yaml"),
        ("  - api.yaml", "  - github.com/a/b//deploy?ref=main"),
        ("  - api.yaml", "  - ../api.yaml"),
        ("  - api.yaml", "  - other.yaml"),
        ("components:\n  - images", "components:\n  - images\n  - https://example.com/x"),
        ("components:\n  - images", "components: []"),
        ("namespace: fastapi-demo", "namespace: default"),
        ("    literals:\n      - REDIS", "    behavior: merge\n    literals:\n      - REDIS"),
        ("    literals:\n      - REDIS", "    envs: [x.env]\n    literals:\n      - REDIS"),
    ],
)
def test_a_bad_kustomization_is_rejected(folder, old, new):
    path = folder / "kustomization.yaml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    assert list(checker.rule_source(folder))


@pytest.mark.parametrize(
    "name", [".argocd-source.yaml", ".argocd-source-fastapi-demo.yaml", "images/extra.yaml"]
)
def test_an_unlisted_file_is_rejected(folder, name):
    # Argo CD reads .argocd-source*.yaml as Application overrides, images included.
    (folder / name).write_text("kustomize:\n  images: [app=x]\n")
    assert list(checker.rule_source(folder))


def test_a_finder_file_is_ignored(folder):
    for path in (folder / ".DS_Store", folder / "images" / ".DS_Store"):
        path.write_bytes(b"x")
    assert list(checker.rule_source(folder)) == []


def test_a_linked_resource_is_rejected(folder, tmp_path):
    (tmp_path / "other.yaml").write_text((folder / "api.yaml").read_text())
    (folder / "api.yaml").unlink()
    (folder / "api.yaml").symlink_to(tmp_path / "other.yaml")
    assert list(checker.rule_source(folder))


@pytest.mark.parametrize("text", ["images: []", "- a", "", "x: ["])
def test_bad_pins_are_rejected(folder, text):
    (folder / "images" / "kustomization.yaml").write_text(text)
    assert list(checker.rule_source(folder))


# Malformed input: a message, never a traceback


@pytest.mark.parametrize(
    "text",
    [
        "x: [",
        "- a",
        "just text",
        "kind: Deployment",
        "kind: Deployment\nmetadata: null\nspec: 3",
        "kind: Service\nspec: null\nmetadata: []",
        "kind: ConfigMap\ndata: [1]",
    ],
)
def test_malformed_input_fails_with_a_message(monkeypatch, capsys, text):
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert checker.main() == 1
    assert "Traceback" not in capsys.readouterr().err
