"""Every check_deploy.py rule accepts deploy/app and rejects a bad edit. No cluster."""

import copy
import importlib.util
import io
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
    if value is ...:
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
APP_IMAGE = checker.REGISTRY
BAD_EDITS = [
    ("secret-kind", "rule_kinds", "ConfigMap/api-settings/kind", "Secret"),
    ("ingress-kind", "rule_kinds", "Service/api/kind", "Ingress"),
    ("other-group", "rule_kinds", "Deployment/api/apiVersion", "evil/v1"),
    ("other-namespace", "rule_kinds", "Service/api/metadata/namespace", "default"),
    ("extra-workload", "rule_workloads", "Deployment/api/metadata/name", "other"),
    ("no-wave", "rule_waves", "Deployment/api/metadata/annotations", ...),
    (
        "early-api",
        "rule_waves",
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
    ("node-port", "rule_services", "Service/api/spec/type", "NodePort"),
    ("external-ip", "rule_services", "Service/api/spec/externalIPs", ["1.2.3.4"]),
    ("pruned-db", "rule_database", "StatefulSet/db/metadata/annotations", {}),
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
    ("no-hook", "rule_migrate", f"{HOOK}hook", ...),
    ("hook-phase", "rule_migrate", f"{HOOK}hook", "PostSync"),
    ("hook-kept", "rule_migrate", f"{HOOK}hook-delete-policy", "HookSucceeded"),
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
    assert {"app", "migrate", "postgres"} <= used & set(names)
