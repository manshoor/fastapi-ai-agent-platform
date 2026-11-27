"""GitOps cluster script: guard, Secrets and Argo CD files. No cluster, no network."""

import base64
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load("gitops_cluster")
checker = _load("check_deploy")
COMMIT = "a" * 40
ARGOCD = "quay.io/argoproj/argocd:v3.6.1"
REDIS = "public.ecr.aws/docker/library/redis:8.10.1-alpine"
# The shape of Argo CD's workloads: an image line, sometimes a policy line after it.
MANIFEST = f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: argocd-redis
spec:
  template:
    spec:
      initContainers:
      - command: [argocd, admin]
        image: {ARGOCD}
        imagePullPolicy: IfNotPresent
        name: secret-init
      containers:
      - args: [--save, ""]
        image: {REDIS}
        imagePullPolicy: Always
        name: redis
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: argocd-application-controller
spec:
  template:
    spec:
      containers:
      - image: {ARGOCD}
        name: argocd-application-controller
"""
REPOSITORY = "https://github.com/manshoor/fastapi-ai-agent-platform.git"


class FakeTools:
    """Records commands; answers from (substring, output) pairs, newest first."""

    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []
        self.answers: list[tuple[str, object]] = [
            ("config view", json.dumps(kube_config("https://127.0.0.1:6443"))),
            ("docker context inspect", "unix:///var/run/docker.sock\n"),
        ]

    def __call__(self, command, stdin="", timeout=0):
        self.calls.append((command, stdin))
        line = " ".join(command)
        for part, answer in reversed(self.answers):
            if part in line:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return ""

    def lines(self) -> list[str]:
        return [" ".join(command) for command, _ in self.calls]


def kube_config(server: str, **extra) -> dict:
    return {"clusters": [{"name": "x", "cluster": {"server": server, **extra}}]}


@pytest.fixture
def tools(monkeypatch, tmp_path) -> FakeTools:
    for name in ("DOCKER_HOST", "KIND_EXPERIMENTAL_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(script, "KUBECONFIG", tmp_path / "private" / "kubeconfig")
    fake = FakeTools()
    monkeypatch.setattr(script, "run", fake)
    monkeypatch.setattr(script, "_download_manifest", lambda: MANIFEST)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    return fake


# Guard


@pytest.mark.parametrize("server", ["https://127.0.0.1:6443", "https://[::1]:6443"])
def test_a_loopback_server_is_accepted(server):
    script.check_server(kube_config(server))


@pytest.mark.parametrize(
    "config",
    [
        kube_config("https://10.0.0.1:6443"),
        kube_config("https://example.com"),
        kube_config("https://127.0.0.1.example.com"),
        kube_config("https://127.0.0.1@example.com"),
        kube_config("https://localhost:6443"),
        kube_config("https://0.0.0.0:6443"),
        kube_config("https://127.0.0.1:6443", **{"proxy-url": "http://example.com"}),
        {"clusters": []},
        {"clusters": None},
        {"clusters": kube_config("https://127.0.0.1")["clusters"] * 2},
        {"clusters": [{"cluster": {}}]},
    ],
)
def test_any_other_server_is_refused(config):
    with pytest.raises(script.ClusterError):
        script.check_server(config)


def test_up_changes_nothing_on_a_remote_server(tools):
    tools.answers.append(("config view", json.dumps(kube_config("https://example.com"))))
    with pytest.raises(script.ClusterError):
        script.up(None)
    assert not [line for line in tools.lines() if " apply" in line or " create -f" in line]


def test_every_kubectl_call_names_the_context(tools):
    script.up(COMMIT)
    calls = [command for command, _ in tools.calls if command[0] == "kubectl"]
    assert calls
    for command in calls:
        assert command[1:3] == ["--kubeconfig", str(script.KUBECONFIG)]
        assert command[3:5] == ["--context", "kind-fastapi-gitops"]
        assert command.count("--kubeconfig") == 1
        assert command.count("--context") == 1


def test_kind_only_touches_its_own_cluster(tools):
    script.up(None)
    script.down()
    calls = [line for line in tools.lines() if line.startswith("kind ") and "get" not in line]
    assert len(calls) == 2
    assert all("cluster --name fastapi-gitops" in line for line in calls)


def test_an_existing_cluster_is_kept(tools):
    tools.answers.append(("kind get clusters", "other\nfastapi-gitops\n"))
    script.up(None)
    assert not [line for line in tools.lines() if line.startswith("kind create")]


def test_kind_never_writes_the_users_kubeconfig(tools):
    script.up(None)
    script.down()
    tools.answers.append(("kind get clusters", "fastapi-gitops"))
    script.up(None)
    calls = [c for c, _ in tools.calls if c[0] == "kind" and c[1] != "get"]
    assert {c[1] for c in calls} == {"create", "delete", "export"}
    for command in calls:
        assert command[command.index("--kubeconfig") + 1] == str(script.KUBECONFIG)
    assert script.KUBECONFIG.parent.stat().st_mode & 0o777 == 0o700


def _make_link(path: Path) -> None:
    path.parent.joinpath("elsewhere").mkdir(mode=0o700)
    path.symlink_to(path.parent / "elsewhere")


UNSAFE_FOLDERS = {
    "open": lambda path: path.mkdir(mode=0o755),
    "link": _make_link,
    "file": lambda path: path.write_text(""),
}


@pytest.mark.parametrize("make", UNSAFE_FOLDERS.values(), ids=UNSAFE_FOLDERS)
@pytest.mark.parametrize(
    "act", [lambda: script.up(None), lambda: script.down()], ids=["up", "down"]
)
def test_an_unsafe_private_folder_is_refused(tools, act, make):
    make(script.KUBECONFIG.parent)
    with pytest.raises(script.ClusterError, match="private"):
        act()
    assert not [line for line in tools.lines() if line.startswith("kind ")]


def test_another_users_folder_is_refused(tools, monkeypatch):
    script.KUBECONFIG.parent.mkdir(mode=0o700)
    owner = script.KUBECONFIG.parent.stat().st_uid
    monkeypatch.setattr(script.os, "getuid", lambda: owner + 1)
    with pytest.raises(script.ClusterError, match="private"):
        script.down()


def test_down_works_before_up_and_after(tools):
    script.down()
    script.down()
    assert script.KUBECONFIG.parent.stat().st_mode & 0o777 == 0o700


REMOTE = ["tcp://127.0.0.1:2375", "ssh://user@example.com", "tcp://example.com:2376", "", "x"]


@pytest.mark.parametrize("host", REMOTE)
@pytest.mark.parametrize("act", [lambda: script.up(None), script.down], ids=["up", "down"])
def test_a_remote_docker_context_is_refused(tools, act, host):
    tools.answers.append(("docker context inspect", host + "\n"))
    with pytest.raises(script.ClusterError, match="Docker"):
        act()
    assert not [line for line in tools.lines() if not line.startswith("docker context")]


@pytest.mark.parametrize("host", [host for host in REMOTE if host])
@pytest.mark.parametrize("act", [lambda: script.up(None), script.down], ids=["up", "down"])
def test_a_remote_docker_host_is_refused(tools, monkeypatch, act, host):
    monkeypatch.setenv("DOCKER_HOST", host)
    with pytest.raises(script.ClusterError, match="Docker"):
        act()
    assert tools.calls == []


def test_another_kind_provider_is_refused(tools, monkeypatch):
    # It would use its own endpoint, which the Docker check does not cover.
    monkeypatch.setenv("KIND_EXPERIMENTAL_PROVIDER", "podman")
    for act in (lambda: script.up(None), script.down):
        with pytest.raises(script.ClusterError, match="provider"):
            act()
    assert tools.calls == []


def test_a_local_docker_host_is_accepted(tools, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    script.down()
    assert [line for line in tools.lines() if line.startswith("kind delete")]


# Secrets


def test_missing_secrets_are_created_over_stdin(tools):
    script.up(None)
    created = [(command, stdin) for command, stdin in tools.calls if "create" in command]
    created = [call for call in created if call[0][0] == "kubectl"]
    assert len(created) == len(script.SECRETS)
    for command, stdin in created:
        assert command[-3:] == ["create", "-f", "-"]
        for value in json.loads(stdin)["stringData"].values():
            assert not [line for line in tools.lines() if value in line]


def test_an_existing_secret_is_never_written(tools):
    tools.answers.append(("get secret db-credentials", "secret/db-credentials"))
    tools.answers.append(("get secret db-credentials --ignore-not-found --output go", "password\n"))
    script.ensure_secret("db-credentials")
    assert not [line for line in tools.lines() if "create" in line or "apply" in line]


def test_an_incomplete_secret_is_reported_not_repaired(tools):
    tools.answers.append(("get secret api-secrets", "secret/api-secrets"))
    tools.answers.append(("--output go-template", "jwt-secret\n"))
    with pytest.raises(script.ClusterError, match="memory-key, totp-key"):
        script.ensure_secret("api-secrets")
    assert not [line for line in tools.lines() if "create" in line or "apply" in line]


def test_a_failed_create_shows_no_value(tools):
    tools.answers.append(("create -f -", script.ClusterError("kubectl failed: leaked")))
    with pytest.raises(script.ClusterError) as failure:
        script.ensure_secret("db-credentials")
    assert "leaked" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_secret_values_have_the_shape_the_app_needs():
    values = script.secret_manifest("api-secrets")["stringData"]
    assert len(bytes.fromhex(values["jwt-secret"])) == 32
    totp, memory = (
        base64.b64decode(values[key], validate=True) for key in ("totp-key", "memory-key")
    )
    assert len(totp) == len(memory) == 32
    assert totp != memory
    assert values != script.secret_manifest("api-secrets")["stringData"]


def test_secret_readers_match_the_created_secrets():
    assert set(checker.SECRET_READERS) == set(script.SECRETS)


# Namespace, Argo CD and the Application


def test_the_namespace_enforces_the_restricted_standard():
    (namespace,) = [
        item for item in script.namespace_manifests() if item["metadata"]["name"] == "fastapi-demo"
    ]
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"


def test_the_namespace_and_secrets_exist_before_the_application(tools):
    script.up(None)
    inputs = [stdin for _, stdin in tools.calls]
    (application,) = [i for i, text in enumerate(inputs) if '"Application"' in text]
    others = [i for i, text in enumerate(inputs) if '"Namespace"' in text or '"Secret"' in text]
    assert others and max(others) < application


def test_argocd_is_applied_server_side(tools):
    script.up(None)
    ((command, _),) = [call for call in tools.calls if "argocd-redis" in call[1]]
    assert "--server-side" in command


# Argo CD's images


def _pulls(tools: FakeTools) -> list[str]:
    return [line for line in tools.lines() if " crictl " in line]


def _applied_containers(tools: FakeTools) -> list[dict]:
    (manifest,) = [stdin for _, stdin in tools.calls if "argocd-redis" in stdin]
    containers = []
    for document in yaml.safe_load_all(manifest):
        spec = document["spec"]["template"]["spec"]
        containers += spec.get("initContainers", []) + spec["containers"]
    return containers


def _digest(reference: str) -> str:
    return script.ARGOCD_IMAGES[reference].digest


def test_argocd_images_are_pulled_into_the_node_before_argocd(tools):
    script.up(None)
    lines = tools.lines()
    pulls = _pulls(tools)
    assert pulls == [
        "docker exec fastapi-gitops-control-plane crictl pull"
        f" --pull-timeout {script.PULL_SECONDS}s {name}@{_digest(tag)}"
        for tag, name in (
            (ARGOCD, "quay.io/argoproj/argocd"),
            (REDIS, "public.ecr.aws/docker/library/redis"),
        )
    ]
    argocd = next(i for i, (_, stdin) in enumerate(tools.calls) if "argocd-redis" in stdin)
    created = next(i for i, line in enumerate(lines) if line.startswith("kind create"))
    assert created < lines.index(pulls[0]) and lines.index(pulls[-1]) < argocd


def test_argocd_runs_only_the_pulled_digests(tools):
    script.up(None)
    containers = _applied_containers(tools)
    assert [c["name"] for c in containers] == [
        "secret-init",
        "redis",
        "argocd-application-controller",
    ]
    for container in containers:
        # Never: a pod can only start from the image pulled above, not from a registry.
        assert container["imagePullPolicy"] == "Never"
        assert re.fullmatch(r"[\w./-]+@sha256:[0-9a-f]{64}", container["image"])
    assert containers[1]["image"] == f"public.ecr.aws/docker/library/redis@{_digest(REDIS)}"


def test_a_failed_pull_is_retried_then_the_next_source_is_used(tools):
    tools.answers.append(("public.ecr.aws/docker/library/redis@", script.ClusterError("429")))
    script.up(None)
    redis = [line for line in _pulls(tools) if "redis" in line]
    attempts = script.PULL_ATTEMPTS
    assert [line.split()[-1] for line in redis] == [
        f"public.ecr.aws/docker/library/redis@{_digest(REDIS)}"
    ] * attempts + [f"docker.io/library/redis@{_digest(REDIS)}"]
    assert _applied_containers(tools)[1]["image"] == f"docker.io/library/redis@{_digest(REDIS)}"


def test_argocd_is_not_applied_if_an_image_cannot_be_pulled(tools):
    tools.answers.append(("crictl pull", script.ClusterError("429")))
    with pytest.raises(script.ClusterError, match="could not be pulled"):
        script.up(None)
    sources = len(script.ARGOCD_IMAGES[ARGOCD].sources)
    assert len(_pulls(tools)) == script.PULL_ATTEMPTS * sources
    assert not [stdin for _, stdin in tools.calls if "argocd-redis" in stdin]


def test_each_pull_has_its_own_time_limit(tools):
    script.up(None)
    assert _pulls(tools)
    for command, _ in tools.calls:
        if "crictl" in command:
            assert command[3:6] == ["crictl", "pull", "--pull-timeout"]
            assert command[6] == f"{script.PULL_SECONDS}s"


@pytest.mark.parametrize(
    "manifest",
    [
        MANIFEST.replace(REDIS, "docker.io/library/redis:latest"),
        MANIFEST.replace(f"        image: {REDIS}\n", ""),
        MANIFEST.replace(f"image: {ARGOCD}", f"image: {ARGOCD}@sha256:{'0' * 64}"),
    ],
    ids=["unpinned", "missing", "other-digest"],
)
def test_a_manifest_with_other_images_is_refused(manifest):
    pulled = {tag: f"x@{image.digest}" for tag, image in script.ARGOCD_IMAGES.items()}
    with pytest.raises(script.ClusterError, match="Argo CD images"):
        script.pin_images(manifest, pulled)


def test_argocd_images_are_pinned_by_digest():
    assert set(script.ARGOCD_IMAGES) == {ARGOCD, REDIS}
    for tag, image in script.ARGOCD_IMAGES.items():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", image.digest)
        # The manifest's own registry comes first; a mirror only after it.
        assert image.sources[0] == tag.rsplit(":", 1)[0]
        assert len(set(image.sources)) == len(image.sources) <= 2
    assert ARGOCD.rsplit(":", 1)[1] in script.ARGOCD_URL


def test_the_pulls_fit_the_start_step():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "gitops.yml").read_text())
    (step,) = [
        s for s in workflow["jobs"]["cluster"]["steps"] if s.get("name") == "Start the cluster"
    ]
    attempts = sum(len(image.sources) for image in script.ARGOCD_IMAGES.values())
    attempts *= script.PULL_ATTEMPTS
    worst = attempts * (script.PULL_SECONDS + script.PULL_WAIT_SECONDS)
    # Half the step: the cluster and Argo CD's rollout need the rest.
    assert worst <= step["timeout-minutes"] * 60 / 2


def test_the_argocd_url_names_an_exact_version():
    assert "/argo-cd/v3.6.1/" in script.ARGOCD_URL
    assert script.ARGOCD_URL.startswith("https://")


def test_another_argocd_manifest_is_refused():
    with pytest.raises(script.ClusterError, match="SHA-256"):
        script.check_manifest(b"other")


def test_a_revision_replaces_only_the_target():
    committed = yaml.safe_load(script.APPLICATION_FILE.read_text())
    assert script.application(None) == committed
    committed["spec"]["source"]["targetRevision"] = COMMIT
    assert script.application(COMMIT) == committed


@pytest.mark.parametrize("revision", ["main", "", "A" * 40, "a" * 39, "a" * 40 + "\n", "--x"])
def test_a_bad_revision_is_refused_before_any_command(tools, revision):
    with pytest.raises(script.ClusterError):
        script.up(revision)
    assert tools.calls == []


def test_the_project_limits_what_git_can_deliver():
    spec = yaml.safe_load(script.PROJECT_FILE.read_text())["spec"]
    assert spec["sourceRepos"] == [REPOSITORY]
    assert spec["destinations"] == [
        {"server": "https://kubernetes.default.svc", "namespace": "fastapi-demo"}
    ]
    assert spec["clusterResourceWhitelist"] == []
    allowed = {(item["group"], item["kind"]) for item in spec["namespaceResourceWhitelist"]}
    assert allowed == {(version.rpartition("/")[0], kind) for version, kind in checker.KINDS}
    assert len(allowed) == 6
    assert set(spec) == {
        "sourceRepos",
        "destinations",
        "clusterResourceWhitelist",
        "namespaceResourceWhitelist",
    }


def test_the_application_repairs_prunes_last_and_retries_a_little():
    spec = yaml.safe_load(script.APPLICATION_FILE.read_text())["spec"]
    assert spec["project"] == "fastapi-demo"
    assert spec["source"] == {"repoURL": REPOSITORY, "targetRevision": "main", "path": "deploy/app"}
    assert spec["destination"]["namespace"] == script.NAMESPACE == checker.NAMESPACE
    policy = spec["syncPolicy"]
    assert policy["automated"] == {"prune": True, "selfHeal": True}
    assert policy["syncOptions"] == ["PruneLast=true"]
    assert 0 < policy["retry"]["limit"] <= 3


def test_the_kind_node_is_pinned_and_local():
    config = yaml.safe_load(script.KIND_CONFIG.read_text())
    assert config["networking"]["apiServerAddress"] == "127.0.0.1"
    (node,) = config["nodes"]
    assert re.fullmatch(r"kindest/node(:v[\d.]+)?@sha256:[0-9a-f]{64}", node["image"])
