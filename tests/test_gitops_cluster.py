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
    monkeypatch.setattr(script, "_download_manifest", lambda: "argocd: manifest")
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
    ((command, _),) = [call for call in tools.calls if call[1] == "argocd: manifest"]
    assert "--server-side" in command


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
