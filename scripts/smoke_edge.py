"""Check the TLS edge (compose.edge.yaml) from the host. Sends no credentials.

    TLS      the certificate verifies against .edge-local/ca.crt for localhost and
             127.0.0.1, and not for other names; TLS 1.2 and 1.3 connect; the server
             refuses TLS 1.1
    port     the API's own plain HTTP port is closed
    headers  nosniff, no-referrer, DENY and a strict CSP on /api/; no HSTS; no version
    limits   a 40 KiB body gets 413; the eval and demo-setup routes get 404
    burst    failed logins for an unknown email reach 429. The address nginx logged
             shows which address its per-address limits see for this client.

The 413, 404 and 429 must carry X-Edge-Reply: 1, so the check fails if the API answered
in nginx's place (the API has its own 32 KiB limit and may have both routes off).

The burst uses up the edge's auth zone for about a minute.
"""

import argparse
import socket
import ssl
import subprocess
import warnings
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

CA_FILE = Path(".edge-local/ca.crt")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_NAMES = ("localhost", "127.0.0.1")
FOREIGN_NAME = "example.com"
LOGIN = "/api/v1/auth/login"
BLOCKED = ("/api/v1/residence/demo", "/api/v1/rag/query/diagnostics")
OVERSIZED_BYTES = 40 * 1024
BURST_ATTEMPTS = 30
CONNECT_SECONDS = 5
DOCKER_SECONDS = 10
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_TOO_LARGE = 413
HTTP_TOO_MANY = 429
SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
}
NOT_TESTABLE = "TLS 1.1 not testable from this Python"
EDGE_REPLY = "x-edge-reply"
VERSIONS = {ssl.TLSVersion.TLSv1_2: "TLSv1.2", ssl.TLSVersion.TLSv1_3: "TLSv1.3"}


def _destination(base_url: str) -> tuple[str, int]:
    parts = urlsplit(base_url)
    if (
        parts.scheme != "https"
        or parts.hostname not in LOOPBACK_HOSTS
        or parts.path not in {"", "/"}
    ):
        raise ValueError("edge smoke requires an https loopback base URL")
    return parts.hostname, parts.port or 443


def _expect(response: httpx.Response, status: int) -> httpx.Response:
    if response.status_code != status:
        raise AssertionError(
            f"{response.request.method} {response.request.url.path}: "
            f"expected {status}, received {response.status_code}"
        )
    return response


def _expect_edge(response: httpx.Response, status: int) -> httpx.Response:
    """The status, answered by nginx itself rather than passed on from the API."""
    _expect(response, status)
    if response.headers.get(EDGE_REPLY) != "1":
        raise AssertionError(
            f"{response.request.method} {response.request.url.path}: "
            f"{status} came from the API, not the edge"
        )
    return response


def _handshake(host: str, port: int, context: ssl.SSLContext, name: str) -> str:
    with socket.create_connection((host, port), timeout=CONNECT_SECONDS) as raw:
        with context.wrap_socket(raw, server_hostname=name) as tls:
            return tls.version()


def _pinned(ca: Path, version: ssl.TLSVersion) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=ca)
    context.minimum_version = version
    context.maximum_version = version
    return context


def _tls11(host: str, port: int, ca: Path) -> str:
    # This client must offer TLS 1.1 at all; then only the server can refuse it.
    if not ssl.HAS_TLSv1_1:
        return NOT_TESTABLE
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(ca)
    try:
        context.set_ciphers("DEFAULT:@SECLEVEL=0")
        # Python deprecates TLS 1.1; it is offered here only to see the server refuse it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            context.minimum_version = ssl.TLSVersion.TLSv1_1
            context.maximum_version = ssl.TLSVersion.TLSv1_1
    except (ssl.SSLError, ValueError):
        return NOT_TESTABLE
    try:
        _handshake(host, port, context, LOCAL_NAMES[0])
    except ssl.SSLError as error:
        if error.reason == "NO_PROTOCOLS_AVAILABLE":
            return NOT_TESTABLE
        return "TLS 1.1 refused"
    except ConnectionError:
        return "TLS 1.1 refused"
    raise AssertionError("TLS 1.1 handshake succeeded")


def _check_tls(host: str, port: int, ca: Path) -> str:
    """Names and versions; returns what happened to TLS 1.1."""
    for version, label in VERSIONS.items():
        context = _pinned(ca, version)
        for name in LOCAL_NAMES:
            negotiated = _handshake(host, port, context, name)
            if negotiated != label:
                raise AssertionError(f"{label} for {name}: negotiated {negotiated}")
    try:
        _handshake(host, port, ssl.create_default_context(cafile=ca), FOREIGN_NAME)
    except ssl.SSLCertVerificationError:
        return _tls11(host, port, ca)
    raise AssertionError(f"certificate verified for {FOREIGN_NAME}")


def _check_closed(port: int) -> None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=CONNECT_SECONDS):
            pass
    except ConnectionRefusedError:
        return
    raise AssertionError(f"plain API port {port} still accepts connections")


def _check_headers(client: httpx.Client) -> None:
    # The API's own 401 carries them too: nginx adds them to errors as well.
    api = _expect(client.get("/api/v1/items"), HTTP_UNAUTHORIZED)
    health = _expect(client.get("/health/live"), HTTP_OK)
    for response in (api, health):
        path = response.request.url.path
        for name, value in SECURITY_HEADERS.items():
            if response.headers.get(name) != value:
                raise AssertionError(f"{path}: header {name} missing or changed")
        if "strict-transport-security" in response.headers:
            raise AssertionError(f"{path}: HSTS is on")
        if response.headers.get("server") != "nginx":
            raise AssertionError(f"{path}: server header names a version or another server")
        if EDGE_REPLY in response.headers:
            raise AssertionError(f"{path}: an API reply is marked as the edge's own")
    if "default-src 'none'" not in api.headers.get("content-security-policy", ""):
        raise AssertionError("/api/ response lacks the strict content policy")


def _check_limits(client: httpx.Client) -> None:
    oversized = client.post(
        "/api/v1/items",
        content=b"0" * OVERSIZED_BYTES,
        headers={"content-type": "application/json"},
    )
    _expect_edge(oversized, HTTP_TOO_LARGE)
    for path in BLOCKED:
        _expect_edge(client.post(path, json={}), HTTP_NOT_FOUND)


def _burst(client: httpx.Client) -> int:
    """Failed logins for an unknown email until the edge answers 429: the attempt number.

    The API records nothing for an unknown email, so no real account is throttled."""
    form = {"username": f"edge-smoke-{uuid4().hex}@example.com", "password": "not-a-password"}
    for attempt in range(1, BURST_ATTEMPTS + 1):
        response = client.post(LOGIN, data=form)
        if response.status_code == HTTP_TOO_MANY:
            _expect_edge(response, HTTP_TOO_MANY)
            return attempt
        _expect(response, HTTP_UNAUTHORIZED)
    raise AssertionError(f"no 429 after {BURST_ATTEMPTS} failed logins")


def _observed_address(project: str, marker: str) -> str | None:
    """The client address nginx logged for the marker request; None if unreadable."""
    command = [
        "docker",
        "ps",
        "-q",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--filter",
        "label=com.docker.compose.service=edge",
    ]
    try:
        containers = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=DOCKER_SECONDS
        ).stdout.split()
        if len(containers) != 1:
            return None
        log = subprocess.run(
            ["docker", "logs", "--tail", "200", containers[0]],
            capture_output=True,
            text=True,
            check=True,
            timeout=DOCKER_SECONDS,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # Only the address field of the marker line is read.
    for line in log.splitlines():
        if marker in line:
            return line.split(" ", 1)[0]
    return None


def _address_note(address: str | None) -> str:
    if address is None:
        return "Edge log not readable: the address nginx keys its limits on is unknown."
    return (
        f"nginx saw this host client as {address}, not its own 127.0.0.1: host clients "
        "share that address, so here the per-address limits cap them together."
    )


def run_smoke(base_url: str, ca: Path, plain_port: int, project: str) -> list[str]:
    host, port = _destination(base_url)
    tls11 = _check_tls(host, port, ca)
    _check_closed(plain_port)
    marker = f"edge-smoke-{uuid4().hex}"
    context = ssl.create_default_context(cafile=ca)
    with httpx.Client(base_url=base_url, verify=context, timeout=10) as client:
        _check_headers(client)
        _check_limits(client)
        attempt = _burst(client)
        client.get(f"/{marker}")
    summary = (
        f"PASS: TLS 1.2 and 1.3 verify for localhost and 127.0.0.1; {tls11}; plain port "
        f"{plain_port} closed; security headers; 413 over 32 KiB; eval and demo routes 404; "
        f"login burst 429 at attempt {attempt}."
    )
    return [summary, _address_note(_observed_address(project, marker))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="https://127.0.0.1:8443")
    parser.add_argument("--ca", type=Path, default=CA_FILE)
    parser.add_argument(
        "--plain-port", type=int, default=8002, help="API port the edge closes (normal: 8000)"
    )
    parser.add_argument("--project", default="fastapi-ai-local", help="Compose project name")
    args = parser.parse_args()
    if not 1 <= args.plain_port <= 65535:
        parser.error("plain port must be 1..65535")
    if not args.ca.is_file():
        parser.error(f"{args.ca} not found: run scripts/init_edge_tls.py first")
    try:
        lines = run_smoke(args.base_url, args.ca, args.plain_port, args.project)
    except ValueError as error:
        parser.error(str(error))
    except AssertionError as error:
        raise SystemExit(f"FAIL: {error}") from None
    except (OSError, httpx.HTTPError) as error:
        raise SystemExit(f"Edge smoke transport failed: {type(error).__name__}") from None
    print("\n".join(lines))


if __name__ == "__main__":
    main()
