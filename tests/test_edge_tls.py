"""scripts/init_edge_tls.py: a private CA limited to local names, and a localhost leaf."""

import asyncio
import contextlib
import ipaddress
import os
import runpy
import ssl
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ROOT = Path(__file__).resolve().parent.parent
EDGE = Path(".edge-local")
FILES = ["ca.crt", "server.crt", "server.key"]


@pytest.fixture
def script(tmp_path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    return runpy.run_path(str(ROOT / "scripts/init_edge_tls.py"))


def certificate(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


@contextlib.asynccontextmanager
async def tls_server(cert: Path, key: Path):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    listener = await asyncio.start_server(
        lambda reader, writer: writer.close(), "127.0.0.1", 0, ssl=context
    )
    try:
        yield listener.sockets[0].getsockname()[1]
    finally:
        listener.close()


async def handshake(port: int, ca: Path, name: str) -> None:
    """A handshake under Python's default context: strict X.509 checks, as httpx makes."""
    client = ssl.create_default_context(cafile=ca)
    _, writer = await asyncio.open_connection("127.0.0.1", port, ssl=client, server_hostname=name)
    writer.close()
    with contextlib.suppress(ssl.SSLError, ConnectionError):
        await writer.wait_closed()


def test_creates_a_name_constrained_ca_and_a_localhost_certificate(script, tmp_path):
    assert script["initialize"](EDGE) == "created"
    root = tmp_path / EDGE
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert sorted(path.name for path in root.iterdir()) == FILES
    assert stat.S_IMODE((root / "server.key").stat().st_mode) == 0o600

    # The leaf: local names only, server authentication only, 397 days plus the backdate.
    ca, leaf = certificate(root / "ca.crt"), certificate(root / "server.crt")
    leaf.verify_directly_issued_by(ca)
    names = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert names.get_values_for_type(x509.DNSName) == ["localhost"]
    assert [str(ip) for ip in names.get_values_for_type(x509.IPAddress)] == ["127.0.0.1", "::1"]
    assert list(leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [
        ExtendedKeyUsageOID.SERVER_AUTH
    ]
    lifetime = leaf.not_valid_after_utc - leaf.not_valid_before_utc
    assert lifetime == timedelta(days=397, minutes=5)

    # The CA: one level deep.
    basic = ca.extensions.get_extension_for_class(x509.BasicConstraints)
    assert (basic.critical, basic.value.ca, basic.value.path_length) == (True, True, 0)


async def test_the_certificate_verifies_for_local_names_only(script, tmp_path):
    script["initialize"](EDGE)
    root = tmp_path / EDGE
    async with tls_server(root / "server.crt", root / "server.key") as port:
        for name in ("localhost", "127.0.0.1"):
            await handshake(port, root / "ca.crt", name)
        with pytest.raises(ssl.SSLCertVerificationError):
            await handshake(port, root / "ca.crt", "example.com")


@pytest.mark.parametrize(
    ("name", "hostname", "accepted"),
    [
        (x509.DNSName("localhost"), "localhost", True),
        (x509.DNSName("example.com"), "example.com", False),
        (x509.IPAddress(ipaddress.ip_address("127.0.0.2")), "127.0.0.2", False),
    ],
    ids=["localhost", "other-name", "other-address"],
)
async def test_the_ca_cannot_vouch_for_other_names(
    script, tmp_path, monkeypatch, name, hostname, accepted
):
    # Keep the CA key, which the script never writes, to sign leaves it would never issue.
    generated = []
    original = ec.generate_private_key

    def capture(curve):
        generated.append(original(curve))
        return generated[-1]

    monkeypatch.setattr(ec, "generate_private_key", capture)
    script["initialize"](EDGE)
    root = tmp_path / EDGE
    ca = certificate(root / "ca.crt")
    ca_key = next(key for key in generated if key.public_key() == ca.public_key())

    # A leaf like the real one, but for `name`.
    key, now = original(ec.SECP256R1()), datetime.now(UTC)
    usage = dict.fromkeys(
        (
            "content_commitment",
            "key_encipherment",
            "data_encipherment",
            "key_agreement",
            "key_cert_sign",
            "crl_sign",
            "encipher_only",
            "decipher_only",
        ),
        False,
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([name]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, **usage), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    (tmp_path / "rogue.crt").write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "rogue.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    async with tls_server(tmp_path / "rogue.crt", tmp_path / "rogue.key") as port:
        if accepted:
            await handshake(port, root / "ca.crt", hostname)
            return
        with pytest.raises(ssl.SSLCertVerificationError, match="subtree"):
            await handshake(port, root / "ca.crt", hostname)


def test_existing_files_are_kept_and_a_partial_set_is_refused(script, tmp_path):
    assert script["initialize"](EDGE) == "created"
    root = tmp_path / EDGE
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    assert script["initialize"](EDGE) == "unchanged"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before

    (root / "server.crt").unlink()
    with pytest.raises(ValueError, match="partial"):
        script["initialize"](EDGE)
    assert sorted(path.name for path in root.iterdir()) == ["ca.crt", "server.key"]


@pytest.mark.parametrize(
    "unsafe",
    [
        "nested",
        "other-prefix",
        "open-directory",
        "directory-symlink",
        "open-key",
        "key-symlink",
        "key-hardlink",
    ],
)
def test_unsafe_paths_are_refused(script, tmp_path, unsafe):
    root = tmp_path / EDGE
    directory = EDGE
    if unsafe == "nested":
        directory = Path("child") / EDGE
    elif unsafe == "other-prefix":
        directory = Path(".tls-local")
    elif unsafe == "open-directory":
        root.mkdir(mode=0o700)
        root.chmod(0o755)
    elif unsafe == "directory-symlink":
        target = tmp_path / "elsewhere"
        target.mkdir(mode=0o700)
        root.symlink_to(target)
    else:
        script["initialize"](EDGE)
        key = root / "server.key"
        if unsafe == "open-key":
            key.chmod(0o644)
        elif unsafe == "key-symlink":
            key.rename(tmp_path / "moved.key")
            key.symlink_to(tmp_path / "moved.key")
        else:
            os.link(key, tmp_path / "linked.key")

    with pytest.raises(ValueError):
        script["initialize"](directory)
    # Nothing new was written anywhere.
    assert not (tmp_path / "child").exists()
    assert not (tmp_path / ".tls-local").exists()
    if unsafe in {"open-directory", "directory-symlink"}:
        assert os.listdir(root) == []
