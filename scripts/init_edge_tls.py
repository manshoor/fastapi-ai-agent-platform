"""Create the TLS edge's local CA certificate and its localhost server certificate.

    .edge-local/      0700, current owner, not a symlink
      ca.crt          local CA: give it to clients explicitly (curl --cacert, httpx verify)
      server.crt      localhost, 127.0.0.1 and ::1; valid 397 days
      server.key      P-256 key, 0600

The CA key is never written: the CA signs this one certificate and is gone. Its name
constraints allow only localhost (and names under it), 127.0.0.1 and ::1, so it cannot
vouch for any other name, even from a trust store. Existing files are never replaced; to
renew, delete the directory and run this again.
"""

import argparse
import ipaddress
import os
import stat
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CA_FILE = "ca.crt"
CERT_FILE = "server.crt"
KEY_FILE = "server.key"
FILES = (CA_FILE, CERT_FILE, KEY_FILE)
# The longest server certificate lifetime browsers accept.
LIFETIME = timedelta(days=397)
# Room for a client clock that runs a little behind this one.
BACKDATE = timedelta(minutes=5)
HOSTNAMES = ("localhost",)
ADDRESSES = ("127.0.0.1", "::1")
DIRECTORY_MODE = 0o700
KEY_MODE = 0o600
CERT_MODE = 0o644


class TlsResult(StrEnum):
    CREATED = "created"
    UNCHANGED = "unchanged"


def _require_private(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != DIRECTORY_MODE
    ):
        raise ValueError("unsafe edge directory: require a current-owner nonsymlink 0700 directory")


def _present(root_fd: int) -> list[str]:
    present = []
    for name in FILES:
        try:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        # Kept files must be the current owner's single-link regular files; the key private.
        mode_ok = name != KEY_FILE or stat.S_IMODE(metadata.st_mode) == KEY_MODE
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or not mode_ok
        ):
            raise ValueError(f"unsafe edge file {name}: require current-owner regular files")
        present.append(name)
    return present


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _builder(subject: x509.Name, issuer: x509.Name, public_key, now: datetime):
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + LIFETIME)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
    )


def _issue() -> tuple[bytes, bytes, bytes]:
    """CA certificate, server certificate and server key, PEM-encoded."""
    now = datetime.now(UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    server_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("Local TLS edge CA")
    hostnames = [x509.DNSName(name) for name in HOSTNAMES]

    # The CA: one level deep, local names only, signing certificates only.
    networks = [x509.IPAddress(ipaddress.ip_network(address)) for address in ADDRESSES]
    ca = (
        _builder(ca_name, ca_name, ca_key.public_key(), now)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.NameConstraints(permitted_subtrees=hostnames + networks, excluded_subtrees=None),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # The server certificate nginx presents.
    addresses = [x509.IPAddress(ipaddress.ip_address(address)) for address in ADDRESSES]
    server = (
        _builder(_name(HOSTNAMES[0]), ca_name, server_key.public_key(), now)
        .add_extension(x509.SubjectAlternativeName(hostnames + addresses), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = server_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pem = serialization.Encoding.PEM
    return ca.public_bytes(pem), server.public_bytes(pem), key


def _write(root_fd: int, name: str, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(name, flags, mode, dir_fd=root_fd), "wb") as handle:
        handle.write(data)


def initialize(directory: Path) -> TlsResult:
    # A single dedicated child avoids attacker-controlled parent symlinks and keeps the
    # key apart from normal .env, artifacts and AI directories.
    if len(directory.parts) != 1 or not directory.name.startswith(".edge-"):
        raise ValueError("directory must be a single project child name beginning .edge-")
    try:
        directory.mkdir(mode=DIRECTORY_MODE)
    except FileExistsError:
        pass
    _require_private(directory.lstat())
    root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require_private(os.fstat(root_fd))

        # All three files kept as they are, or none: a partial set is never completed.
        present = _present(root_fd)
        if len(present) == len(FILES):
            return TlsResult.UNCHANGED
        if present:
            raise ValueError("partial edge TLS files: delete the directory and run again")

        # The key first: a failure after it leaves a partial set, refused on the next run.
        ca, certificate, key = _issue()
        _write(root_fd, KEY_FILE, key, KEY_MODE)
        _write(root_fd, CERT_FILE, certificate, CERT_MODE)
        _write(root_fd, CA_FILE, ca, CERT_MODE)
        return TlsResult.CREATED
    finally:
        os.close(root_fd)


def _valid_until(directory: Path) -> datetime:
    certificate = x509.load_pem_x509_certificate((directory / CERT_FILE).read_bytes())
    return certificate.not_valid_after_utc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--directory", type=Path, default=Path(".edge-local"))
    args = parser.parse_args()
    try:
        result = initialize(args.directory)
        valid_until = _valid_until(args.directory)
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        parser.error(f"cannot prepare edge TLS files: {error.strerror}")

    # An expired certificate is kept, not replaced: renewal is an explicit delete.
    day = valid_until.date().isoformat()
    if valid_until <= datetime.now(UTC):
        raise SystemExit(f"Server certificate expired on {day}: delete {args.directory}/ and rerun")
    action = "Created" if result == TlsResult.CREATED else "Kept existing"
    print(f"{action} edge TLS files in {args.directory}/; server certificate valid until {day}.")


if __name__ == "__main__":
    main()
