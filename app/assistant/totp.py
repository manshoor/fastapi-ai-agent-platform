"""Time-based one-time codes (RFC 6238: HMAC-SHA-1, 6 digits, 30-second steps).

Standard library only, so scripts can load this file by path.

    secret (20 random bytes) -> base32 for the authenticator app (otpauth URI)
    code for step s          =  HOTP(secret, s) mod 10^6
    a submitted code matches -> a step within s-1..s+1 that is later than the last used
                                one; the earliest such step wins
"""

import base64
import hashlib
import hmac
import secrets
import struct
from urllib.parse import quote, urlencode

STEP_SECONDS = 30
DIGITS = 6
# Clock drift allowed between the phone and the server, in steps either side.
WINDOW_STEPS = 1
SECRET_BYTES = 20
_MODULUS = 10**DIGITS


def new_secret() -> bytes:
    return secrets.token_bytes(SECRET_BYTES)


def encode_secret(secret: bytes) -> str:
    """Base32 without padding, as authenticator apps expect."""
    return base64.b32encode(secret).decode("ascii").rstrip("=")


def code_at(secret: bytes, step: int) -> str:
    """The code for one time step (RFC 4226 dynamic truncation)."""
    digest = hmac.new(secret, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(value % _MODULUS).zfill(DIGITS)


def matching_step(secret: bytes, code: str, step: int, last_used: int | None) -> int | None:
    """The earliest step in the window whose code equals `code` and that was never used.

    Every candidate is compared in constant time, so timing does not tell which one
    matched. A used step is skipped rather than ending the search: when a used step and a
    newer one share a code, the newer one still matches.
    """
    found = None
    for candidate in range(step - WINDOW_STEPS, step + WINDOW_STEPS + 1):
        matches = hmac.compare_digest(code_at(secret, candidate), code)
        unused = last_used is None or candidate > last_used
        if matches and unused and found is None:
            found = candidate
    return found


def provisioning_uri(secret: bytes, account: str, issuer: str) -> str:
    """otpauth:// URI for a QR code or manual entry."""
    label = quote(f"{issuer}:{account}", safe="")
    query = urlencode(
        {
            "secret": encode_secret(secret),
            "issuer": issuer,
            "algorithm": "SHA1",
            "digits": DIGITS,
            "period": STEP_SECONDS,
        },
        # %20, not +, for spaces: some authenticator apps keep a + as it is.
        quote_via=quote,
    )
    return f"otpauth://totp/{label}?{query}"
