"""RFC 6238 codes, the drift window, single use, and the otpauth URI."""

import base64
from urllib.parse import parse_qs, urlsplit

import pytest

from app.assistant import totp

RFC_SECRET = b"12345678901234567890"
STEP = 1_000


@pytest.mark.parametrize(
    ("unix_time", "code"),
    [
        (59, "287082"),
        (1111111109, "081804"),
        (1111111111, "050471"),
        (1234567890, "005924"),
        (2000000000, "279037"),
        (20000000000, "353130"),
    ],
)
def test_codes_match_the_rfc_6238_sha1_vectors(unix_time, code):
    # RFC 6238 appendix B lists 8 digits; a 6-digit code is their last six.
    assert totp.code_at(RFC_SECRET, unix_time // totp.STEP_SECONDS) == code


@pytest.mark.parametrize(
    ("offset", "found"),
    [(-2, None), (-1, STEP - 1), (0, STEP), (1, STEP + 1), (2, None)],
)
def test_a_code_matches_one_step_either_side(offset, found):
    code = totp.code_at(RFC_SECRET, STEP + offset)
    assert totp.matching_step(RFC_SECRET, code, STEP, None) == found


def test_a_used_step_never_matches_again():
    code = totp.code_at(RFC_SECRET, STEP)
    assert totp.matching_step(RFC_SECRET, code, STEP, STEP) is None
    assert totp.matching_step(RFC_SECRET, code, STEP, STEP - 1) == STEP
    later = totp.code_at(RFC_SECRET, STEP + 1)
    assert totp.matching_step(RFC_SECRET, later, STEP, STEP) == STEP + 1


def test_a_used_step_does_not_hide_a_newer_one_with_the_same_code(monkeypatch):
    # Every step shares one code: a used step is skipped, not the end of the search.
    monkeypatch.setattr(totp, "code_at", lambda secret, step: "123456")
    assert totp.matching_step(RFC_SECRET, "123456", STEP, STEP - 1) == STEP
    assert totp.matching_step(RFC_SECRET, "123456", STEP, STEP + 1) is None


def test_a_wrong_code_matches_nothing():
    window = {totp.code_at(RFC_SECRET, STEP + offset) for offset in (-1, 0, 1)}
    wrong = next(code for n in range(4) if (code := f"{n:06d}") not in window)
    assert totp.matching_step(RFC_SECRET, wrong, STEP, None) is None


def test_a_secret_is_unpadded_base32_of_twenty_random_bytes():
    secret = totp.new_secret()
    encoded = totp.encode_secret(secret)

    assert (len(secret), len(encoded)) == (20, 32)
    assert "=" not in encoded
    assert base64.b32decode(encoded) == secret
    assert totp.new_secret() != secret


def test_provisioning_uri_names_issuer_account_and_parameters():
    uri = totp.provisioning_uri(RFC_SECRET, "resident@example.com", "Resident assistant")
    parts = urlsplit(uri)

    assert (parts.scheme, parts.netloc) == ("otpauth", "totp")
    assert parts.path == "/Resident%20assistant%3Aresident%40example.com"
    assert "+" not in uri
    assert parse_qs(parts.query) == {
        "secret": [totp.encode_secret(RFC_SECRET)],
        "issuer": ["Resident assistant"],
        "algorithm": ["SHA1"],
        "digits": ["6"],
        "period": ["30"],
    }
