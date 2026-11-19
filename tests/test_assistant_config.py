"""Assistant settings: unlock step-up needs a 32-byte sealing key while the assistant runs."""

import base64

import pytest
from pydantic import ValidationError

from app.assistant.config import AssistantSettings

VALID_KEY = base64.b64encode(bytes(range(32))).decode()


def build(**values) -> AssistantSettings:
    return AssistantSettings(ai_enabled=True, rag_enabled=True, assistant_enabled=True, **values)


def test_a_32_byte_key_is_accepted():
    settings = build(assistant_totp_key=VALID_KEY)
    assert settings.assistant_totp_key.get_secret_value() == VALID_KEY
    assert VALID_KEY not in repr(settings)


@pytest.mark.parametrize(
    "key",
    [
        None,
        "",
        base64.b64encode(bytes(31)).decode(),
        base64.b64encode(bytes(33)).decode(),
        "not*base64*at*all*",
        "é" * 44,
    ],
    ids=["missing", "empty", "short", "long", "not-base64", "non-ascii"],
)
def test_assistant_needs_a_32_byte_totp_key(monkeypatch, key):
    monkeypatch.delenv("ASSISTANT_TOTP_KEY", raising=False)
    values = {} if key is None else {"assistant_totp_key": key}
    with pytest.raises(ValidationError) as error:
        build(**values)
    assert "ASSISTANT_TOTP_KEY must hold 32 bytes encoded as base64" in str(error.value)
    if key:
        assert key not in str(error.value)


def test_no_key_is_needed_while_the_assistant_is_off(monkeypatch):
    monkeypatch.delenv("ASSISTANT_TOTP_KEY", raising=False)
    settings = AssistantSettings(ai_enabled=True, rag_enabled=True)
    assert settings.assistant_totp_key is None


@pytest.mark.parametrize("issuer", ["Acme: Homes", "", "x" * 65])
def test_issuer_is_short_and_has_no_colon(issuer):
    with pytest.raises(ValidationError):
        build(assistant_totp_key=VALID_KEY, assistant_totp_issuer=issuer)
