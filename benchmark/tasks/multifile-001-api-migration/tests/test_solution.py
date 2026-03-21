"""Tests that verify the v1 -> v2 API migration was applied correctly."""

import sys
import os
import inspect

# Make the workspace root importable (tests run from workspace/)
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, WORKSPACE)

import api_client  # noqa: E402
from user_service import UserService  # noqa: E402
import user_display  # noqa: E402


# ---------- api_client checks ----------

def test_get_user_exists():
    """api_client must expose a 'get_user' function."""
    assert hasattr(api_client, "get_user"), "api_client should have a 'get_user' function"
    assert callable(api_client.get_user)


def test_fetch_user_removed():
    """The old 'fetch_user' function must no longer exist in api_client."""
    assert not hasattr(api_client, "fetch_user"), (
        "api_client should NOT have the old 'fetch_user' function"
    )


def test_get_user_returns_v2_keys():
    """get_user must return dict with keys 'name', 'email', 'age' (no 'user_' prefix)."""
    result = api_client.get_user(1)
    assert "name" in result, "Expected key 'name' in get_user result"
    assert "email" in result, "Expected key 'email' in get_user result"
    assert "age" in result, "Expected key 'age' in get_user result"
    for old_key in ("user_name", "user_email", "user_age"):
        assert old_key not in result, f"Old key '{old_key}' should not be in get_user result"


# ---------- user_service checks ----------

def test_user_service_get_user_info():
    """UserService.get_user_info must work with the v2 API."""
    svc = UserService()
    info = svc.get_user_info(1)
    assert info["name"] == "Alice"
    assert info["email"] == "alice@example.com"


def test_user_service_is_adult():
    """UserService.is_adult must work with the v2 API."""
    svc = UserService()
    assert svc.is_adult(1) is True   # Alice is 30
    assert svc.is_adult(2) is False  # Bob is 17


def test_user_service_no_old_keys():
    """user_service.py source must not reference old 'user_name'/'user_email'/'user_age' keys."""
    src = inspect.getsource(UserService)
    for old_key in ("user_name", "user_email", "user_age"):
        assert old_key not in src, (
            f"user_service.py still references old key '{old_key}'"
        )


# ---------- user_display checks ----------

def test_format_user_card():
    """format_user_card must work with the v2 API."""
    card = user_display.format_user_card(1)
    assert "Alice" in card
    assert "30" in card
    assert "alice@example.com" in card


def test_format_user_email():
    """format_user_email must work with the v2 API."""
    result = user_display.format_user_email(1)
    assert result == "Alice <alice@example.com>"


def test_user_display_no_old_keys():
    """user_display.py source must not reference old 'user_name'/'user_email'/'user_age' keys."""
    src_file = os.path.join(os.path.abspath(WORKSPACE), "user_display.py")
    with open(src_file) as f:
        src = f.read()
    for old_key in ("user_name", "user_email", "user_age"):
        assert old_key not in src, (
            f"user_display.py still references old key '{old_key}'"
        )
