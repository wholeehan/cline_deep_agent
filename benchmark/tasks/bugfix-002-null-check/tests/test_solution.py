"""Tests for the UserProfile bug-fix task.

All tests should pass once the null-check bugs have been fixed.
"""

import pytest

from workspace.user_profile import Address, ContactInfo, UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_profile() -> UserProfile:
    return UserProfile(
        name="Alice Smith",
        address=Address(city="Portland", state="OR", country="US"),
        contact_info=ContactInfo(email="alice@example.com", phone="555-0100"),
        nickname="Ali",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullProfile:
    """A fully-populated profile should work correctly."""

    def test_display_name_returns_nickname(self):
        profile = _full_profile()
        assert profile.get_display_name() == "Ali"

    def test_location_returns_city_state(self):
        profile = _full_profile()
        assert profile.get_location() == "Portland, OR"

    def test_primary_email(self):
        profile = _full_profile()
        assert profile.get_primary_email() == "alice@example.com"

    def test_summary_contains_all_parts(self):
        profile = _full_profile()
        summary = profile.get_summary()
        assert "Ali" in summary
        assert "Portland" in summary
        assert "alice@example.com" in summary


class TestNoneAddress:
    """Profiles with address=None must not crash."""

    def test_get_location_does_not_crash(self):
        profile = UserProfile(name="Bob", address=None)
        location = profile.get_location()
        assert isinstance(location, str)
        assert len(location) > 0  # should return a sensible default


class TestNoneContactInfo:
    """Profiles with contact_info=None must not crash."""

    def test_get_primary_email_does_not_crash(self):
        profile = UserProfile(name="Carol", contact_info=None)
        result = profile.get_primary_email()
        # Acceptable defaults: None or empty string
        assert result is None or result == ""


class TestNoneNickname:
    """Profiles with nickname=None should fall back to the full name."""

    def test_display_name_falls_back_to_name(self):
        profile = UserProfile(name="Dana Johnson", nickname=None)
        assert profile.get_display_name() == "Dana Johnson"

    def test_empty_string_nickname_falls_back_to_name(self):
        profile = UserProfile(name="Dana Johnson", nickname="  ")
        assert profile.get_display_name() == "Dana Johnson"


class TestAllOptionalFieldsNone:
    """A profile with every optional field set to None must survive get_summary."""

    def test_get_summary_does_not_crash(self):
        profile = UserProfile(name="Eve", address=None, contact_info=None, nickname=None)
        summary = profile.get_summary()
        assert isinstance(summary, str)
        assert "Eve" in summary
