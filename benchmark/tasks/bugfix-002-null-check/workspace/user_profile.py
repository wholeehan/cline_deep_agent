"""User profile module for managing user display information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Address:
    """Represents a user's physical address."""

    city: str
    state: str
    country: str


@dataclass
class ContactInfo:
    """Represents a user's contact information."""

    email: str
    phone: str


class UserProfile:
    """Stores and presents user profile data.

    Parameters
    ----------
    name : str
        The user's full legal name (always required).
    address : Address | None
        The user's address, if provided.
    contact_info : ContactInfo | None
        The user's contact details, if provided.
    nickname : str | None
        An optional display nickname.
    """

    def __init__(
        self,
        name: str,
        address: Optional[Address] = None,
        contact_info: Optional[ContactInfo] = None,
        nickname: Optional[str] = None,
    ) -> None:
        self.name = name
        self.address = address
        self.contact_info = contact_info
        self.nickname = nickname

    # BUG: does not check whether self.nickname is None before calling .strip()
    def get_display_name(self) -> str:
        """Return the nickname (stripped) if set, otherwise the full name."""
        if self.nickname.strip():
            return self.nickname.strip()
        return self.name

    # BUG: does not check whether self.address is None
    def get_location(self) -> str:
        """Return a human-readable location string."""
        return f"{self.address.city}, {self.address.state}"

    # BUG: does not check whether self.contact_info is None
    def get_primary_email(self) -> Optional[str]:
        """Return the user's primary email address."""
        return self.contact_info.email

    def get_summary(self) -> str:
        """Return a one-line summary combining display name, location, and email."""
        display = self.get_display_name()
        location = self.get_location()
        email = self.get_primary_email()
        return f"{display} | {location} | {email}"
