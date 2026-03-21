"""Display helpers that format user data for output."""

from api_client import fetch_user


def format_user_card(user_id):
    """Return a human-readable user card string."""
    data = fetch_user(user_id)
    name = data["user_name"]
    email = data["user_email"]
    age = data["user_age"]
    return f"{name} ({age}) <{email}>"


def format_user_email(user_id):
    """Return a formatted 'Name <email>' string."""
    data = fetch_user(user_id)
    name = data["user_name"]
    email = data["user_email"]
    return f"{name} <{email}>"
