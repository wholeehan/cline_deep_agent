"""API client module (v1) for fetching user data."""

USERS = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 17},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 25},
}


def fetch_user(user_id):
    """Fetch a user by ID and return v1 formatted data."""
    user = USERS.get(user_id)
    if user is None:
        raise ValueError(f"User with id {user_id} not found")
    return {
        "user_name": user["name"],
        "user_email": user["email"],
        "user_age": user["age"],
    }
