"""User service module that consumes the API client."""

from api_client import fetch_user


class UserService:
    """Service layer for user-related operations."""

    def get_user_info(self, user_id):
        """Return a summary dict with the user's name and email."""
        data = fetch_user(user_id)
        return {
            "name": data["user_name"],
            "email": data["user_email"],
        }

    def get_user_age(self, user_id):
        """Return the user's age as an integer."""
        data = fetch_user(user_id)
        return data["user_age"]

    def is_adult(self, user_id):
        """Return True if the user is 18 or older."""
        age = self.get_user_age(user_id)
        return age >= 18
