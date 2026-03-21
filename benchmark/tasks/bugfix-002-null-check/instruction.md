# Bug Fix: Missing Null Checks in UserProfile

The `UserProfile` class in `workspace/user_profile.py` crashes with
`AttributeError` when accessing nested fields on users with incomplete data.

The `get_display_name()`, `get_location()`, and `get_primary_email()` methods
do not handle `None` values for nested objects (`address`, `contact_info`,
`nickname`).

## Steps to reproduce

```python
from user_profile import UserProfile

user = UserProfile(name="Alice", address=None, contact_info=None, nickname=None)
user.get_summary()  # AttributeError
```

## What to fix

Fix **all** methods in `UserProfile` to safely handle `None` or missing data
and return sensible defaults:

- `get_display_name()` should fall back to `name` when `nickname` is `None` or
  empty.
- `get_location()` should return a sensible default (e.g. `"Unknown"`) when
  `address` is `None`.
- `get_primary_email()` should return `None` (or `""`) when `contact_info` is
  `None`.
- `get_summary()` should work without crashing regardless of which optional
  fields are missing.
