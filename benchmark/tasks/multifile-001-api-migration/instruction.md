# Migrate API v1 to v2 across 3 files

The codebase uses an old `api_client.py` (v1) with the function `fetch_user(user_id)` that returns a dict with keys `"user_name"` (str), `"user_email"` (str), and `"user_age"` (int).

The API has been updated to v2. The function is renamed to `get_user(user_id)` and now returns keys **without** the `"user_"` prefix: `"name"` (str), `"email"` (str), `"age"` (int).

Update all 3 files in the workspace:

1. **`workspace/api_client.py`** — Rename the function `fetch_user` to `get_user` and update the returned dict keys from `"user_name"`, `"user_email"`, `"user_age"` to `"name"`, `"email"`, `"age"`.
2. **`workspace/user_service.py`** — Update all call sites to use the new function name `get_user` and the new field names (`"name"`, `"email"`, `"age"`).
3. **`workspace/user_display.py`** — Update all call sites to use the new function name `get_user` and the new field names.
