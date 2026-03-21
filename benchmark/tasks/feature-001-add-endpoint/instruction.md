# Task: Add POST /items Endpoint

The Flask app in `workspace/app.py` has endpoints for listing and getting items.

Add a **POST `/items`** endpoint that:

- Accepts a JSON body with the following fields:
  - `name` — **required**, string, must be between 1 and 100 characters
  - `price` — **required**, must be a positive number (greater than zero)
  - `category` — **optional**, string
- Validates all inputs and returns **400** with error details on invalid input
- Adds the item to the in-memory `ITEMS` store with an auto-incremented ID (one greater than the current maximum ID)
- Returns **201** with the created item as JSON
