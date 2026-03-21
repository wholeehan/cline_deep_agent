"""Minimal Flask item catalogue API."""

from flask import Flask, jsonify

app = Flask(__name__)

ITEMS = [
    {"id": 1, "name": "Widget", "price": 9.99, "category": "tools"},
    {"id": 2, "name": "Gadget", "price": 24.99, "category": "electronics"},
]


@app.route("/items", methods=["GET"])
def list_items():
    """Return every item in the store."""
    return jsonify(ITEMS)


@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    """Return a single item by its ID, or 404."""
    for item in ITEMS:
        if item["id"] == item_id:
            return jsonify(item)
    return jsonify({"error": "Item not found"}), 404


# --- TODO: Add a POST /items endpoint here ---


if __name__ == "__main__":
    app.run(debug=True)
