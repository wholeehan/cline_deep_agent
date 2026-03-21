"""Tests for the POST /items endpoint."""

import sys
import os
import json

import pytest

# Make the workspace app importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workspace"))

from app import app, ITEMS  # noqa: E402


@pytest.fixture(autouse=True)
def reset_items():
    """Reset the in-memory store before each test."""
    ITEMS.clear()
    ITEMS.extend([
        {"id": 1, "name": "Widget", "price": 9.99, "category": "tools"},
        {"id": 2, "name": "Gadget", "price": 24.99, "category": "electronics"},
    ])


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_post_valid_item_returns_201(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "Doohickey", "price": 5.50, "category": "misc"}),
        content_type="application/json",
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert "id" in body
    assert body["name"] == "Doohickey"
    assert body["price"] == 5.50
    assert body["category"] == "misc"


def test_post_missing_name_returns_400(client):
    resp = client.post(
        "/items",
        data=json.dumps({"price": 10.0}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_post_empty_name_returns_400(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "", "price": 10.0}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_post_negative_price_returns_400(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "Thing", "price": -1}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_post_zero_price_returns_400(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "Thing", "price": 0}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_post_without_category_succeeds(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "NoCat", "price": 1.00}),
        content_type="application/json",
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["name"] == "NoCat"


def test_created_item_appears_in_get(client):
    client.post(
        "/items",
        data=json.dumps({"name": "Visible", "price": 3.00}),
        content_type="application/json",
    )
    resp = client.get("/items")
    items = resp.get_json()
    names = [i["name"] for i in items]
    assert "Visible" in names


def test_auto_incremented_id(client):
    resp = client.post(
        "/items",
        data=json.dumps({"name": "Third", "price": 7.77}),
        content_type="application/json",
    )
    body = resp.get_json()
    assert body["id"] == 3
