"""Tests for the pagination module after the off-by-one fix."""

import sys
from pathlib import Path

import pytest

# Ensure the workspace module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workspace"))

from paginator import Paginator, paginate, total_pages


# ---------- paginate() ----------

class TestPaginate:
    def test_first_page(self):
        items = list(range(1, 11))  # [1..10]
        assert paginate(items, page=1, per_page=3) == [1, 2, 3]

    def test_last_page_partial(self):
        items = list(range(1, 11))  # [1..10]
        assert paginate(items, page=4, per_page=3) == [10]

    def test_middle_page(self):
        items = list(range(1, 11))
        assert paginate(items, page=2, per_page=3) == [4, 5, 6]

    def test_empty_items(self):
        assert paginate([], page=1, per_page=5) == []

    def test_page_out_of_range(self):
        items = list(range(1, 6))
        assert paginate(items, page=100, per_page=3) == []

    def test_invalid_page_zero(self):
        assert paginate([1, 2, 3], page=0, per_page=2) == []


# ---------- total_pages() ----------

class TestTotalPages:
    def test_not_evenly_divisible(self):
        assert total_pages(10, 3) == 4

    def test_evenly_divisible(self):
        assert total_pages(9, 3) == 3

    def test_zero_items(self):
        assert total_pages(0, 5) == 0

    def test_single_item(self):
        assert total_pages(1, 10) == 1

    def test_items_equal_per_page(self):
        assert total_pages(5, 5) == 1


# ---------- Paginator class ----------

class TestPaginator:
    def test_num_pages(self):
        p = Paginator(list(range(1, 11)), per_page=3)
        assert p.num_pages == 4

    def test_get_first_page(self):
        p = Paginator(list(range(1, 11)), per_page=3)
        assert p.get_page(1) == [1, 2, 3]

    def test_get_last_page(self):
        p = Paginator(list(range(1, 11)), per_page=3)
        assert p.get_page(4) == [10]

    def test_repr(self):
        p = Paginator([1, 2, 3], per_page=2)
        assert "total_items=3" in repr(p)
        assert "per_page=2" in repr(p)
