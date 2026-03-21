"""Pagination utilities for splitting sequences into discrete pages.

Pages are 1-indexed: the first page is ``page=1``.
"""

from __future__ import annotations

from typing import List, TypeVar

T = TypeVar("T")


def paginate(items: List[T], page: int, per_page: int) -> List[T]:
    """Return a single page of *items*.

    Parameters
    ----------
    items:
        The full collection to paginate.
    page:
        The 1-based page number to retrieve.
    per_page:
        Maximum number of items per page.

    Returns
    -------
    List[T]
        The slice of *items* corresponding to the requested page.
        Returns an empty list when *page* is out of range or *items*
        is empty.
    """
    if page < 1 or per_page < 1:
        return []

    # BUG: should be (page - 1) * per_page for 1-indexed pages
    start = page * per_page
    end = start + per_page
    return items[start:end]


def total_pages(total_items: int, per_page: int) -> int:
    """Calculate how many pages are needed to hold *total_items*.

    Parameters
    ----------
    total_items:
        Total number of items in the collection.
    per_page:
        Maximum number of items per page.

    Returns
    -------
    int
        The number of pages (>= 0).
    """
    if total_items <= 0 or per_page < 1:
        return 0

    # BUG: floor division drops the remainder — should use ceiling division
    return total_items // per_page


class Paginator:
    """Convenience wrapper around :func:`paginate` and :func:`total_pages`."""

    def __init__(self, items: List[T], per_page: int = 10) -> None:
        self.items = list(items)
        self.per_page = per_page

    @property
    def num_pages(self) -> int:
        """Total number of pages."""
        return total_pages(len(self.items), self.per_page)

    def get_page(self, page: int) -> List[T]:
        """Return items for the given 1-based *page* number."""
        return paginate(self.items, page, self.per_page)

    def __repr__(self) -> str:
        return (
            f"Paginator(total_items={len(self.items)}, "
            f"per_page={self.per_page}, num_pages={self.num_pages})"
        )
