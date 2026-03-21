# Fix Off-by-One Error in Pagination

## Problem

The file `workspace/paginator.py` contains a pagination utility used to split a list of items into discrete pages. Pages are **1-indexed** (the first page is page 1), but the current implementation has two off-by-one bugs:

1. **`paginate()` function** — The slice start index is computed as `page * per_page` instead of `(page - 1) * per_page`. This causes page 1 to return the *second* page of results, and the last page's items to be silently dropped.

2. **`total_pages()` function** — The total number of pages is calculated with integer floor-division (`total_items // per_page`), which undercounts by one whenever the items do not divide evenly. For example, 10 items with 3 per page should yield 4 pages, but the current code returns 3.

The `Paginator` class delegates to both of these functions, so fixing them will also fix the class.

## Task

Edit `workspace/paginator.py` to correct both bugs so that:

- `paginate(items, page=1, per_page=3)` returns the first 3 items.
- `paginate(items, page=last, per_page=3)` returns the remaining items on the final page.
- `total_pages(10, 3)` returns `4`.
- `total_pages(9, 3)` returns `3`.

Do **not** change function signatures or remove any existing functionality.
