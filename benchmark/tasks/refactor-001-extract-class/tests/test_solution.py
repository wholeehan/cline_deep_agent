"""Tests for the refactor-001-extract-class benchmark task.

Validates that formatting responsibilities have been extracted from
ReportManager into a new ReportFormatter class while preserving all
original behaviour.
"""

import importlib
import os
import sys

import pytest

# Make the workspace importable.
WORKSPACE = os.path.join(
    os.path.dirname(__file__), os.pardir, "workspace"
)
sys.path.insert(0, os.path.abspath(WORKSPACE))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"name": "Alice", "score": 85, "grade": "B"},
    {"name": "Bob", "score": 92, "grade": "A"},
    {"name": "Charlie", "score": 78, "grade": "C"},
]


@pytest.fixture()
def sample_records():
    return [dict(r) for r in SAMPLE_RECORDS]


# ---------------------------------------------------------------------------
# 1. ReportFormatter class exists in report_formatter.py
# ---------------------------------------------------------------------------

def test_report_formatter_module_exists():
    mod = importlib.import_module("report_formatter")
    assert hasattr(mod, "ReportFormatter"), (
        "report_formatter.py must export a ReportFormatter class"
    )


# ---------------------------------------------------------------------------
# 2. ReportFormatter has all four formatting methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    ["format_as_text", "format_as_csv", "format_as_html_table", "format_summary"],
)
def test_report_formatter_has_formatting_methods(method):
    from report_formatter import ReportFormatter
    assert callable(getattr(ReportFormatter, method, None)), (
        f"ReportFormatter must have a callable '{method}' method"
    )


# ---------------------------------------------------------------------------
# 3. ReportManager still has all data methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    ["add_record", "remove_record", "get_record", "filter_records", "sort_records"],
)
def test_report_manager_keeps_data_methods(method):
    from report_manager import ReportManager
    assert callable(getattr(ReportManager, method, None)), (
        f"ReportManager must still have a callable '{method}' method"
    )


# ---------------------------------------------------------------------------
# 4. ReportManager no longer owns formatting implementations directly
# ---------------------------------------------------------------------------

def test_report_manager_does_not_own_formatting():
    """ReportManager should delegate formatting or not define them at all."""
    import inspect
    from report_manager import ReportManager

    formatting_methods = [
        "format_as_text", "format_as_csv", "format_as_html_table", "format_summary",
    ]
    for name in formatting_methods:
        attr = getattr(ReportManager, name, None)
        if attr is not None:
            # If the method still exists, its source should show delegation
            # (i.e. it should reference ReportFormatter or a formatter attribute).
            source = inspect.getsource(attr)
            assert "ReportFormatter" in source or "formatter" in source.lower(), (
                f"ReportManager.{name} should delegate to ReportFormatter, "
                "not contain its own implementation"
            )


# ---------------------------------------------------------------------------
# 5. format_as_csv produces correct output
# ---------------------------------------------------------------------------

def test_format_as_csv_output(sample_records):
    from report_formatter import ReportFormatter

    formatter = _make_formatter(sample_records)
    csv_output = formatter.format_as_csv()
    lines = csv_output.strip().splitlines()
    assert lines[0] == "name,score,grade", "CSV header must match record keys"
    assert len(lines) == len(sample_records) + 1, "CSV must have header + one row per record"
    assert "Alice,85,B" in lines[1]


# ---------------------------------------------------------------------------
# 6. format_as_html_table contains <table> tags
# ---------------------------------------------------------------------------

def test_format_as_html_table_output(sample_records):
    formatter = _make_formatter(sample_records)
    html = formatter.format_as_html_table()
    assert "<table>" in html and "</table>" in html, "HTML output must contain <table> tags"
    assert "<th>" in html, "HTML output must contain <th> header cells"
    assert "<td>" in html, "HTML output must contain <td> data cells"


# ---------------------------------------------------------------------------
# 7. Data operations still work after refactoring
# ---------------------------------------------------------------------------

def test_data_operations_work(sample_records):
    from report_manager import ReportManager

    mgr = ReportManager("Test", sample_records)
    mgr.add_record({"name": "Diana", "score": 90, "grade": "A"})
    assert len(mgr.records) == 4

    mgr.remove_record(0)
    assert mgr.records[0]["name"] == "Bob"

    assert mgr.get_record(0)["name"] == "Bob"

    filtered = mgr.filter_records("grade", "A")
    assert len(filtered) == 2  # Bob and Diana

    mgr.sort_records("score")
    assert mgr.records[0]["score"] <= mgr.records[-1]["score"]


# ---------------------------------------------------------------------------
# 8. Formatting results match original behaviour
# ---------------------------------------------------------------------------

def test_formatting_results_match_original(sample_records):
    """The extracted formatter must produce the same output as the original."""
    formatter = _make_formatter(sample_records)

    # CSV check
    csv = formatter.format_as_csv()
    assert "name,score,grade" in csv
    assert "Bob,92,A" in csv

    # Summary check
    summary = formatter.format_summary()
    assert "Total records: 3" in summary

    # Text check
    text = formatter.format_as_text()
    assert "Alice" in text and "Charlie" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formatter(records):
    """Create a ReportFormatter however the solution exposes it."""
    from report_formatter import ReportFormatter

    # The formatter needs a title and records.  Accept common constructor
    # signatures: (title, records) or (report_manager).
    import inspect
    sig = inspect.signature(ReportFormatter.__init__)
    params = list(sig.parameters.keys())
    if len(params) >= 3:  # self, title, records
        return ReportFormatter("Test Report", list(records))
    else:
        # Possibly accepts a ReportManager instance
        from report_manager import ReportManager
        mgr = ReportManager("Test Report", list(records))
        return ReportFormatter(mgr)
