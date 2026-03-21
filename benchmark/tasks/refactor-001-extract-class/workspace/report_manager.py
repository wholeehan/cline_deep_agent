"""Report manager module -- handles data management and formatting."""


class ReportManager:
    """Manages report data and produces formatted output.

    This class currently violates the Single Responsibility Principle by
    combining data-management logic with formatting concerns.
    """

    def __init__(self, title: str, records: list[dict] | None = None):
        self.title = title
        self.records: list[dict] = list(records) if records else []

    # ---- data methods ----

    def add_record(self, record: dict) -> None:
        """Append a record to the report."""
        self.records.append(record)

    def remove_record(self, index: int) -> None:
        """Remove the record at *index*."""
        del self.records[index]

    def get_record(self, index: int) -> dict:
        """Return the record at *index*."""
        return self.records[index]

    def filter_records(self, key: str, value) -> list[dict]:
        """Return records where *key* equals *value*."""
        return [r for r in self.records if r.get(key) == value]

    def sort_records(self, key: str, reverse: bool = False) -> None:
        """Sort records in place by *key*."""
        self.records.sort(key=lambda r: r[key], reverse=reverse)

    # ---- formatting methods (should be extracted) ----

    def format_as_text(self) -> str:
        """Return a plain-text representation of the report."""
        lines = [self.title, "=" * len(self.title)]
        for i, rec in enumerate(self.records, start=1):
            parts = [f"{k}: {v}" for k, v in rec.items()]
            lines.append(f"  {i}. " + ", ".join(parts))
        return "\n".join(lines)

    def format_as_csv(self) -> str:
        """Return a CSV representation (header + rows)."""
        if not self.records:
            return ""
        headers = list(self.records[0].keys())
        rows = [",".join(headers)]
        for rec in self.records:
            rows.append(",".join(str(rec.get(h, "")) for h in headers))
        return "\n".join(rows)

    def format_as_html_table(self) -> str:
        """Return an HTML table representation."""
        if not self.records:
            return "<table></table>"
        headers = list(self.records[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)
        body_rows = ""
        for rec in self.records:
            cells = "".join(f"<td>{rec.get(h, '')}</td>" for h in headers)
            body_rows += f"<tr>{cells}</tr>"
        return (
            "<table>"
            f"<thead><tr>{header_row}</tr></thead>"
            f"<tbody>{body_rows}</tbody>"
            "</table>"
        )

    def format_summary(self) -> str:
        """Return a short summary of the report."""
        total = len(self.records)
        numeric_keys: list[str] = []
        if self.records:
            numeric_keys = [
                k for k, v in self.records[0].items()
                if isinstance(v, (int, float))
            ]
        summary_lines = [f"Report: {self.title}", f"Total records: {total}"]
        for key in numeric_keys:
            values = [r[key] for r in self.records if key in r]
            if values:
                avg = sum(values) / len(values)
                summary_lines.append(
                    f"  {key}: min={min(values)}, max={max(values)}, avg={avg:.1f}"
                )
        return "\n".join(summary_lines)
