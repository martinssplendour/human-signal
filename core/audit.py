import csv
import time
from pathlib import Path
from typing import Dict, Optional


def audit_log(action: str, details: Optional[Dict] = None, path: str | Path = "logs/audit.csv") -> None:
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"t_epoch": time.time(), "action": action, "details": details or {}}
    new_file = not audit_path.exists()
    with audit_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_epoch", "action", "details"])
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_healthcare_report(path: str | Path, metadata: Dict, summary: Dict, event_path: Optional[Path]) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Healthcare Observation Report",
        "",
        "For observation support only. This report is not diagnostic and is not a medical device output.",
        "",
        "## Session Metadata",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Summary"])
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Files", f"- Event CSV: {event_path or 'not exported'}", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    audit_log("healthcare_report_exported", {"path": str(report_path), "event_csv": str(event_path) if event_path else None})
    return report_path
