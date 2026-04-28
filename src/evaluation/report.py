import json
from pathlib import Path
from datetime import datetime


def save_report(metrics: dict, model_name: str, reports_dir: str = "reports"):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "model":     model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics":   metrics,
    }
    path = f"{reports_dir}/{model_name}_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved → {path}")