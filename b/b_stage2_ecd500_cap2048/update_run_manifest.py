"""Record Stage2 wall time, process status, and sampled GPU memory peak."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("state", choices=("running", "finished"))
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    path = args.run_dir / "run_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if args.state == "running":
        data["status"] = "running"
        data["start_utc"] = utc_now()
    else:
        if args.exit_code is None or args.wall_seconds is None:
            parser.error("finished requires --exit-code and --wall-seconds")
        data["status"] = "completed" if args.exit_code == 0 else "failed"
        data["end_utc"] = utc_now()
        data["wall_seconds"] = args.wall_seconds
        data["process_exit_code"] = args.exit_code
        telemetry = args.run_dir / "gpu_telemetry.csv"
        samples = []
        if telemetry.is_file():
            with telemetry.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    try:
                        samples.append({"used_mib": int(row["used_mib"]),
                                        "utilization_pct": int(row["utilization_pct"])})
                    except (ValueError, KeyError):
                        continue
        data["gpu_samples"] = len(samples)
        data["gpu_peak_used_mib"] = max((sample["used_mib"] for sample in samples), default=None)
        data["gpu_peak_measurement"] = "nvidia-smi memory.used sampled approximately every 5 seconds"
        data["gpu_max_utilization_pct"] = max((sample["utilization_pct"] for sample in samples), default=None)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"status": data["status"], "run_manifest": str(path),
                      "gpu_peak_used_mib": data.get("gpu_peak_used_mib")}, sort_keys=True))


if __name__ == "__main__":
    main()
